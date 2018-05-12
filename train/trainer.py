# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import json

import tensorflow as tf
from tensorflow.python.client import timeline

from environment.environment import Environment
from model.model import UnrealModel
from train.experience import Experience, ExperienceFrame

LOG_INTERVAL = 200
PERFORMANCE_LOG_INTERVAL = 1000
LOSS_AND_EVAL_LOG_INTERVAL = 1000

GPU_LOG = False # Change main.py

if GPU_LOG:
  run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True, trace_level=tf.RunOptions.FULL_TRACE)
else:
  run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

class Trainer(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               env_type,
               env_name,
               use_lstm,
               use_pixel_change,
               use_value_replay,
               use_reward_prediction,
               pixel_change_lambda,
               entropy_beta,
               local_t_max,
               n_step_TD,
               gamma,
               gamma_pc,
               experience_history_size,
               max_global_time_step,
               device,
               segnet_param_dict,
               image_shape,
               is_training,
               n_classes,
               random_state,
               termination_time,
               segnet_lambda,
               dropout):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.env_type = env_type
    self.env_name = env_name
    self.use_lstm = use_lstm
    self.use_pixel_change = use_pixel_change
    self.use_value_replay = use_value_replay
    self.use_reward_prediction = use_reward_prediction
    self.local_t_max = local_t_max
    self.n_step_TD = n_step_TD
    self.gamma = gamma
    self.gamma_pc = gamma_pc
    self.experience_history_size = experience_history_size
    self.max_global_time_step = max_global_time_step
    self.action_size = Environment.get_action_size(env_type, env_name)
    self.objective_size = Environment.get_objective_size(env_type, env_name)

    self.segnet_param_dict = segnet_param_dict
    self.segnet_mode = self.segnet_param_dict.get("segnet_mode", None)

    self.is_training = is_training
    self.n_classes = n_classes
    self.segnet_lambda = segnet_lambda

    self.run_metadata = tf.RunMetadata()
    self.many_runs_timeline = TimeLiner()

    self.random_state = random_state
    self.termination_time = termination_time
    self.dropout = dropout

    try:
      self.local_network = UnrealModel(self.action_size,
                                       self.objective_size,
                                       thread_index,
                                       use_lstm,
                                       use_pixel_change,
                                       use_value_replay,
                                       use_reward_prediction,
                                       pixel_change_lambda,
                                       entropy_beta,
                                       device,
                                       segnet_param_dict=self.segnet_param_dict ,
                                       image_shape=image_shape,
                                       is_training=is_training,
                                       n_classes=n_classes,
                                       segnet_lambda=self.segnet_lambda,
                                       dropout=dropout)

      self.local_network.prepare_loss()

      self.apply_gradients = grad_applier.minimize_local(self.local_network.total_loss,
                                                         global_network.get_vars(),
                                                         self.local_network.get_vars(), self.thread_index)

      self.sync = self.local_network.sync_from(global_network)
      self.experience = Experience(self.experience_history_size, random_state=self.random_state)
      self.local_t = 0
      self.initial_learning_rate = initial_learning_rate
      self.episode_reward = 0
      # For log output
      self.prev_local_t = 0
      self.prev_local_t_loss = 0
    except Exception as e:
      print(str(e), flush=True)
      raise Exception("Problem in Trainer {} initialization".format(thread_index))

  def prepare(self, termination_time=50.0, termination_dist_value=-10.0):
    self.environment = Environment.create_environment(self.env_type,
                                                      self.env_name, self.termination_time,
                                                      thread_index=self.thread_index)

  def stop(self):
    self.environment.stop()
    
  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  
  def choose_action(self, pi_values):
    return self.random_state.choice(len(pi_values), p=pi_values)

  
  def _record_one(self, sess, summary_writer, summary_op, score_input, score, global_t):
    if self.thread_index >= 0:
      summary_str = sess.run(summary_op, feed_dict={
        score_input: score
      })
      summary_writer.add_summary(summary_str, global_t)

  def _record_all(self, sess, summary_writer, summary_op,
                    dict_input, dict_eval, global_t):
    if self.thread_index >= 0:
      assert set(dict_input.keys()) == set(dict_eval.keys()), print(dict_input.keys(), dict_eval.keys())

      feed_dict = {}
      for key in dict_input.keys():
        feed_dict.update({dict_input[key]: dict_eval[key]})
      summary_str = sess.run(summary_op, feed_dict=feed_dict)

      summary_writer.add_summary(summary_str, global_t)

    
  def set_start_time(self, start_time):
    self.start_time = start_time


  def _fill_experience(self, sess):
    """
    Fill experience buffer until buffer is full.
    """
    #print("Start experience filling", flush=True)
    prev_state = self.environment.last_state
    last_action = self.environment.last_action
    last_reward = self.environment.last_reward
    last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                  self.action_size,
                                                                  last_reward, prev_state)

    #print("Local network run base policy, value!", flush=True)
    pi_, _, _ = self.local_network.run_base_policy_and_value(sess,
                                                              self.environment.last_state,
                                                              last_action_reward)
    action = self.choose_action(pi_)
    
    new_state, reward, terminal, pixel_change = self.environment.process(action)

    frame = ExperienceFrame({key: val for key, val in prev_state.items() if 'objectType' not in key},
                            reward, action, terminal, pixel_change,
                            last_action, last_reward)
    self.experience.add_frame(frame)
    
    if terminal:
      self.environment.reset()
    if self.experience.is_full():
      self.environment.reset()
      print("Replay buffer filled")


  def _print_log(self, global_t):
    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))#, flush=True)
      # print("### Experience : {}".format(self.experience.get_debug_string()))


  def _process_base(self, sess, global_t, summary_writer, summary_op_dict, summary_dict):#, losses_input):
    # [Base A3C]
    states = []
    last_action_rewards = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    start_lstm_state = None
    if self.use_lstm:
      start_lstm_state = self.local_network.base_lstm_state_out


    mode = "segnet" if self.segnet_mode >= 2 else ""
    # t_max times loop

    for _ in range(self.n_step_TD):
      # Prepare last action reward
      last_action = self.environment.last_action
      last_reward = self.environment.last_reward
      last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                    self.action_size,
                                                                    last_reward, self.environment.last_state)
      
      pi_, value_, losses = self.local_network.run_base_policy_and_value(sess,
                                                                 self.environment.last_state,
                                                                 last_action_reward, mode)
      
      action = self.choose_action(pi_)

      states.append(self.environment.last_state)
      last_action_rewards.append(last_action_reward)
      actions.append(action)
      values.append(value_)

      if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
        print("Trainer {}>>> Local step {}:".format(self.thread_index, self.local_t))
        print("Trainer {}>>> pi={}".format(self.thread_index, pi_))
        print("Trainer {}>>> V={}".format(self.thread_index, value_), flush=True)

      prev_state = self.environment.last_state

      # Process game
      new_state, reward, terminal, pixel_change = self.environment.process(action)
      frame = ExperienceFrame({key: val for key, val in prev_state.items() if 'objectType' not in key},
                              reward, action, terminal, pixel_change,
                              last_action, last_reward)

      # Store to experience
      self.experience.add_frame(frame)

      # Use to know about Experience collection
      #print(self.experience.get_debug_string())

      self.episode_reward += reward
      rewards.append(reward)
      self.local_t += 1

      if terminal:
        terminal_end = True
        print("Trainer {}>>> score={}".format(self.thread_index, self.episode_reward))#, flush=True)

        summary_dict['values'].update({'score_input': self.episode_reward})
          
        self.episode_reward = 0
        self.environment.reset()
        self.local_network.reset_state()
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_base_value(sess, new_state, frame.get_action_reward(self.action_size))

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_adv = []
    batch_R = []
    batch_sobjT = []

    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + self.gamma * R
      adv = R - Vi
      a = np.zeros([self.action_size])
      a[ai] = 1.0

      batch_si.append(si['image'])
      batch_a.append(a)
      batch_adv.append(adv)
      batch_R.append(R)
      if self.segnet_param_dict["segnet_mode"] >= 2:
        batch_sobjT.append(si['objectType'])

    batch_si.reverse()
    batch_a.reverse()
    batch_adv.reverse()
    batch_R.reverse()
    batch_sobjT.reverse()

    #print(np.unique(batch_sobjT))

    ## HERE Mathematical Error A3C: only last values should be used for base/ or aggregate with last made

    return batch_si, batch_sobjT, last_action_rewards, batch_a, batch_adv, batch_R, start_lstm_state

  
  def _process_pc(self, sess):
    # [pixel change]
    # Sample 20+1 frame (+1 for last next state)
    #print(">>> Process run!", flush=True)
    pc_experience_frames = self.experience.sample_sequence(self.local_t_max+1)
    # Reverse sequence to calculate from the last
    # pc_experience_frames.reverse()
    pc_experience_frames = pc_experience_frames[::-1]
    #print(">>> Process ran!", flush=True)

    batch_pc_si = []
    batch_pc_a = []
    batch_pc_R = []
    batch_pc_last_action_reward = []
    
    pc_R = np.zeros([20,20], dtype=np.float32)
    if not pc_experience_frames[1].terminal:
      pc_R = self.local_network.run_pc_q_max(sess,
                                             pc_experience_frames[0].state,
                                             pc_experience_frames[0].get_last_action_reward(self.action_size))

    #print(">>> Process run!", flush=True)

    for frame in pc_experience_frames[1:]:

      pc_R = frame.pixel_change + self.gamma_pc * pc_R
      a = np.zeros([self.action_size])
      a[frame.action] = 1.0
      last_action_reward = frame.get_last_action_reward(self.action_size)

      batch_pc_si.append(frame.state['image'])
      batch_pc_a.append(a)
      batch_pc_R.append(pc_R)
      batch_pc_last_action_reward.append(last_action_reward)

    batch_pc_si.reverse()
    batch_pc_a.reverse()
    batch_pc_R.reverse()
    batch_pc_last_action_reward.reverse()

    #print(">>> Process ended!", flush=True)
    return batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R

  
  def _process_vr(self, sess):
    # [Value replay]
    # Sample 20+1 frame (+1 for last next state)
    vr_experience_frames = self.experience.sample_sequence(self.local_t_max+1)
    # Reverse sequence to calculate from the last
    vr_experience_frames.reverse()

    batch_vr_si = []
    batch_vr_R = []
    batch_vr_last_action_reward = []

    vr_R = 0.0
    if not vr_experience_frames[1].terminal:
      vr_R = self.local_network.run_vr_value(sess,
                                             vr_experience_frames[0].state,
                                             vr_experience_frames[0].get_last_action_reward(self.action_size))
    
    # t_max times loop
    for frame in vr_experience_frames[1:]:
      vr_R = frame.reward + self.gamma * vr_R
      batch_vr_si.append(frame.state['image'])
      batch_vr_R.append(vr_R)
      last_action_reward = frame.get_last_action_reward(self.action_size)
      batch_vr_last_action_reward.append(last_action_reward)

    batch_vr_si.reverse()
    batch_vr_R.reverse()
    batch_vr_last_action_reward.reverse()

    return batch_vr_si, batch_vr_last_action_reward, batch_vr_R

  
  def _process_rp(self):
    # [Reward prediction]
    rp_experience_frames = self.experience.sample_rp_sequence()
    # 4 frames

    batch_rp_si = []
    batch_rp_c = []
    
    for i in range(3):
      batch_rp_si.append(rp_experience_frames[i].state['image'])

    # one hot vector for target reward
    r = rp_experience_frames[3].reward
    rp_c = [0.0, 0.0, 0.0]
    if -1e-10 < r < 1e-10:
      rp_c[0] = 1.0 # zero
    elif r > 0:
      rp_c[1] = 1.0 # positive
    else:
      rp_c[2] = 1.0 # negative
    batch_rp_c.append(rp_c)
    return batch_rp_si, batch_rp_c

  def process(self, sess, global_t, summary_writer, summary_op_dict,
              score_input, eval_input, entropy_input, losses_input):

    if self.prev_local_t == 0 and self.segnet_mode >= 2:
      self.prev_local_t = self.local_t
      sess.run(self.local_network.reset_evaluation_vars)
    # Fill experience replay buffer
    #print("Inside train process of thread!", flush=True)
    if not self.experience.is_full():
      self._fill_experience(sess)
      return 0

    start_local_t = self.local_t

    cur_learning_rate = self._anneal_learning_rate(global_t)

    #print("Weights copying!", flush=True)
    # Copy weights from shared to local
    sess.run( self.sync )
    #print("Weights copied successfully!", flush=True)

    summary_dict = {'placeholders': {}, 'values': {}}
    summary_dict['placeholders'].update(losses_input)

    # [Base]
    #print("[Base]", flush=True)
    batch_si, batch_sobjT, batch_last_action_rewards, batch_a, batch_adv, batch_R, start_lstm_state,  = \
          self._process_base(sess,
                             global_t,
                             summary_writer,
                             summary_op_dict,
                             summary_dict)
    if summary_dict['values'].get('score_input', None) is not None:
      self._record_one(sess, summary_writer, summary_op_dict['score_input'], score_input,
                       summary_dict['values']['score_input'], self.local_t)
      summary_writer.flush()
      summary_dict['values'] = {}

    feed_dict = {
      self.local_network.base_input: batch_si,
      self.local_network.base_last_action_reward_input: batch_last_action_rewards,
      self.local_network.base_a: batch_a,
      self.local_network.base_adv: batch_adv,
      self.local_network.base_r: batch_R,
      # [common]
      self.learning_rate_input: cur_learning_rate,
      self.is_training: True
    }

    if self.use_lstm:
      feed_dict[self.local_network.base_initial_lstm_state] = start_lstm_state

    if self.segnet_param_dict["segnet_mode"] >= 2:
      feed_dict[self.local_network.base_segm_mask] = batch_sobjT

    #print("[Pixel change]", flush=True)
    # [Pixel change]
    if self.use_pixel_change:
      batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R = self._process_pc(sess)

      pc_feed_dict = {
        self.local_network.pc_input: batch_pc_si,
        self.local_network.pc_last_action_reward_input: batch_pc_last_action_reward,
        self.local_network.pc_a: batch_pc_a,
        self.local_network.pc_r: batch_pc_R
      }
      feed_dict.update(pc_feed_dict)

    #print("[Value replay]", flush=True)
    # [Value replay]
    if self.use_value_replay:
      batch_vr_si, batch_vr_last_action_reward, batch_vr_R = self._process_vr(sess)
      
      vr_feed_dict = {
        self.local_network.vr_input: batch_vr_si,
        self.local_network.vr_last_action_reward_input : batch_vr_last_action_reward,
        self.local_network.vr_r: batch_vr_R
      }
      feed_dict.update(vr_feed_dict)

    # [Reward prediction]
    #print("[Reward prediction]", flush=True)
    if self.use_reward_prediction:
      batch_rp_si, batch_rp_c = self._process_rp()
      rp_feed_dict = {
        self.local_network.rp_input: batch_rp_si,
        self.local_network.rp_c_target: batch_rp_c
      }
      feed_dict.update(rp_feed_dict)
      #print(len(batch_rp_c), batch_rp_c)


    grad_check = None
    if self.local_t - self.prev_local_t_loss >= LOSS_AND_EVAL_LOG_INTERVAL:
      grad_check = [tf.add_check_numerics_ops()]
    #print("Applying gradients in train!", flush=True)
    # Calculate gradients and copy them to global network.
    out_list = [self.apply_gradients]
    out_list += [self.local_network.total_loss,
                 self.local_network.base_loss, self.local_network.policy_loss,
                 self.local_network.value_loss, self.local_network.entropy]
    if self.segnet_mode >= 2:
      out_list += [self.local_network.decoder_loss]
      out_list += [self.local_network.regul_loss]
    if self.use_pixel_change:
      out_list += [self.local_network.pc_loss]
    if self.use_value_replay:
      out_list += [self.local_network.vr_loss]
    if self.use_reward_prediction:
      out_list += [self.local_network.rp_loss]
    if self.segnet_mode >= 2:
      out_list += [self.local_network.update_evaluation_vars]
      if self.local_t - self.prev_local_t_loss >= LOSS_AND_EVAL_LOG_INTERVAL:
        out_list += [self.local_network.evaluation]

    with tf.control_dependencies(grad_check):
      if GPU_LOG:
        return_list = sess.run(out_list,
                  feed_dict=feed_dict, options=run_options, run_metadata=self.run_metadata)
      else:
        return_list = sess.run(out_list,
                  feed_dict=feed_dict, options=run_options)

    gradients_tuple, total_loss, base_loss, policy_loss, value_loss, entropy = return_list[:6]
    grad_norm = gradients_tuple[1]
    return_list = return_list[6:]
    return_string = "Trainer {}>>> Total loss: {}, Base loss: {}\n".format(self.thread_index, total_loss, base_loss)
    return_string += "\t\tPolicy loss: {}, Value loss: {}, Grad norm: {}\nEntropy: {}\n".format(policy_loss, value_loss,
                                                                                                grad_norm, entropy)
    losses_eval = {'all/total_loss': total_loss, 'all/base_loss': base_loss,
                   'all/policy_loss': policy_loss, 'all/value_loss': value_loss, 'all/loss/grad_norm': grad_norm}
    if self.segnet_mode >= 2:
      decoder_loss, l2_loss = return_list[:2]
      return_list = return_list[2:]
      return_string += "\t\tDecoder loss: {}, L2 weights loss: {}\n".format(decoder_loss, l2_loss)
      losses_eval.update({'all/decoder_loss': decoder_loss, 'all/l2_weights_loss': l2_loss})
    if self.use_pixel_change:
      pc_loss = return_list[0]
      return_list = return_list[1:]
      return_string += "\t\tPC loss: {}\n".format(pc_loss)
      losses_eval.update({'all/pc_loss': pc_loss})
    if self.use_value_replay:
      vr_loss = return_list[0]
      return_list = return_list[1:]
      return_string += "\t\tVR loss: {}\n".format(vr_loss)
      losses_eval.update({'all/vr_loss': vr_loss})
    if self.use_reward_prediction:
      rp_loss = return_list[0]
      return_list = return_list[1:]
      return_string += "\t\tRP loss: {}\n".format(rp_loss)
      losses_eval.update({'all/rp_loss': rp_loss})
    if self.local_t - self.prev_local_t_loss >= LOSS_AND_EVAL_LOG_INTERVAL:
      if self.segnet_mode >= 2:
        return_string += "\t\tmIoU: {}\n".format(return_list[-1])

    summary_dict['values'].update(losses_eval)

      # Printing losses
    if self.local_t - self.prev_local_t_loss >= LOSS_AND_EVAL_LOG_INTERVAL:
      if self.segnet_mode >= 2:
        self._record_one(sess, summary_writer, summary_op_dict['eval_input'], eval_input,
                         return_list[-1], self.local_t)
      self._record_one(sess, summary_writer, summary_op_dict['entropy'], entropy_input,
                       entropy, self.local_t)
      summary_writer.flush()
      print(return_string)
      self.prev_local_t_loss += LOSS_AND_EVAL_LOG_INTERVAL

    if GPU_LOG:
      fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
      chrome_trace = fetched_timeline.generate_chrome_trace_format()
      self.many_runs_timeline.update_timeline(chrome_trace)
    
    self._print_log(global_t)

    #Recording score and losses
    self._record_all(sess, summary_writer, summary_op_dict['losses_input'], summary_dict['placeholders'],
                     summary_dict['values'], self.local_t)
    
    # Return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t


  ### TimeLiner class

class TimeLiner:
  _timeline_dict = None

  def update_timeline(self, chrome_trace):
    # convert crome trace to python dict
    chrome_trace_dict = json.loads(chrome_trace)
    # for first run store full trace
    if self._timeline_dict is None:
      self._timeline_dict = chrome_trace_dict
    # for other - update only time consumption, not definitions
    else:
      for event in chrome_trace_dict['traceEvents']:
        # events time consumption started with 'ts' prefix
        if 'ts' in event:
          self._timeline_dict['traceEvents'].append(event)

  def save(self, f_name):
    with open(f_name, 'w') as f:
      json.dump(self._timeline_dict, f)

