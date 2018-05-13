# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback

import tensorflow as tf
import numpy as np

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from environment.environment import Environment
from model.model import UnrealModel
from train.experience import ExperienceFrame
from options import get_options
import minos.config.sim_config as sim_config

# get command line args
flags = get_options("evaluate")
tf.logging.set_verbosity(tf.logging.DEBUG)

if flags.segnet >= 1:
  flags.use_pixel_change = False

USE_GPU = False
device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

class Evaluate(object):
  def __init__(self):
    self.action_size = Environment.get_action_size(flags.env_type, flags.env_name)
    self.objective_size = Environment.get_objective_size(flags.env_type, flags.env_name)

    env_config = sim_config.get(flags.env_name)
    self.image_shape = [env_config['height'], env_config['width']]
    segnet_param_dict = {'segnet_mode': flags.segnet}
    is_training = tf.placeholder(tf.bool, name="training") # for display param in UnrealModel says its value

    self.global_network = UnrealModel(self.action_size,
                                      self.objective_size,
                                      -1,
                                      flags.use_lstm,
                                      flags.use_pixel_change,
                                      flags.use_value_replay,
                                      flags.use_reward_prediction,
                                      0.0, #flags.pixel_change_lambda
                                      0.0, #flags.entropy_beta
                                      device,
                                      segnet_param_dict=segnet_param_dict,
                                      image_shape=self.image_shape,
                                      is_training=is_training,
                                      n_classes=flags.n_classes,
                                      segnet_lambda=flags.segnet_lambda,
                                      dropout=flags.dropout,
                                      for_display=True)
    self.environment = Environment.create_environment(flags.env_type, flags.env_name,
                                                      env_args={'episode_schedule': flags.split,
                                                                'log_action_trace': flags.log_action_trace,
                                                                'max_states_per_scene': flags.episodes_per_scene,
                                                                'episodes_per_scene_test': flags.episodes_per_scene})

    self.global_network.prepare_loss()

    self.total_loss = []
    self.segm_loss = []
    self.episode_reward = [0]
    self.success_rate = []
    self.batch_size = 20
    self.batch_cur_num = 0
    self.batch_prev_num = 0
    self.batch_si = []
    self.batch_sobjT = []
    self.batch_a = []
    self.batch_reward = []

  def update(self, sess):
    self.process(sess)

  def is_done(self):
    return self.environment.is_all_scheduled_episodes_done()

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def process(self, sess):
    last_action = self.environment.last_action
    last_reward = np.clip(self.environment.last_reward, -1, 1)
    last_action_reward = ExperienceFrame.concat_action_and_reward(last_action, self.action_size,
                                                                  last_reward, self.environment.last_state)
    mode = "segnet" if flags.segnet >= 2 else ""
    if not flags.use_pixel_change:
      pi_values, v_value, _ = self.global_network.run_base_policy_and_value(sess,
                                                                         self.environment.last_state,
                                                                         last_action_reward, mode=mode)
    else:
      pi_values, v_value, pc_q = self.global_network.run_base_policy_value_pc_q(sess,
                                                                                self.environment.last_state,
                                                                                last_action_reward)

    self.batch_cur_num += 1

    if flags.segnet >= 2:
      if self.batch_cur_num != 0 and self.batch_cur_num - self.batch_prev_num >= self.batch_size:

        #print(np.unique(self.batch_sobjT))
        feed_dict = {self.global_network.base_input: self.batch_si,
                     self.global_network.base_segm_mask: self.batch_sobjT,
                     self.global_network.is_training: not True}

        segm_loss, preds, confusion_mtx = sess.run([self.global_network.decoder_loss,
                                                  self.global_network.preds, self.global_network.update_evaluation_vars],
                                                 feed_dict=feed_dict)
        total_loss = 0
        self.total_loss += [total_loss]
        self.segm_loss += [segm_loss] # TODO: here do something with it, store somwhere?

        #update every_thing else
        self.batch_prev_num = self.batch_cur_num
        self.batch_si = []
        self.batch_sobjT = []
        self.batch_a = []
      else:
        self.batch_si += [self.environment.last_state["image"]]
        self.batch_sobjT += [self.environment.last_state["objectType"]]
        self.batch_a += [self.environment.ACTION_LIST[self.environment.last_action]]

    action = self.choose_action(pi_values)
    state, reward, terminal, pixel_change = self.environment.process(action)
    self.episode_reward[-1] += reward

    if terminal:
      self.success_rate += [int(self.environment._last_full_state["success"])]
      self.environment.reset()
      self.episode_reward += [0]


def main(args):
  evaluate = None
  try:
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)  # avoid using all gpu memory
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False) #gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    evaluate = Evaluate()

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)

    print("Checkpoint file path:", checkpoint.model_checkpoint_path)

    # List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
    print_tensors_in_checkpoint_file(file_name=checkpoint.model_checkpoint_path,
                                     tensor_name='', all_tensors=True, all_tensor_names=True)
    #print(tf.get_default_graph().as_graph_def())
    #
    # from tensorflow.python import pywrap_tensorflow
    # import os
    # checkpoint_path = os.path.join(model_dir, "model.ckpt")
    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint.model_checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #   print("tensor_name: ", key)
    #   #print(reader.get_tensor(key))

    if checkpoint and checkpoint.model_checkpoint_path:
      saver.restore(sess, checkpoint.model_checkpoint_path)
      print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    else:
      print("Could not find old checkpoint")

    if flags.segnet >= 2:
      sess.run([evaluate.global_network.reset_evaluation_vars])

    while not evaluate.is_done():
      evaluate.update(sess)

    evaluate.episode_reward = evaluate.episode_reward[:-1] # last is unnecessary
    n_episode = len(evaluate.episode_reward)
    evaluate.success_rate = np.sum(evaluate.success_rate) / n_episode
    if flags.segnet >= 2:
      score = sess.run(evaluate.global_network.evaluation)
      # print(type(score),
      #       np.isnan(score),
      #       score is None)
      print("Global mIoU: {}".format(score))
    print("Success Rate:{} ".format(evaluate.success_rate))
    print("Episode rewards: {}".format(evaluate.episode_reward))

  except Exception as e:
    print(traceback.format_exc())
  finally:
    if evaluate is not None:
      evaluate.environment.stop()

if __name__ == '__main__':
  tf.app.run()
