# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.client import timeline

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import threading
import signal
import math
import os
import sys
import time
import traceback

import json
import numpy as np
import importlib

from environment.environment import Environment
from model.model import UnrealModel
from train.trainer import Trainer
from train.rmsprop_applier import RMSPropApplier
from options import get_options
import minos.config.sim_config as sim_config

USE_GPU = True # To use GPU, set True

base_dir = "/root/hdd/repos/minos_unreal"

# get command line args
flags = get_options("training")
flags.log_dir = os.path.join(base_dir, flags.checkpoint_dir, flags.log_dir)


if flags.segnet >= 1:
  flags.use_pixel_change = False

Environment.set_log_dir(flags.log_dir)

tf.logging.set_verbosity(tf.logging.ERROR)
#### Logging in file
# import logging
#
# # get TF logger
# log = logging.getLogger('tensorflow')
# log.setLevel(logging.DEBUG)
#
# # create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
# # create file handler which logs even debug messages
# fh = logging.FileHandler('tensorflow.log')
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# log.addHandler(fh)

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)


class Application(object):
  def __init__(self):
    pass
  
  def train_function(self, parallel_index, preparing):
    """ Train each environment. """
    
    trainer = self.trainers[parallel_index]
    try:
      if preparing:
        trainer.prepare()
    except Exception as e:
      print(str(e))#, flush=True)
      raise Exception("Problem with trainer environment creation")

    # set start_time
    trainer.set_start_time(self.start_time)
    print("Trainer ", parallel_index, " process (re)start!")

    prev_print_t = 0
    while True:
      if self.global_t - prev_print_t >= 1000 or not prev_print_t and self.global_t != prev_print_t:
        prev_print_t = self.global_t
        print("Trainer {0}>>> stop_requested: {1}, terminate_requested: {2}, global_t: {3}".format(parallel_index, self.stop_requested,
                                                self.terminate_requested, self.global_t))
        #if parallel_index == 0:
        #    print("Graph size is {}".format(
        #        len([n.name for n in self.sess.graph.as_graph_def().node])))

      if self.stop_requested:
        print("Trainer ", parallel_index, ": stop requested!")
        break
      if self.terminate_requested:
        print("Trainer ", parallel_index, ": terminate_requested => process stop!")
        trainer.stop()
        break
      if self.global_t > flags.max_time_step:
        print("Trainer ", parallel_index, ": end of training!")
        trainer.stop()
        break
      if parallel_index == 0 and self.global_t > self.next_save_steps:
        # Save checkpoint
        self.save()


      try:
        diff_global_t, score = trainer.process(self.sess,
                                            self.global_t,
                                            self.summary_writer[parallel_index],
                                            self.summary_op_dict,
                                            self.score_input,
                                            self.sr_input,
                                            self.mIoU_input,
                                            self.entropy_input,
                                            self.term_global_t,
                                            self.losses_input)

        self.global_t += diff_global_t
        # if parallel_index == 0 and score is not None:
        #     #print("Got score", flush=True)
        #     self.last_scores += [score]
        #     if len(self.last_scores) >= 50:
        #         print("Last scores len >= 50")
        #         cur_score = np.mean(self.last_scores)
        #         print("Best score: {}, Cur score: {}".format(self.best_score, cur_score))
        #         self.last_scores = self.last_scores[-10:]
        #         if cur_score > self.best_score:
        #             self.best_score = cur_score
        #             self.save(name="best-checkpoint")

        # [n.name for n in tf.get_default_graph().as_graph_def().node]
        # [op for op in tf.get_default_graph().get_operations()] #op.name


        # GPU logging memory
        # prev_runs_t = 0
        # if self.global_t - prev_runs_t > 1000 or prev_runs_t == 0:
        #   prev_runs_t = self.global_t
        #   trainer.many_runs_timeline.save(os.path.join(flags.checkpoint_dir,
        #                                                'timeline_{}_merged_{}_runs.json'.format(
        #                                                  parallel_index, self.global_t)))
        # else:
        #   trainer.many_runs_timeline.save(os.path.join(flags.checkpoint_dir,
        #                                                'timeline_{}_merged_{}_runs.json'.format(
        #                                                  parallel_index, prev_runs_t)))
        #

      except Exception as e:
        print(traceback.format_exc())#, flush=True)
        trainer.stop()
        ## Let it be here!!!
        print("Trainer ", parallel_index, " process Error!")#, flush=True)
        break

    print("Trainer ", parallel_index, " after a while return!")

  def run(self):
    device = "/cpu:0"
    if USE_GPU:
      device = "/gpu:0"

    self.print_flags_info()

    if flags.segnet == -1:
      with open(flags.segnet_config) as f:
        self.config = json.load(f)

      self.num_classes = self.config["NUM_CLASSES"]
      self.use_vgg = self.config["USE_VGG"]

      if self.use_vgg is False:
        self.vgg_param_dict = None
        print("No VGG path in config, so learning from scratch")
      else:
        self.vgg16_npy_path = self.config["VGG_FILE"]
        self.vgg_param_dict = np.load(self.vgg16_npy_path, encoding='latin1').item()
        print("VGG parameter loaded")

      self.bayes = self.config["BAYES"]
      segnet_param_dict = {'segnet_mode': flags.segnet, 'vgg_param_dict': self.vgg_param_dict, 'use_vgg': self.use_vgg,
                       'num_classes': self.num_classes, 'bayes': self.bayes}
    else: # 0, 1, 2, 3
      segnet_param_dict = {'segnet_mode': flags.segnet}

    if flags.env_type != 'indoor':
        env_config = {}
    else:
        env_config = sim_config.get(flags.env_name)
    self.image_shape = [env_config.get('height', 84), env_config.get('width', 84)]
    self.map_file = env_config.get('objecttypes_file', '../../objectTypes_1x.csv')
    
    initial_learning_rate = log_uniform(flags.initial_alpha_low,
                                        flags.initial_alpha_high,
                                        flags.initial_alpha_log_rate)
    self.global_t = 0
    
    self.stop_requested = False
    self.terminate_requested = False
    
    action_size = Environment.get_action_size(flags.env_type,
                                              flags.env_name)
    objective_size = Environment.get_objective_size(flags.env_type, flags.env_name)

    is_training = tf.placeholder(tf.bool, name="training")

    self.random_state = np.random.RandomState(seed=env_config.get("seed", 0xA3C))

    print("Global network initializing!")#, flush=True)

    self.global_network = UnrealModel(action_size,
                                      objective_size,
                                      -1,
                                      flags.use_lstm,
                                      flags.use_pixel_change,
                                      flags.use_value_replay,
                                      flags.use_reward_prediction,
                                      flags.pixel_change_lambda,
                                      flags.entropy_beta,
                                      device,
                                      segnet_param_dict=segnet_param_dict,
                                      image_shape=self.image_shape,
                                      is_training=is_training,
                                      n_classes=flags.n_classes,
                                      segnet_lambda=flags.segnet_lambda,
                                      dropout=flags.dropout)
    self.trainers = []
    
    learning_rate_input = tf.placeholder("float")

    
    grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = flags.rmsp_alpha,
                                  momentum = 0.0,
                                  epsilon = flags.rmsp_epsilon,
                                  clip_norm = flags.grad_norm_clip,
                                  device = device)
    
    for i in range(flags.parallel_size):
      trainer = Trainer(i,
                        self.global_network,
                        initial_learning_rate,
                        learning_rate_input,
                        grad_applier,
                        flags.env_type,
                        flags.env_name,
                        flags.use_lstm,
                        flags.use_pixel_change,
                        flags.use_value_replay,
                        flags.use_reward_prediction,
                        flags.pixel_change_lambda,
                        flags.entropy_beta,
                        flags.local_t_max,
                        flags.n_step_TD,
                        flags.gamma,
                        flags.gamma_pc,
                        flags.experience_history_size,
                        flags.max_time_step,
                        device,
                        segnet_param_dict=segnet_param_dict,
                        image_shape=self.image_shape,
                        is_training=is_training,
                        n_classes = flags.n_classes,
                        random_state=self.random_state,
                        termination_time=flags.termination_time_sec,
                        segnet_lambda=flags.segnet_lambda,
                        dropout=flags.dropout)
      self.trainers.append(trainer)


    self.last_scores = []
    self.best_score = -1.0
    
    # prepare session
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)

    # Wrap sess.run for debugging messages!
    def run_(*args, **kwargs):
      #print(">>> RUN!", args[0] if args else None)#, flush=True)
      return self.sess.__run(*args, **kwargs)  # getattr(self, "__run")(self, *args, **kwargs)
    self.sess.__run, self.sess.run = self.sess.run, run_

    self.sess.run(tf.global_variables_initializer())

    # summary for tensorboard
    self.score_input = tf.placeholder(tf.float32)
    self.sr_input = tf.placeholder(tf.float32)
    self.mIoU_input = tf.placeholder(tf.float32)
    self.term_global_t = tf.placeholder(tf.int32)

    self.losses_input = {}

    self.total_loss = tf.placeholder(tf.float32,  name='total_loss')
    self.losses_input.update({'all/total_loss': self.total_loss})

    self.base_loss = tf.placeholder(tf.float32, name='base_loss')
    self.losses_input.update({'all/base_loss': self.base_loss})

    self.policy_loss = tf.placeholder(tf.float32,  name='policy_loss')
    self.losses_input.update({'all/policy_loss': self.policy_loss})

    self.value_loss = tf.placeholder(tf.float32, name='policy_loss')
    self.losses_input.update({'all/value_loss': self.value_loss})

    self.grad_norm = tf.placeholder(tf.float32, name='grad_norm')
    self.losses_input.update({'all/loss/grad_norm': self.grad_norm})

    self.entropy_input = tf.placeholder(tf.float32, shape=[None], name='entropy')

    if segnet_param_dict["segnet_mode"] >= 2:
      self.decoder_loss = tf.placeholder(tf.float32,  name='decoder_loss')
      self.losses_input.update({'all/decoder_loss': self.decoder_loss})
      self.l2_weights_loss = tf.placeholder(tf.float32, name='regul_weights_loss')
      self.losses_input.update({'all/l2_weights_loss': self.l2_weights_loss})
    if flags.use_pixel_change:
      self.pc_loss = tf.placeholder(tf.float32,  name='pc_loss')
      self.losses_input.update({'all/pc_loss': self.pc_loss})
    if flags.use_value_replay:
      self.vr_loss = tf.placeholder(tf.float32,  name='vr_loss')
      self.losses_input.update({'all/vr_loss': self.vr_loss})
    if flags.use_reward_prediction:
      self.rp_loss = tf.placeholder(tf.float32,  name='rp_loss')
      self.losses_input.update({'all/rp_loss': self.rp_loss})

    score_summary = tf.summary.scalar("all/eval/score", self.score_input)
    sr_summary = tf.summary.scalar("all/eval/success_rate", self.sr_input)
    term_summary = tf.summary.scalar("all/eval/term_global_t", self.term_global_t)
    eval_summary = tf.summary.scalar("all/eval/mIoU_all", self.mIoU_input)
    losses_summary_list = []
    for key, val in self.losses_input.items():
      losses_summary_list += [tf.summary.scalar(key, val)]


    self.summary_op_dict = {'score_input': score_summary, 'eval_input': eval_summary, 'sr_input':sr_summary,
                            'losses_input': tf.summary.merge(losses_summary_list),
                            'entropy': tf.summary.scalar('all/eval/entropy_stepTD', tf.reduce_mean(self.entropy_input)),
                            'term_global_t': term_summary}
    flags.checkpoint_dir = os.path.join(base_dir, flags.checkpoint_dir)
    #print("First dirs {}::{}".format(flags.log_dir, flags.checkpoint_dir))
    flags.checkpoint_dir = flags.checkpoint_dir
    print("Checkpoint dir: {}, Log dir: {}".format(flags.checkpoint_dir, flags.log_dir))
    overall_FW = tf.summary.FileWriter(os.path.join(flags.log_dir, 'overall'),
                          self.sess.graph)
    self.summary_writer = [(tf.summary.FileWriter(os.path.join(flags.log_dir, 'worker_{}'.format(i)),
                                                self.sess.graph), overall_FW) for i in range(flags.parallel_size)]
    
    # init or load checkpoint with saver
    self.saver = tf.train.Saver(self.global_network.get_global_vars(), max_to_keep=20)


    
    #checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir, latest_filename ="best-checkpoint")
    #if checkpoint is None or checkpoint.model_checkpoint_path is None:
    #  checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
    checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)

    if checkpoint and checkpoint.model_checkpoint_path:
      if flags.segnet == -1:
          from tensorflow.python import pywrap_tensorflow
          reader = pywrap_tensorflow.NewCheckpointReader(checkpoint.model_checkpoint_path)
          big_var_to_shape_map = reader.get_variable_to_shape_map()
          s = []
          for key in big_var_to_shape_map:
              s += [key]
              # print("tensor_name: ", key)
          glob_var_names = [v.name for v in tf.global_variables()]
          endings = [r.split('/')[-1][:-2] for r in glob_var_names]
          old_ckpt_to_new_ckpt = {[k for k in s if endings[i] in k][0]: v for i, v in enumerate(tf.global_variables())}
          saver1 = tf.train.Saver(var_list=old_ckpt_to_new_ckpt)
          saver1.restore(self.sess, checkpoint.model_checkpoint_path)
      else:
          self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
      print("checkpoint loaded:", checkpoint.model_checkpoint_path)
      tokens = checkpoint.model_checkpoint_path.split("-")
      # set global step
      if 'best' in checkpoint.model_checkpoint_path:
          files = os.listdir(flags.checkpoint_dir)
          max_g_step = 0
          max_best_score = -10
          for file in files:
            if '.meta' not in file or 'checkpoint' not in file:
              continue
            if len(tokens) == 2:
              continue
            if len(tokens) > 3:
              best_score = float('-0.'+file.split('-')[2]) if 'best' in file else float('-0.'+file.split('-')[1])
              if best_score > max_best_score:
                g_step = int(file.split('-')[3]).split('.')[0] if 'best' in file else int(file.split('-')[2].split('.')[0])
                if max_g_step < g_step:
                  max_g_step = g_step
            else:
              self.best_score = -1.0
              g_step = int(file.split('-')[2]) if 'best' in file else int(file.split('-')[1])
              if max_g_step < g_step:
                max_g_step = g_step
          self.best_score = max_best_score
          self.global_t = max_g_step
          print("Chosen g_step >>", g_step)
      else:
        if len(tokens) == 3:
          self.global_t = int(tokens[2])
        else:
          self.global_t = int(tokens[1])
      #for i in range(flags.parallel_size):
      #  self.trainers[i].local_t = self.global_t
      print(">>> global step set: ", self.global_t)
      # set wall time
      wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
      with open(wall_t_fname, 'r') as f:
        self.wall_t = float(f.read())
        self.next_save_steps = (self.global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step
      print_tensors_in_checkpoint_file(file_name=checkpoint.model_checkpoint_path,
                                     tensor_name='', all_tensors=False, all_tensor_names=True)
    else:
      print("Could not find old checkpoint")
      # set wall time
      self.wall_t = 0.0
      self.next_save_steps = flags.save_interval_step
    print("Global step {}, max best score {}".format(self.global_t, self.best_score))

    if flags.segnet_pretrain:
      checkpoint_dir = "../erfnet_segmentation/models"
      checkpoint_dir = os.path.join(checkpoint_dir, "aug_erfnetC_0_{}x{}_{}x/snapshots_best".format(
          self.image_shape[1],
          self.image_shape[0],
          self.map_file.split('_')[1].split('x')[0]))
      checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

      big_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net_-1/base_encoder')
      big_weights += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net_-1/base_decoder')

      erfnet_weights = [l.name.split(':')[0].rsplit('net_-1/base_encoder/')[-1] for l in big_weights
             if len(l.name.split(':')[0].rsplit('net_-1/base_encoder/')) == 2]
      erfnet_weights += [l.name.split(':')[0].rsplit('net_-1/base_decoder/')[-1] for l in big_weights
              if len(l.name.split(':')[0].rsplit('net_-1/base_decoder/')) == 2]

      if checkpoint and checkpoint.model_checkpoint_path:
          saver2 = tf.train.Saver(var_list=dict(zip(erfnet_weights, big_weights)))
          saver2.restore(self.sess, checkpoint.model_checkpoint_path)
          print("ErfNet pretrained weights restored from file ", checkpoint_dir)
      else:
          print("Can't load pretrained weights for ErfNet from file ", checkpoint_dir)

    # run training threads
    self.train_threads = []
    for i in range(flags.parallel_size):
      self.train_threads.append(threading.Thread(target=self.train_function, args=(i,True)))
      
    signal.signal(signal.SIGINT, self.signal_handler)
  
    # set start time
    self.start_time = time.time() - self.wall_t

    print("Ready to start")
    for t in self.train_threads:
      t.start()
  
    print('Press Ctrl+C to stop')#, flush=True)
    signal.pause()

  def save(self, name=""):
    """ Save checkpoint. 
    Called from thread-0.
    """
    self.stop_requested = True
  
    # Wait for all other threads to stop
    print("Waiting for childs!")#, flush=True)
    for (i, t) in enumerate(self.train_threads):
      if i != 0:
        t.join()
  
    # Save
    try:
        if not os.path.exists(flags.checkpoint_dir):
          os.mkdir(flags.checkpoint_dir)

        # Write wall time
        wall_t = time.time() - self.start_time
        wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(self.global_t)
        #if not os.path.exists(wall_t_fname):
        #  os.mkdir(wall_t_fname)

        with open(wall_t_fname, 'w') as f:
          f.write(str(wall_t))

        ckpt_name = 'checkpoint'
        if name != "":
            ckpt_name = name
        ckpt_name += '-' + str(abs(self.best_score))[2:8] # Here may be bug
        print('Start saving.')
        self.saver.save(self.sess,
                        flags.checkpoint_dir + '/' + ckpt_name,
                        global_step = self.global_t)
        print('End saving.')

        self.stop_requested = False
        self.next_save_steps += flags.save_interval_step
    except Exception as e:
        self.terminate_requested = True
        ## Let it be here for debug save() function!!!
        print(traceback.format_exc())#, flush=True)
        raise Exception("Error in 'save' occured!")
    finally:
        # Restart other threads
        print("Restarting other threads!")
        for i in range(flags.parallel_size):
          if i != 0:
            thread = threading.Thread(target=self.train_function, args=(i,False))
            self.train_threads[i] = thread
            thread.start()
    
  def signal_handler(self, signal, frame):
    print('You pressed Ctrl+C!')#, flush=True)
    self.terminate_requested = True

  def print_flags_info(self):
    return_string = "\n\n\n"
    return_string += "Envs FILE:{}\n".format(flags.env_name)
    return_string += "Checkpoint dir: {}, Termination time in sec: " \
                     "{}, Max steps to train: {:2.3E}, Parallel threads:{}\n".format(flags.checkpoint_dir,
                                                                                   flags.termination_time_sec,
                                                                                   flags.max_time_step,
                                                                                   flags.parallel_size)
    return_string += "Use ErfNet Encoder-Decoder, N classes: {}\n".format(flags.n_classes) if flags.segnet >= 2 else ""
    return_string += "Use ErfNet Encoder only\n" if flags.segnet == 1 else ""
    return_string += "Use vanilla encoder\n" if flags.segnet == 0 else ""
    return_string += "Use PC:{}, Use VR:{}, use RP:{}\n".format(flags.use_pixel_change,
                                                              flags.use_value_replay,
                                                              flags.use_reward_prediction)
    return_string += "Experience hist size: {}, Local_t max: {}, n-step-TD: {}\n".format(flags.experience_history_size,
                                                                                   flags.local_t_max,
                                                                                   flags.n_step_TD)
    return_string += "Entropy beta: {}, Gradient norm clipping: {}, Rmsprop alpha: {}, Saving step: {}\n".format(
                                                                                     flags.entropy_beta,
                                                                                     flags.grad_norm_clip,
                                                                                     flags.rmsp_alpha,
                                                                                     flags.save_interval_step)
    print(return_string)

def main(argv):
  app = Application()
  app.run()

if __name__ == '__main__':
  tf.app.run()
