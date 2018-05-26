# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import os

def get_options(option_type):
  """
  option_type: string
    'training' or 'display' or 'visualize'
  """
  # Common
  tf.app.flags.DEFINE_string("env_type", "lab", "environment type (lab or gym or maze or indoor)")
  tf.app.flags.DEFINE_string("env_name", "nav_maze_static_01",  "environment name")
  tf.app.flags.DEFINE_boolean("use_lstm", True, "whether to use lstm")
  tf.app.flags.DEFINE_boolean("use_pixel_change", False, "whether to use pixel change")
  tf.app.flags.DEFINE_boolean("use_value_replay", True, "whether to use value function replay")
  tf.app.flags.DEFINE_boolean("use_reward_prediction", True, "whether to use reward prediction")
  tf.app.flags.DEFINE_boolean("segnet_pretrain", False, "whether to use pretrained ErfNet weights")

  tf.app.flags.DEFINE_string("checkpoint_dir", "lab_ckpt", "checkpoint directory")
  tf.app.flags.DEFINE_string("checkpoint", "", "checkpoint directory")

  # Segmentation options
  tf.app.flags.DEFINE_integer("segnet", 2, "use segmentation mode: 0 - no, "
                                           "1 - encoder only, 2 - decoder on encoder, "
                                           "3 - decoder on lstm output")
  tf.app.flags.DEFINE_string("segnet_config", "config.json", "segnet config file")
  tf.app.flags.DEFINE_integer("n_classes", 19, "segmentation classes")
  tf.app.flags.DEFINE_float("termination_time_sec", 50.0, "seconds until termination (steps/10)")

  tf.app.flags.DEFINE_float("segnet_lambda", 1.0, "weighting of segmentation network loss")
  tf.app.flags.DEFINE_float("dropout", 0.3, "dropout for encoder")
  tf.app.flags.DEFINE_integer("parallel_size", 8, "parallel thread size")
  tf.app.flags.DEFINE_integer("local_t_max", 20, "repeat step size")
  tf.app.flags.DEFINE_integer("n_step_TD", 20, "size n for n-step TD")
  tf.app.flags.DEFINE_float("entropy_beta", 0.001, "entropy regularization constant")

  # For training
  if option_type == 'training':
    tf.app.flags.DEFINE_float("greedy_epsilon", 0.99, "decay parameter for rmsprop")
    tf.app.flags.DEFINE_float("rmsp_alpha", 0.99, "decay parameter for rmsprop")
    tf.app.flags.DEFINE_float("rmsp_epsilon", 0.1, "epsilon parameter for rmsprop")
    file = os.path.join("./logs", time.strftime("%Y_%m_%d_%H_%M_%S"))
    tf.app.flags.DEFINE_string("log_dir",
                              file, #"./logs"
                              "log file directory")
    # tf.app.flags.DEFINE_string("log_file", "/tmp/unreal_log/unreal_log", "log file directory")
    tf.app.flags.DEFINE_float("initial_alpha_low", 1e-4, "log_uniform low limit for learning rate")
    tf.app.flags.DEFINE_float("initial_alpha_high", 5e-3, "log_uniform high limit for learning rate")
    tf.app.flags.DEFINE_float("initial_alpha_log_rate", 0.5, "log_uniform interpolate rate for learning rate")
    tf.app.flags.DEFINE_float("gamma", 0.99, "discount factor for rewards")
    tf.app.flags.DEFINE_float("gamma_pc", 0.9, "discount factor for pixel control")
    tf.app.flags.DEFINE_float("pixel_change_lambda", 0.05, "pixel change lambda") # 0.05, 0.01 ~ 0.1 for lab, 0.0001 ~ 0.01 for gym
    tf.app.flags.DEFINE_integer("experience_history_size", 2000, "experience replay buffer size") # 2000
    tf.app.flags.DEFINE_integer("max_time_step", int(13.2 * 10**6), "max time steps")
    tf.app.flags.DEFINE_integer("save_interval_step", 100 * 1000, "saving interval steps") # 100*1000
    tf.app.flags.DEFINE_float("grad_norm_clip", 40.0, "gradient norm clipping")

  # For display
  if option_type == 'display':
    tf.app.flags.DEFINE_string("frame_save_dir", "/tmp/unreal_frames", "frame save directory")
    tf.app.flags.DEFINE_boolean("recording", False, "whether to record movie")
    tf.app.flags.DEFINE_boolean("frame_saving", False, "whether to save frames")

  # For evaluation or display
  if option_type in ('evaluate', 'display'):
    tf.app.flags.DEFINE_string("split", "val", "What data split to use")
    tf.app.flags.DEFINE_integer("episodes_per_scene", 1, "How many episodes to test per scene")
    tf.app.flags.DEFINE_boolean("log_action_trace", True, "Whether to log action trace")

  return tf.app.flags.FLAGS
