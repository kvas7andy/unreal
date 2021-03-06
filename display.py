# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
import cv2
import os
from collections import deque
import pygame

import traceback
import pandas as pd

from environment.environment import Environment
from model.model import UnrealModel
from train.experience import ExperienceFrame
from options import get_options
import minos.config.sim_config as sim_config

BLUE  = (128, 128, 255)
RED   = (255, 192, 192)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# get command line args
flags = get_options("display")

if flags.segnet >= 1:
  flags.use_pixel_change = False
tf.logging.set_verbosity(tf.logging.DEBUG)


class MovieWriter(object):
  def __init__(self, file_name, frame_size, fps):
    """
    frame_size is (w, h)
    """
    self._frame_size = frame_size
    fourcc = cv2.VideoWriter_fourcc('m','p', '4', 'v')
    self.vout = cv2.VideoWriter()
    success = self.vout.open(file_name, fourcc, fps, frame_size, True)
    if not success:
      print("Create movie failed: {0}".format(file_name))

  def add_frame(self, frame):
    """
    frame shape is (h, w, 3), dtype is np.uint8
    """
    self.vout.write(frame)

  def close(self):
    self.vout.release() 
    self.vout = None


class StateHistory(object):
  def __init__(self):
    self._states = deque(maxlen=3)

  def add_state(self, state):
    self._states.append(state)

  @property
  def is_full(self):
    return len(self._states) >= 3

  @property
  def states(self):
    return list(self._states)


class ValueHistory(object):
  def __init__(self):
    self._values = deque(maxlen=100)

  def add_value(self, value):
    self._values.append(value)

  @property    
  def is_empty(self):
    return len(self._values) == 0

  @property
  def values(self):
    return self._values


class Display(object):
  def __init__(self, display_size):
    pygame.init()
    
    self.surface = pygame.display.set_mode(display_size, 0, 24)
    name = 'UNREAL' if flags.segnet == 0 else "A3C ErfNet"
    pygame.display.set_caption(name)

    env_config = sim_config.get(flags.env_name)
    self.image_shape = [env_config.get('height', 88), env_config.get('width', 88)]
    segnet_param_dict = {'segnet_mode': flags.segnet}
    is_training = tf.placeholder(tf.bool, name="training")
    map_file = env_config.get('objecttypes_file', '../../objectTypes.csv')
    self.label_mapping = pd.read_csv(map_file, sep=',', header=0)
    self.get_col_index()

    self.action_size = Environment.get_action_size(flags.env_type, flags.env_name)
    self.objective_size = Environment.get_objective_size(flags.env_type, flags.env_name)
    self.global_network = UnrealModel(self.action_size,
                                      self.objective_size,
                                      -1,
                                      flags.use_lstm,
                                      flags.use_pixel_change,
                                      flags.use_value_replay,
                                      flags.use_reward_prediction,
                                      0.0,
                                      0.0,
                                      "/gpu:0",
                                      segnet_param_dict=segnet_param_dict,
                                      image_shape=self.image_shape,
                                      is_training=is_training,
                                      n_classes=flags.n_classes,
                                      segnet_lambda=flags.segnet_lambda,
                                      dropout=flags.dropout,
                                      for_display=True)
    self.environment = Environment.create_environment(flags.env_type, flags.env_name, flags.termination_time_sec,
                                                      env_args={'episode_schedule': flags.split,
                                                                'log_action_trace': flags.log_action_trace,
                                                                'max_states_per_scene': flags.episodes_per_scene,
                                                                'episodes_per_scene_test': flags.episodes_per_scene})
    self.font = pygame.font.SysFont(None, 20)
    self.value_history = ValueHistory()
    self.state_history = StateHistory()
    self.episode_reward = 0

  def update(self, sess):
    self.surface.fill(BLACK)
    self.process(sess)
    pygame.display.update()

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def scale_image(self, image, scale):
    return image.repeat(scale, axis=0).repeat(scale, axis=1)

  def draw_text(self, str, left, top, color=WHITE):
    text = self.font.render(str, True, color, BLACK)
    text_rect = text.get_rect()
    text_rect.left = left    
    text_rect.top = top
    self.surface.blit(text, text_rect)  

  def draw_center_text(self, str, center_x, top):
    text = self.font.render(str, True, WHITE, BLACK)
    text_rect = text.get_rect()
    text_rect.centerx = center_x
    text_rect.top = top
    self.surface.blit(text, text_rect)

  def show_pixel_change(self, pixel_change, left, top, rate, label):
    """
    Show pixel change
    """
    if "PC" in label:
      pixel_change_ = np.clip(pixel_change * 255.0 * rate, 0.0, 255.0)
      data = pixel_change_.astype(np.uint8)
      data = np.stack([data for _ in range(3)], axis=2)
      data = self.scale_image(data, 4)
      #print("PC shape", data.shape)
      image = pygame.image.frombuffer(data, (20*4, 20*4), 'RGB')
    else:
      pixel_change = self.scale_image(pixel_change, 2)
      #print("Preds shape", pixel_change.shape)
      image = pygame.image.frombuffer(pixel_change.astype(np.uint8), (self.image_shape[0]*2, self.image_shape[1]*2), 'RGB')
    self.surface.blit(image, (2*left+16+8, 2*top+16+8))
    self.draw_center_text(label, 2*left + 200/2, 2*top + 200)
    

  def show_policy(self, pi):
    """
    Show action probability.
    """
    start_x = 10

    y = 150
  
    for i in range(len(pi)):
      width = pi[i] * 100
      pygame.draw.rect(self.surface, WHITE, (2*start_x, 2*y, 2*width, 2*10))
      y += 20
    self.draw_center_text("PI", 2*50, 2*y)
  
  def show_image(self, state):
    """
    Show input image
    """
    state_ = state * 255.0
    data = state_.astype(np.uint8)
    data = self.scale_image(data, 2)
    image = pygame.image.frombuffer(data, (self.image_shape[0]*2, self.image_shape[1]*2), 'RGB')
    self.surface.blit(image, (8*2, 8*2))
    self.draw_center_text("input", 2*50, 2*100)

  def show_value(self):
    if self.value_history.is_empty:
      return

    min_v = float("inf")
    max_v = float("-inf")

    values = self.value_history.values

    for v in values:
      min_v = min(min_v, v)
      max_v = max(max_v, v)

    top = 150*2
    left = 150*2
    width = 100*2
    height = 100*2
    bottom = top + width
    right = left + height

    d = max_v - min_v
    last_r = 0.0
    for i,v in enumerate(values):
      r = (v - min_v) / d
      if i > 0:
        x0 = i-1 + left
        x1 = i   + left
        y0 = bottom - last_r * height
        y1 = bottom - r * height
        pygame.draw.line(self.surface, BLUE, (x0, y0), (x1, y1), 1)
      last_r = r

    pygame.draw.line(self.surface, WHITE, (left,  top),    (left,  bottom), 1)
    pygame.draw.line(self.surface, WHITE, (right, top),    (right, bottom), 1)
    pygame.draw.line(self.surface, WHITE, (left,  top),    (right, top),    1)
    pygame.draw.line(self.surface, WHITE, (left,  bottom), (right, bottom), 1)

    self.draw_center_text("V", left + width/2, bottom+10)

  def show_reward_prediction(self, rp_c, reward):
    start_x = 310
    reward_index = 0
    if reward == 0:
      reward_index = 0
    elif reward > 0:
      reward_index = 1
    elif reward < 0:
      reward_index = 2

    y = 150

    labels = ["0", "+", "-"]
    
    for i in range(len(rp_c)):
      width = rp_c[i] * 100

      if i == reward_index:
        color = RED
      else:
        color = WHITE
      pygame.draw.rect(self.surface, color, (2*start_x+2*15, 2*y, 2*width, 2*10))
      self.draw_text(labels[i], 2*start_x, 2*y-2*1, color)
      y += 20
    
    self.draw_center_text("RP", 2*start_x + 2*100/2, y)

  def show_reward(self):
    self.draw_text("REWARD: {:.4}".format(float(self.episode_reward)), 300, 2*10)

  def process(self, sess):
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #sess.run(tf.initialize_all_variables())

    last_action = self.environment.last_action
    last_reward = self.environment.last_reward
    last_action_reward = ExperienceFrame.concat_action_and_reward(last_action, self.action_size,
                                                                  last_reward, self.environment.last_state)
    preds=None
    mode = "segnet" if flags.segnet >= 2 else ""
    mode="" #don't want preds
    if not flags.use_pixel_change:
      pi_values, v_value, preds = self.global_network.run_base_policy_and_value(sess,
                                                                         self.environment.last_state,
                                                                         last_action_reward, mode=mode)
    else:
      pi_values, v_value, pc_q = self.global_network.run_base_policy_value_pc_q(sess,
                                                                                self.environment.last_state,
                                                                                last_action_reward)

    #print(preds)
    self.value_history.add_value(v_value)

    prev_state = self.environment.last_state
    
    action = self.choose_action(pi_values)
    state, reward, terminal, pixel_change = self.environment.process(action)
    self.episode_reward += reward
  
    if terminal:
      self.environment.reset()
      self.episode_reward = 0
      
    self.show_image(state['image'])
    self.show_policy(pi_values)
    self.show_value()
    self.show_reward()
    
    if not flags.use_pixel_change:
      if preds is not None:
        self.show_pixel_change(self.label_to_rgb(preds), 100, 0, 3.0, "Preds")
        self.show_pixel_change(self.label_to_rgb(state['objectType']), 200, 0, 0.4, "Segm Mask")
    else:
      self.show_pixel_change(pixel_change, 100, 0, 3.0, "PC")
      self.show_pixel_change(pc_q[:,:,action], 200, 0, 0.4, "PC Q")
  
    if flags.use_reward_prediction:
      if self.state_history.is_full:
        rp_c = self.global_network.run_rp_c(sess, self.state_history.states)
        self.show_reward_prediction(rp_c, reward)
  
    self.state_history.add_state(state)

  def get_frame(self):
    data = self.surface.get_buffer().raw
    return data

  def get_col_index(self):
    ind_col = self.label_mapping[["index", "color"]].values
    index = ind_col[:, 0].astype(np.int)
    self.index, ind = np.unique(index, return_index=True)
    self.col = np.array([[int(x) for x in col.split('_')] for col in ind_col[ind, 1]])

  def label_to_rgb(self, labels):
    #print(self.col)
    rgb_img = self.col[np.where(self.index[np.newaxis, :] == labels.ravel()[:, np.newaxis])[1]].reshape(labels.shape + (3,))
    return rgb_img


def main(args):
  # prepare session
  config = tf.ConfigProto(allow_soft_placement=True)
  # log_device_placement = False,
  config.gpu_options.allow_growth = True

  sess = tf.Session(config=config)
  try:
    display_size = (440, 300)
    if flags.segnet >= 2:
      display_size = (660, 600)
    display = Display(display_size)
    saver = tf.train.Saver()

    if flags.checkpoint:
      saver.restore(sess, os.path.join(flags.checkpoint_dir, flags.checkpoint))
      print("Restored from checkpoint!")
    else:
      checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
      if checkpoint and checkpoint.model_checkpoint_path:
        if flags.segnet == 0:
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
          saver1.restore(sess, checkpoint.model_checkpoint_path)
        else:
          saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
      else:
        print("Could not find old checkpoint")
    # checkpoint_file = tf.train.latest_checkpoint(flags.checkpoint_dir)
    # print(checkpoint_file)
    # if checkpoint_file is None:
    #   pass
    # else:
    #   saver.restore(sess, checkpoint_file)

    clock = pygame.time.Clock()

    running = True
    FPS = 15

    if flags.recording:
      name = "out_{}.mov".format(flags.checkpoint_dir)
      i = 0
      while os.path.exists(name):
        name = "{}_{}".format(name, i)
      writer = MovieWriter(name, display_size, FPS)

    if flags.frame_saving:
      frame_count = 0
      if not os.path.exists(flags.frame_save_dir):
        os.mkdir(flags.frame_save_dir)

    while running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False

      display.update(sess)
      clock.tick(FPS)

      if flags.recording or flags.frame_saving:
        frame_str = display.get_frame()
        d = np.fromstring(frame_str, dtype=np.uint8)
        d = d.reshape((display_size[1], display_size[0], 3))
        if flags.recording:
          writer.add_frame(d)
        else:
          frame_file_path = "{0}/{1:06d}.png".format(flags.frame_save_dir, frame_count)
          cv2.imwrite(frame_file_path, d)
          frame_count += 1

    if flags.recording:
      writer.close()
  except Exception as e:
    print(traceback.format_exc())
  finally:
    display.environment.stop()

    
if __name__ == '__main__':
  tf.app.run()
