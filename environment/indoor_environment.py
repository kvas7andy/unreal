# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import time

from environment import environment
from minos.lib.RoomSimulator import RoomSimulator
from minos.config import sim_config

class IndoorEnvironment(environment.Environment):

  ACTION_LIST = [
    [1,0,0],
    [0,1,0],
    [0,0,1]
  ]

  @staticmethod
  def get_action_size(env_name):
    return len(IndoorEnvironment.ACTION_LIST)

  @staticmethod
  def get_objective_size(env_name):
    simargs = sim_config.get(env_name)
    return simargs.get('objective_size', 0)

  def __init__(self, env_name, env_args, termination_time, thread_index):
    environment.Environment.__init__(self)
    try:
      self.last_state = None
      self.last_action = 0
      self.last_reward = 0

      self.prev_state = None
      self.prev_action = 0
      self.prev_reward = 0

      simargs = sim_config.get(env_name)
      simargs['id'] = 'sim%02d' % thread_index
      simargs['logdir'] = os.path.join(IndoorEnvironment.get_log_dir(), simargs['id'])

      # Merge in extra env args
      if env_args is not None:
        simargs.update(env_args)

      simargs["measure_fun"].termination_time = termination_time

      self.termination_time = termination_time

      # try:
      self._sim = RoomSimulator(simargs)
      self._sim_obs_space = self._sim.get_observation_space(simargs['outputs'])
      self.reset()
    except Exception as e:
      print("Error in indoor_env init():", str(e))#, flush=True)
      raise Exception


  def reset(self):
    result = self._sim.reset()
    
    self._episode_info = result.get('episode_info')
    self._last_full_state = result.get('observation')
    img = self._last_full_state['observation']['sensors']['color']['data']
    objective = self._last_full_state.get('measurements') # with measure function!
    state = {'image': self._preprocess_frame(img),
             'objective': objective}
    object_type = self._last_full_state["observation"]["sensors"].get("objectType", None)
    if object_type is not None:
      object_type = object_type["data"][:, :, 2]
      state.update({'objectType': self._preprocess_frame(object_type, "segm")})

    # print(object_type.shape)
    self.last_state = state
    self.last_action = 0
    self.last_reward = 0

    self.prev_state = None
    self.prev_action = 0
    self.prev_reward = 0

  def stop(self):
    if self._sim is not None:
        self._sim.close_game()

  def _preprocess_frame(self, image, mode="segm"):
    if len(image.shape) == 2:  # assume object_type or depth
      image = image.reshape((image.shape[1], image.shape[0]))
      if "segm" in mode:
        image[image == 255] = 0
        return image.astype(np.int32)
      #image = np.dstack([image, image, image])
    else:  # assume rgba
      image = image[:, :, :-1]
    image = image.reshape((image.shape[1], image.shape[0], image.shape[2]))
    #print(image.shape)
    #Reshape is essential, when non-square image from simulator!
    image = image.astype(np.float32)
    image = image / 255.0
    return image

  def process(self, action):
    real_action = IndoorEnvironment.ACTION_LIST[action]

    full_state = self._sim.step(real_action)
    #print("Step made")
    self._last_full_state = full_state  # Last observed state
    obs = full_state['observation']['sensors']['color']['data']
    reward = full_state['rewards'] / self.termination_time # reward clipping
    terminal = full_state['terminals']
    objective = full_state.get('measurements')
    object_type = self._last_full_state["observation"]["sensors"].get("objectType", None)

    if not terminal:
      state = { 'image': self._preprocess_frame(obs),
                'objective': objective }
      if object_type is not None:
        object_type = object_type["data"][:, :, 2]
        state.update({'objectType': self._preprocess_frame(object_type, "segm")})

    else:
      state = self.last_state

    pixel_change = None
    if object_type is None:
      pixel_change = self._calc_pixel_change(state['image'], self.last_state['image'])

    self.prev_state = self.last_state
    self.prev_action = self.last_action
    self.prev_reward = self.last_reward

    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change

  def is_all_scheduled_episodes_done(self):
    return self._sim.is_all_scheduled_episodes_done()
