# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib as contrib

# USEFUL LAYERS
fc = contrib.layers.fully_connected
conv = contrib.layers.conv2d
# convsep = contrib.layers.separable_conv2d
deconv = contrib.layers.conv2d_transpose
relu = tf.nn.relu
maxpool = contrib.layers.max_pool2d
dropout_layer = tf.layers.dropout
batchnorm = contrib.layers.batch_norm
winit = contrib.layers.xavier_initializer()
repeat = contrib.layers.repeat
arg_scope = contrib.framework.arg_scope
l2_regularizer = contrib.layers.l2_regularizer

from layers_object import conv_layer, up_sampling, max_pool, initialization, \
    variable_with_weight_decay
import json


# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


class UnrealModel(object):
  """
  UNREAL algorithm network model.
  """
  def __init__(self,
               action_size,
               objective_size,
               thread_index, # -1 for global
               use_lstm,
               use_pixel_change,
               use_value_replay,
               use_reward_prediction,
               pixel_change_lambda,
               entropy_beta,
               device,
               segnet_param_dict,
               image_shape,
               is_training,
               n_classes,
               segnet_lambda,
               dropout,
               for_display=False):
    self._device = device
    self._action_size = action_size
    self._objective_size = objective_size
    self._thread_index = thread_index
    self._use_lstm = use_lstm
    self._use_pixel_change = use_pixel_change
    self._use_value_replay = use_value_replay
    self._use_reward_prediction = use_reward_prediction
    self._pixel_change_lambda = pixel_change_lambda
    self._entropy_beta = entropy_beta
    self.segnet_mode = segnet_param_dict['segnet_mode']
    self._image_shape = image_shape #[480,360] # [w, h]: Note much of network parameters are hard coded so if we change image shape, other parameters will need to change

    self.for_display = for_display
    self.segnet_lambda = segnet_lambda
    self.dropout = dropout

    self.num_classes = segnet_param_dict.get('num_classes', None)
    self.use_vgg = segnet_param_dict.get('use_vgg', None)
    self.vgg_param_dict = segnet_param_dict.get('vgg_param_dict', None)
    self.bayes = segnet_param_dict.get('bayes', None)

    self.is_training = is_training

    self.n_classes = n_classes
    self.l2 = 2e-4
    self.class_weights = [25.55382055,  2.02904385,  4.22482301, 14.68550062, 26.92127905, 21.18040629,
                          28.08939361, 46.68124414, 46.96535273]
    self.class_weights = [8.5951025,  2.1445557,   7.83135943 , 7.65860916 , 5.84479155 , 7.00514335,
  6.71583023,  9.44302447,  6.52765042 , 9.34331963 , 8.31182911 , 7.85801503,
  9.66735676 , 9.13632663 ,10.0300138,  10.2537406 , 10.36678681 , 9.50484632,
  9.72274774]
    self.class_names =  np.array(['void', 'wall_ceiling_floor_window', 'otherprop', 'arch_door', 'chair', 'sofa',
                                  'bed', 'bathtub', 'toilet'])

    print("Network start creation in thread {}!".format(self._thread_index))#, flush=True)

    self._create_network(for_display)

  def _create_network(self, for_display):
    scope_name = "net_{0}".format(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name):
      # lstm
      self.lstm_cell = contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
      
      # [base A3C network]
      self._create_base_network(for_display)

      # [Pixel change network]
      if self._use_pixel_change:
        self._create_pc_network()
        if for_display:
          self._create_pc_network_for_display()

      # [Value replay network]
      if self._use_value_replay:
        self._create_vr_network()

      # [Reward prediction network]
      if self._use_reward_prediction:
        self._create_rp_network()

      if self.segnet_mode >= 2:
        self._create_mIoU_evaluation_metric_ops()
      
      self.reset_state()

      self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
      self.global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)


  def _create_base_network(self, for_display=False):
    with tf.name_scope("base_network") as scope:
      # State (Base image input)
      self.base_input = tf.placeholder("float", [None, self._image_shape[0], self._image_shape[1], 3], name='base_input')

      # Last action and reward and objective
      self.base_last_action_reward_input = tf.placeholder("float", [None, self._action_size+1+self._objective_size])

      # Conv layers
      base_enc_output = self.encoder(self.base_input)

      if self._use_lstm:
        # LSTM layer
        self.base_initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256], name='base_initial_lstm_state0_0')
        self.base_initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256], name='base_initial_lstm_state0_1')

        self.base_initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.base_initial_lstm_state0,
                                                                     self.base_initial_lstm_state1)

        self.base_lstm_outputs, self.base_lstm_state = \
          self._base_lstm_layer(base_enc_output,
                                self.base_last_action_reward_input,
                                self.base_initial_lstm_state)

        self.base_pi = self._base_policy_layer(self.base_lstm_outputs) # policy output
        self.base_v  = self._base_value_layer(self.base_lstm_outputs)  # value output
      else:
        self.base_fcn_outputs = self._base_fcn_layer(base_enc_output,
                                                     self.base_last_action_reward_input)
        self.base_pi = self._base_policy_layer(self.base_fcn_outputs) # policy output
        self.base_v  = self._base_value_layer(self.base_fcn_outputs)  # value output

      if self.segnet_mode == 2:
        self.base_segm_mask = tf.placeholder("int32", [None] + self._image_shape, name='base_segm_mask')

        self.base_dec_output = self.decoder(base_enc_output)
        self.preds = tf.to_int32(tf.argmax(self.base_dec_output, axis=-1), name="preds")
        self.probs = tf.nn.softmax(self.base_dec_output, name="probs")  # probability distributions

      elif self.segnet_mode == 3 and self._use_lstm:
        self.base_segm_mask = tf.placeholder("int32", [None] + self._image_shape, name='base_segm_mask')

        encoder_shape = base_enc_output.get_shape().as_list()
        num_outputs = np.prod(encoder_shape)
        #input_size = lstm_outputs.get_shape().as_list()
        #lstm_outputs = np.reshape(self.base_lstm_outputs, (1, -1, 256))
        base_fc_from_lstm = tf.reshape(fc(self.base_lstm_outputs, num_outputs, scope="fc_lstm-decoder"), encoder_shape)
        self.base_dec_output = self.decoder(base_fc_from_lstm, scope="from_fc_lstm")

        self.preds = tf.to_int32(tf.argmax(self.base_dec_output, axis=-1), name="preds")
        self.probs = tf.nn.softmax(self.base_dec_output, name="probs")  # probability distributions


  def encoder(self, state_input, reuse=False):
    with tf.variable_scope("base_encoder", reuse=reuse) as scope:
      if self.segnet_mode  == -1:
        raise Exception("No SegNet encoder, use self.segnet_mode > 0")
        # #self.is_training = tf.placeholder_with_default(True, shape=[], name="is_training_pl")
        # #self.with_dropout_pl = tf.placeholder(tf.bool, name="with_dropout")
        # #??self.keep_prob_pl = tf.placeholder(tf.float32, shape=None, name="keep_rate")
        # #self.labels_pl = tf.placeholder(tf.int64, [None, self.input_h, self.input_w, 1])
        #
        # # Before enter the images into the architecture, we need to do Local Contrast Normalization
        # # But it seems a bit complicated, so we use Local Response Normalization which implement in Tensorflow
        # # Reference page:https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
        # self.norm1 = tf.nn.lrn(state_input, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
        # # first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
        # self.conv1_1 = conv_layer(self.norm1, "conv1_1", [3, 3, 3, 64], self.is_training, self.use_vgg,
        #                           self.vgg_param_dict)
        # self.conv1_2 = conv_layer(self.conv1_1, "conv1_2", [3, 3, 64, 64], self.is_training, self.use_vgg,
        #                           self.vgg_param_dict)
        # self.pool1, self.pool1_index, self.shape_1 = max_pool(self.conv1_2, 'pool1')
        #
        # # Second box of convolution layer(4)
        # self.conv2_1 = conv_layer(self.pool1, "conv2_1", [3, 3, 64, 128], self.is_training, self.use_vgg,
        #                           self.vgg_param_dict)
        # self.conv2_2 = conv_layer(self.conv2_1, "conv2_2", [3, 3, 128, 128], self.is_training, self.use_vgg,
        #                           self.vgg_param_dict)
        # self.pool2, self.pool2_index, self.shape_2 = max_pool(self.conv2_2, 'pool2')
        #
        # # Third box of convolution layer(7)
        # self.conv3_1 = conv_layer(self.pool2, "conv3_1", [3, 3, 128, 256], self.is_training, self.use_vgg,
        #                           self.vgg_param_dict)
        # self.conv3_2 = conv_layer(self.conv3_1, "conv3_2", [3, 3, 256, 256], self.is_training, self.use_vgg,
        #                           self.vgg_param_dict)
        # self.conv3_3 = conv_layer(self.conv3_2, "conv3_3", [3, 3, 256, 256], self.is_training, self.use_vgg,
        #                           self.vgg_param_dict)
        # self.pool3, self.pool3_index, self.shape_3 = max_pool(self.conv3_3, 'pool3')
        #
        # # Fourth box of convolution layer(10)
        # if self.bayes:
        #   self.dropout1 = tf.layers.dropout(self.pool3, rate=(1 - self.keep_prob_pl),
        #                                     training=self.with_dropout_pl, name="dropout1")
        #   self.conv4_1 = conv_layer(self.dropout1, "conv4_1", [3, 3, 256, 512], self.is_training, self.use_vgg,
        #                             self.vgg_param_dict)
        # else:
        #   self.conv4_1 = conv_layer(self.pool3, "conv4_1", [3, 3, 256, 512], self.is_training, self.use_vgg,
        #                             self.vgg_param_dict)
        # self.conv4_2 = conv_layer(self.conv4_1, "conv4_2", [3, 3, 512, 512], self.is_training, self.use_vgg,
        #                           self.vgg_param_dict)
        # self.conv4_3 = conv_layer(self.conv4_2, "conv4_3", [3, 3, 512, 512], self.is_training, self.use_vgg,
        #                           self.vgg_param_dict)
        # self.pool4, self.pool4_index, self.shape_4 = max_pool(self.conv4_3, 'pool4')
        #
        # # Fifth box of convolution layers(13)
        # if self.bayes:
        #   self.dropout2 = tf.layers.dropout(self.pool4, rate=(1 - self.keep_prob_pl),
        #                                     training=self.with_dropout_pl, name="dropout2")
        #   self.conv5_1 = conv_layer(self.dropout2, "conv5_1", [3, 3, 512, 512], self.is_training, self.use_vgg,
        #                             self.vgg_param_dict)
        # else:
        #   self.conv5_1 = conv_layer(self.pool4, "conv5_1", [3, 3, 512, 512], self.is_training, self.use_vgg,
        #                             self.vgg_param_dict)
        # self.conv5_2 = conv_layer(self.conv5_1, "conv5_2", [3, 3, 512, 512], self.is_training, self.use_vgg,
        #                           self.vgg_param_dict)
        # self.conv5_3 = conv_layer(self.conv5_2, "conv5_3", [3, 3, 512, 512], self.is_training, self.use_vgg,
        #                           self.vgg_param_dict)
        # self.pool5, self.pool5_index, self.shape_5 = max_pool(self.conv5_3, 'pool5')
        #
        # return self.pool5
      elif self.segnet_mode > 0:
        #n_classes = self.n_classes, alpha = self.alpha, dropout = self.dropout, l2 = self.l2_scale, is_training = self.is_training
        #with tf.name_scope("preprocess") as scope:
        #  x = tf.div(state_input, 255., name="rescaled_inputs")
        x = state_input
        x = self.downsample(x, n_filters=16, is_training=self.is_training, l2=self.l2, name="d1")

        x = self.downsample(x, n_filters=64, is_training=self.is_training, l2=self.l2, name="d2")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 1], l2=self.l2, name="fres3")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 1], l2=self.l2, name="fres4")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 1], l2=self.l2, name="fres5")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 1], l2=self.l2, name="fres6")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 1], l2=self.l2, name="fres7")

        # TODO: Use dilated convolutions
        x = self.downsample(x, n_filters=128, is_training=self.is_training, l2=self.l2, name="d8")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 2], l2=self.l2, name="fres9")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 4], l2=self.l2, name="fres10")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 8], l2=self.l2, name="fres11")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 16], l2=self.l2, name="fres12")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 2], l2=self.l2, name="fres13")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 4], l2=self.l2, name="fres14")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 8], l2=self.l2, name="fres15")
        x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 16], l2=self.l2, name="fres16")
        return x
      else:
        # Weights
        W_conv1, b_conv1 = self._conv_variable([8, 8, 3, 16],  "base_conv1") # 16 8x8 filters
        W_conv2, b_conv2 = self._conv_variable([4, 4, 16, 32], "base_conv2") # 32 4x4 filters

        # Nodes
        h_conv1 = tf.nn.relu(self._conv2d(state_input, W_conv1, 4) + b_conv1) # stride=4 => 19x19x16
        h_conv2 = tf.nn.relu(self._conv2d(h_conv1,     W_conv2, 2) + b_conv2) # stride=2 => 9x9x32
        return h_conv2


  def decoder(self, layer_input, reuse=False, for_display=False, scope=""):
    with tf.variable_scope("base_decoder" + scope, reuse=reuse) as scope:
      x = self.upsample(layer_input, n_filters=64, is_training=self.is_training, l2=self.l2, name="up17")
      x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 1], l2=self.l2, name="fres18")
      x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 1], l2=self.l2, name="fres19")

      x = self.upsample(x, n_filters=16, is_training=self.is_training, l2=self.l2, name="up20")
      x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 1], l2=self.l2, name="fres21")
      x = self.factorized_res_module(x, is_training=self.is_training, dilation=[1, 1], l2=self.l2, name="fres22")

      x = self.upsample(x, n_filters=self.n_classes, is_training=self.is_training, l2=self.l2, name="up23")
      return x

  def _base_fcn_layer(self, conv_output, last_action_reward_objective_input,
                      reuse=False):
    with tf.variable_scope("base_fcn_layer", reuse=reuse) as scope:
      # Weights (9x9x32=2592)
      W_fc1, b_fc1 = self._fc_variable([2592, 256], "base_fc1")

      # Nodes
      conv_output_flat = tf.reshape(conv_output, [-1, 2592])
      # (-1,9,9,32) -> (-1,2592)
      conv_output_fc = tf.nn.relu(tf.matmul(conv_output_flat, W_fc1) + b_fc1)
      # (unroll_step, 256)

      outputs = tf.concat([conv_output_fc, last_action_reward_objective_input], 1)
      return conv_output_fc


  def _base_lstm_layer(self, conv_output, last_action_reward_objective_input, initial_state_input,
                       reuse=False):
    with tf.variable_scope("base_lstm_layer", reuse=reuse) as scope:
      # Weights (9x9x32=2592)
      # 360x480 input: 12x15x512=92160
      #print(tf.shape(conv_output).value)
      conv_output_ravel = np.prod(conv_output.get_shape().as_list()[1:])

      W_fc1, b_fc1 = self._fc_variable([conv_output_ravel, 256], "base_fc1")

      # Nodes
      conv_output_flat = tf.reshape(conv_output, [-1, conv_output_ravel])
      # (-1,9,9,32) -> (-1,2592)
      conv_output_fc = tf.nn.relu(tf.matmul(conv_output_flat, W_fc1) + b_fc1)
      # (unroll_step, 256)

      step_size = tf.shape(conv_output_fc)[:1]

      lstm_input = tf.concat([conv_output_fc, last_action_reward_objective_input], 1)

      # (unroll_step, 256+action_size+1+objective_size)

      lstm_input_reshaped = tf.reshape(lstm_input, [1, -1, 256+self._action_size+1+self._objective_size])
      # (1, unroll_step, 256+action_size+1+objective_size)

      lstm_outputs, lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,
                                                   lstm_input_reshaped,
                                                   initial_state = initial_state_input,
                                                   sequence_length = step_size,
                                                   time_major = False,
                                                   scope = scope)
      
      lstm_outputs = tf.reshape(lstm_outputs, [-1,256])
      #(1,unroll_step,256) for back prop, (1,1,256) for forward prop.
      return lstm_outputs, lstm_state


  def _base_policy_layer(self, lstm_outputs, reuse=False):
    with tf.name_scope("base_policy_layer") as scope:
      input_size = lstm_outputs.get_shape().as_list()[1]
      # Weight for policy output layer
      W_fc_p, b_fc_p = self._fc_variable([input_size, self._action_size], "base_fc_p")
      # Policy (output)
      base_pi = tf.nn.softmax(tf.matmul(lstm_outputs, W_fc_p) + b_fc_p)
      return base_pi


  def _base_value_layer(self, lstm_outputs, reuse=False):
    with tf.variable_scope("base_value_layer", reuse=reuse) as scope:
      input_size = lstm_outputs.get_shape().as_list()[1]
      # Weight for value output layer
      W_fc_v, b_fc_v = self._fc_variable([input_size, 1], "base_fc_v")
      
      # Value (output)
      v_ = tf.matmul(lstm_outputs, W_fc_v) + b_fc_v
      base_v = tf.reshape( v_, [-1] )
      return base_v


  def _create_pc_network(self):
    with tf.name_scope("pc_network") as scope:
      # State (Image input)
      self.pc_input = tf.placeholder("float", [None, self._image_shape[0], self._image_shape[1], 3])

      # Last action and reward and objective
      self.pc_last_action_reward_input = tf.placeholder("float", [None, self._action_size+1+self._objective_size])

      # pc conv layers
      pc_conv_output = self.encoder(self.pc_input, reuse=True)

      if self._use_lstm:
        # pc lstm layers
        pc_initial_lstm_state = self.lstm_cell.zero_state(1, tf.float32)
        # (Initial state is always reset.)

        pc_lstm_outputs, _ = self._base_lstm_layer(pc_conv_output,
                                                   self.pc_last_action_reward_input,
                                                   pc_initial_lstm_state,
                                                   reuse=True)

        self.pc_q, self.pc_q_max = self._pc_deconv_layers(pc_lstm_outputs)
      else:
        pc_fcn_outputs = self._base_fcn_layer(pc_conv_output, self.pc_last_action_reward_input, reuse=True)
        self.pc_q, self.pc_q_max = self._pc_deconv_layers(pc_fcn_outputs)

    
  def _create_pc_network_for_display(self):
    self.pc_q_disp, self.pc_q_max_disp = self._pc_deconv_layers(self.base_lstm_outputs, reuse=True)
    
  
  def _pc_deconv_layers(self, lstm_outputs, reuse=False):
    with tf.variable_scope("pc_deconv_layers", reuse=reuse) as scope:
      input_size = lstm_outputs.get_shape().as_list()[1]
      # (Spatial map was written as 7x7x32, but here 9x9x32 is used to get 20x20 deconv result?)
      # State (image input for pixel change)
      W_pc_fc1, b_pc_fc1 = self._fc_variable([input_size, 9*9*32], "pc_fc1")
        
      W_pc_deconv_v, b_pc_deconv_v = self._conv_variable([4, 4, 1, 32],
                                                         "pc_deconv_v", deconv=True)
      W_pc_deconv_a, b_pc_deconv_a = self._conv_variable([4, 4, self._action_size, 32],
                                                         "pc_deconv_a", deconv=True)
      
      h_pc_fc1 = tf.nn.relu(tf.matmul(lstm_outputs, W_pc_fc1) + b_pc_fc1)
      h_pc_fc1_reshaped = tf.reshape(h_pc_fc1, [-1,9,9,32])
      # Dueling network for V and Advantage
      h_pc_deconv_v = tf.nn.relu(self._deconv2d(h_pc_fc1_reshaped,
                                                W_pc_deconv_v, 9, 9, 2) +
                                 b_pc_deconv_v)
      h_pc_deconv_a = tf.nn.relu(self._deconv2d(h_pc_fc1_reshaped,
                                                W_pc_deconv_a, 9, 9, 2) +
                                 b_pc_deconv_a)
      # Advantage mean
      h_pc_deconv_a_mean = tf.reduce_mean(h_pc_deconv_a, reduction_indices=3, keep_dims=True)

      # {Pixel change Q (output)
      pc_q = h_pc_deconv_v + h_pc_deconv_a - h_pc_deconv_a_mean
      #(-1, 20, 20, action_size)

      # Max Q
      pc_q_max = tf.reduce_max(pc_q, reduction_indices=3, keep_dims=False)
      #(-1, 20, 20)

      return pc_q, pc_q_max
    

  def _create_vr_network(self):
    with tf.name_scope("vr_network") as scope:
      # State (Image input)
      self.vr_input = tf.placeholder("float", [None, self._image_shape[0], self._image_shape[1], 3])

      # Last action and reward and objective
      self.vr_last_action_reward_input = tf.placeholder("float", [None, self._action_size+1+self._objective_size])

      # VR conv layers
      vr_conv_output = self.encoder(self.vr_input, reuse=True)

      if self._use_lstm:
        # pc lstm layers
        vr_initial_lstm_state = self.lstm_cell.zero_state(1, tf.float32)
        # (Initial state is always reset.)

        vr_lstm_outputs, _ = self._base_lstm_layer(vr_conv_output,
                                                   self.vr_last_action_reward_input,
                                                   vr_initial_lstm_state,
                                                   reuse=True)
        # value output
        self.vr_v  = self._base_value_layer(vr_lstm_outputs, reuse=True)
      else:
        vr_fcn_outputs = self._base_fcn_layer(vr_conv_output, self.vr_last_action_reward_input, reuse=True)
        self.vr_v = self._base_value_layer(vr_fcn_outputs, reuse=True)

    
  def _create_rp_network(self):
    with tf.name_scope("rp_network") as scope:
      self.rp_input = tf.placeholder("float", [3, self._image_shape[0], self._image_shape[1], 3])

      # RP conv layers
      rp_conv_output = self.encoder(self.rp_input, reuse=True)
      rp_conv_output_ravel = np.prod(rp_conv_output.get_shape().as_list())
      rp_conv_output_reshaped = tf.reshape(rp_conv_output, [1,rp_conv_output_ravel])

      with tf.variable_scope("rp_fc") as scope:
        # Weights
        W_fc1, b_fc1 = self._fc_variable([rp_conv_output_ravel, 3], "rp_fc1")

      # Reawrd prediction class output. (zero, positive, negative)
      self.rp_c = tf.nn.softmax(tf.matmul(rp_conv_output_reshaped, W_fc1) + b_fc1)
      # (1,3)

  def _base_loss(self):
    # [base A3C]
    # Taken action (input for policy)
    self.base_a = tf.placeholder("float", [None, self._action_size], name='base_a')
    
    # Advantage (R-V) (input for policy)
    self.base_adv = tf.placeholder("float", [None], name='base_adv')
    
    # Avoid NaN with clipping when value in pi becomes zero
    log_pi = tf.log(tf.clip_by_value(self.base_pi, 1e-20, 1.0))
    
    # Policy entropy
    self.entropy = -tf.reduce_sum(self.base_pi * log_pi, reduction_indices=1)
    
    # Policy loss (output)
    self.policy_loss = -tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.base_a ),
                                                 reduction_indices=1 ) *
                                  self.base_adv + self.entropy * self._entropy_beta)
    
    # R (input for value target)
    self.base_r = tf.placeholder("float", [None], name='base_r')
    
    # Value loss (output)
    # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
    self.value_loss = 0.5 * tf.nn.l2_loss(self.base_r - self.base_v)


    self.base_loss = self.policy_loss + self.value_loss
    decoder_loss = None
    
    if self.segnet_mode >= 2:
      unrolled_logits = tf.reshape(self.base_dec_output, (-1, self.n_classes))
      unrolled_labels = tf.reshape(self.base_segm_mask, (-1,))

      # HANDLE CLASS WEIGHTS
      if self.class_weights is not None:
        class_weights_tensor = tf.constant(self.class_weights, dtype=tf.float32)
        label_weights = tf.gather(class_weights_tensor, indices=unrolled_labels)
        print("- Using Class Weights: \n", self.class_weights)
      else:
        label_weights = 1.0
        print("- Using uniform Class Weights of 1.0")

      # CACLULATE LOSSES
      self.decoder_loss = tf.losses.sparse_softmax_cross_entropy(labels=unrolled_labels, logits=unrolled_logits, weights=label_weights,
                                           reduction="weighted_sum_by_nonzero_weights", scope="decoder_loss")
      self.regul_loss = tf.losses.get_regularization_loss(scope="net_{0}".format(self._thread_index))
      self.decoder_loss += self.regul_loss
      self.decoder_loss *= self.segnet_lambda
      # SUMS ALL LOSSES - even Regularization losses automatically
      decoder_loss = self.decoder_loss

    return self.base_loss, decoder_loss

  
  def _pc_loss(self):
    # [pixel change]
    self.pc_a = tf.placeholder("float", [None, self._action_size], name='pc_a')
    pc_a_reshaped = tf.reshape(self.pc_a, [-1, 1, 1, self._action_size])

    # Extract Q for taken action
    pc_qa_ = tf.multiply(self.pc_q, pc_a_reshaped)
    pc_qa = tf.reduce_sum(pc_qa_, reduction_indices=3, keep_dims=False)
    # (-1, 20, 20)
      
    # TD target for Q
    self.pc_r = tf.placeholder("float", [None, 20, 20], name='pc_r')

    pc_loss = self._pixel_change_lambda * tf.nn.l2_loss(self.pc_r - pc_qa)
    return pc_loss

  
  def _vr_loss(self):
    # R (input for value)
    self.vr_r = tf.placeholder("float", [None], name='vr_r')
    
    # Value loss (output)
    vr_loss = tf.nn.l2_loss(self.vr_r - self.vr_v)
    return vr_loss


  def _rp_loss(self):
    # reward prediction target. one hot vector
    self.rp_c_target = tf.placeholder("float", [1,3], name='rp_c_target')
    
    # Reward prediction loss (output)
    rp_c = tf.clip_by_value(self.rp_c, 1e-20, 1.0)
    rp_loss = -tf.reduce_sum(self.rp_c_target * tf.log(rp_c))
    return rp_loss
    
    
  def prepare_loss(self):
    with tf.device(self._device):
      base_loss, decoder_loss = self._base_loss()
      loss = base_loss
      if decoder_loss is not None:
        loss += decoder_loss


      if self._use_pixel_change:
        self.pc_loss = self._pc_loss()
        loss = loss + self.pc_loss

      if self._use_value_replay:
        self.vr_loss = self._vr_loss()
        loss = loss + self.vr_loss

      if self._use_reward_prediction:
        self.rp_loss = self._rp_loss()
        loss = loss + self.rp_loss
      
      self.total_loss = loss

  def _create_mIoU_evaluation_metric_ops(self):
    # EVALUATION METRIC - IoU
    with tf.name_scope("evaluation") as scope:
      # Define the evaluation metric and update operations
      self.evaluation, self.update_evaluation_vars = tf.metrics.mean_iou(
        tf.reshape(self.base_segm_mask, [-1]),
        tf.reshape(self.preds, [-1]),
        num_classes=self.n_classes,
        name=scope)
      # Isolate metric's running variables & create their initializer/reset op
      evaluation_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
      self.reset_evaluation_vars = tf.variables_initializer(var_list=evaluation_vars)

  def dont_use_create_acc_evaluation_metric_ops(self):
    # EVALUATION METRIC
    with tf.name_scope("evaluation") as scope:
      # Define the evaluation metric and update operations
      self.evaluation, self.update_evaluation_vars = tf.metrics.accuracy(
        labels=tf.reshape(self.base_segm_mask, [-1]),
        predictions=tf.reshape(self.preds, [-1]),
        name=scope)
      # Isolate metric's running variables & create their initializer/reset op
      evaluation_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
      self.reset_evaluation_vars = tf.variables_initializer(var_list=evaluation_vars)

  def reset_state(self):
    if self._use_lstm:
      self.base_lstm_state_out = contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
                                                            np.zeros([1, 256]))

  def run_base_policy_and_value(self, sess, s_t, last_action_reward, mode=""):
    # This run_base_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    if self._use_lstm:
      if "segnet" in mode:
        pi_out, v_out, self.base_lstm_state_out, preds = sess.run([self.base_pi, self.base_v,
                                                                   self.base_lstm_state, self.preds],
                                                           feed_dict={self.base_input: [s_t['image']],
                                                                      self.is_training: not self.for_display,
                                                                      self.base_last_action_reward_input: [last_action_reward],
                                                                      self.base_initial_lstm_state0: self.base_lstm_state_out[0],
                                                                      self.base_initial_lstm_state1: self.base_lstm_state_out[1]})
        return (pi_out[0], v_out[0], preds.reshape(self._image_shape))# {'base_loss':base_loss, 'decoder_loss':decoder_loss})

      else:
        pi_out, v_out, self.base_lstm_state_out = sess.run([self.base_pi, self.base_v, self.base_lstm_state],
                                                            feed_dict={self.base_input: [s_t['image']],
                                                                       self.is_training: not self.for_display,
                                                                       self.base_last_action_reward_input:  [last_action_reward],
                                                                       self.base_initial_lstm_state0: self.base_lstm_state_out[0],
                                                                       self.base_initial_lstm_state1: self.base_lstm_state_out[1]})
        return (pi_out[0], v_out[0], None) #{'base_loss':base_loss})
    else:
      pi_out, v_out = sess.run([self.base_pi, self.base_v],
                               feed_dict = {self.base_input : [s_t['image']],
                                            self.is_training: not self.for_display,
                                            self.base_last_action_reward_input : [last_action_reward]} )


      # pi_out: (1,3), v_out: (1), probs_out: None or (1, h, w, C)
      return (pi_out[0], v_out[0], None)


  def run_base_policy_value_pc_q(self, sess, s_t, last_action_reward):
    # For display tool.
    if self._use_lstm:
      pi_out, v_out, self.base_lstm_state_out, q_disp_out, q_max_disp_out = \
          sess.run( [self.base_pi, self.base_v, self.base_lstm_state, self.pc_q_disp,
                     self.pc_q_max_disp],
                    feed_dict = {self.base_input : [s_t['image']],
                                 self.is_training: not self.for_display,
                                 self.base_last_action_reward_input : [last_action_reward],
                                 self.base_initial_lstm_state0 : self.base_lstm_state_out[0],
                                 self.base_initial_lstm_state1 : self.base_lstm_state_out[1]})
      # pi_out: (1,3), v_out: (1), q_disp_out(1,20,20, action_size)
      return (pi_out[0], v_out[0], q_disp_out[0])
    else:
      pi_out, v_out, q_disp_out, q_max_disp_out = \
        sess.run( [self.base_pi, self.base_v, self.pc_q_disp, self.pc_q_max_disp],
                  feed_dict = {self.base_input : [s_t['image']],
                               self.is_training: not self.for_display,
                               self.base_last_action_reward_input : [last_action_reward] })

      # pi_out: (1,3), v_out: (1), q_disp_out(1,20,20, action_size)
      return (pi_out[0], v_out[0], q_disp_out[0])


  def run_base_value(self, sess, s_t, last_action_reward):
    # This run_base_value() is used for calculating V for bootstrapping at the
    # end of LOCAL_T_MAX time step sequence.
    # When next sequence starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    if self._use_lstm:
      v_out, _ = sess.run( [self.base_v, self.base_lstm_state],
                           feed_dict = {self.base_input : [s_t['image']],
                                        self.is_training: not self.for_display,
                                        self.base_last_action_reward_input : [last_action_reward],
                                        self.base_initial_lstm_state0 : self.base_lstm_state_out[0],
                                        self.base_initial_lstm_state1 : self.base_lstm_state_out[1]} )
    else:
      v_out = sess.run( self.base_v,
                        feed_dict = {self.base_input : [s_t['image']],
                                     self.is_training: not self.for_display,
                                     self.base_last_action_reward_input : [last_action_reward]} )
    return v_out[0]


  def run_pc_q_max(self, sess, s_t, last_action_reward):
    q_max_out = sess.run( self.pc_q_max,
                          feed_dict = {self.pc_input : [s_t['image']],
                                       self.is_training: not self.for_display,
                                       self.pc_last_action_reward_input : [last_action_reward]} )
    return q_max_out[0]


  def run_vr_value(self, sess, s_t, last_action_reward):
    vr_v_out = sess.run( self.vr_v,
                         feed_dict = {self.vr_input : [s_t['image']],
                                      self.is_training: not self.for_display,
                                      self.vr_last_action_reward_input : [last_action_reward]} )
    return vr_v_out[0]


  def run_rp_c(self, sess, state_history):
    # For display tool
    frames = [s_t['image'] for s_t in state_history]
    rp_c_out = sess.run( self.rp_c,
                         feed_dict = {self.rp_input : frames, self.is_training: not self.for_display} )
    return rp_c_out[0]

  
  def get_vars(self):
    return self.variables

  def get_global_vars(self):
    return self.global_variables

  def sync_from(self, src_network, name=None):
    src_vars = src_network.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "UnrealModel",[]) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)
      

  def _fc_variable(self, weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
    return weight, bias

  
  def _conv_variable(self, weight_shape, name, deconv=False):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
      input_channels  = weight_shape[3]
      output_channels = weight_shape[2]
    else:
      input_channels  = weight_shape[2]
      output_channels = weight_shape[3]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape,
                             initializer=conv_initializer(w, h, input_channels))
    bias   = tf.get_variable(name_b, bias_shape,
                             initializer=conv_initializer(w, h, input_channels))
    return weight, bias

  
  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")


  def _get2d_deconv_output_size(self,
                                input_height, input_width,
                                filter_height, filter_width,
                                stride, padding_type):
    if padding_type == 'VALID':
      out_height = (input_height - 1) * stride + filter_height
      out_width  = (input_width  - 1) * stride + filter_width
      
    elif padding_type == 'SAME':
      out_height = input_height * stride
      out_width  = input_width  * stride
    
    return out_height, out_width


  def _deconv2d(self, x, W, input_width, input_height, stride):
    filter_height = W.get_shape()[0].value
    filter_width  = W.get_shape()[1].value
    out_channel   = W.get_shape()[2].value
    
    out_height, out_width = self._get2d_deconv_output_size(input_height,
                                                           input_width,
                                                           filter_height,
                                                           filter_width,
                                                           stride,
                                                           'VALID')
    batch_size = tf.shape(x)[0]
    output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='VALID')

  def get_conv_arg_scope(self, is_training, bn=True, reg=None, use_deconv=False, use_relu=True, bn_decay=0.9):
    with arg_scope(
        [deconv if use_deconv else conv],
        padding="SAME",
        stride=1,
        activation_fn=relu if use_relu else None,
        normalizer_fn=batchnorm if bn else None,
        normalizer_params={"is_training": is_training, "decay": bn_decay},
        weights_regularizer=reg,
        variables_collections=None,
    ) as scope:
      return scope

  def factorized_res_moduleOLD(self, x, is_training, dropout=0.3, dilation=1, name="fres"):
    with arg_scope(self.get_conv_arg_scope(is_training=is_training, bn=True)):
      with tf.variable_scope(name):
        n_filters = x.shape.as_list()[-1]
        y = conv(x, num_outputs=n_filters, kernel_size=[3, 1], normalizer_fn=None, scope="conv_a_3x1")
        y = conv(y, num_outputs=n_filters, kernel_size=[1, 3], scope="conv_a_1x3")
        y = conv(y, num_outputs=n_filters, kernel_size=[3, 1], rate=dilation, normalizer_fn=None, scope="conv_b_3x1")
        y = conv(y, num_outputs=n_filters, kernel_size=[1, 3], rate=dilation, scope="conv_b_1x3")
        y = dropout_layer(y, rate=self.dropout)
        y = tf.add(x, y, name="add")
    # print("DEBUG: {} {}".format(name, y.shape.as_list()))
    return y

  def factorized_res_module(self, x, is_training, dropout=0.3, dilation=[1, 1], l2=None, name="fres"):
    reg = None if l2 is None else l2_regularizer(l2)
    with arg_scope(self.get_conv_arg_scope(reg=reg, is_training=is_training, bn=True)):
      with tf.variable_scope(name):
        n_filters = x.shape.as_list()[-1]
        y = conv(x, num_outputs=n_filters, kernel_size=[3, 1], rate=dilation[0], normalizer_fn=None, scope="conv_a_3x1")
        y = conv(y, num_outputs=n_filters, kernel_size=[1, 3], rate=dilation[0], scope="conv_a_1x3")
        y = conv(y, num_outputs=n_filters, kernel_size=[3, 1], rate=dilation[1], normalizer_fn=None, scope="conv_b_3x1")
        y = conv(y, num_outputs=n_filters, kernel_size=[1, 3], rate=dilation[1], scope="conv_b_1x3")
        y = dropout_layer(y, rate=self.dropout)
        y = tf.add(x, y, name="add")
    # print("DEBUG: {} {}".format(name, y.shape.as_list()))
    # print("DEBUG: L2 in factorized res module {}".format(l2))
    return y

  def downsample(self, x, n_filters, is_training, bn=False, use_relu=False, l2=None, name="down"):
    with tf.variable_scope(name):
      reg = None if l2 is None else l2_regularizer(l2)
      with arg_scope(self.get_conv_arg_scope(reg=reg, is_training=is_training, bn=bn, use_relu=use_relu)):
        n_filters_in = x.shape.as_list()[-1]
        n_filters_conv = n_filters - n_filters_in
        branch_a = conv(x, num_outputs=n_filters_conv, kernel_size=3, stride=2, scope="conv")
        branch_b = maxpool(x, kernel_size=2, stride=2, padding='VALID', scope="maxpool")
        y = tf.concat([branch_a, branch_b], axis=-1, name="concat")
    # print("DEBUG: {} {}".format(name, y.shape.as_list()))
    return y

  def upsample(self, x, n_filters, is_training=False, use_relu=False, bn=False, l2=None, name="up"):
    reg = None if l2 is None else l2_regularizer(l2)
    with arg_scope(self.get_conv_arg_scope(reg=reg, is_training=is_training, bn=bn, use_deconv=True, use_relu=use_relu)):
      y = deconv(x, num_outputs=n_filters, kernel_size=4, stride=2, scope=name)
    # print("DEBUG: {} {}".format(name, y.shape.as_list()))
    return y
