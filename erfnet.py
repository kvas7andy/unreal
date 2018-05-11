from __future__ import print_function, division, unicode_literals
import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib



def romeranetB_logits(X, Y, n_classes, alpha=0.001, dropout=0.3, l2=None, is_training=False):
    """
    """
    # TODO: Use TF repeat for some of these repeating layers
    # TODO: register factorized_res_module and upsample and downsample with arg_scope
    #       and pass dropout, is_training, etc to it.
    # TODO: Add weight decay.
    with tf.variable_scope("preprocess") as scope:
        x = tf.div(X, 255., name="rescaled_inputs")

    x = downsample(x, n_filters=16, is_training=is_training, name="d1")

    x = downsample(x, n_filters=64, is_training=is_training, name="d2")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=1, name="fres3")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=1, name="fres4")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=1, name="fres5")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=1, name="fres6")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=1, name="fres7")

    # TODO: Use dilated convolutions
    x = downsample(x, n_filters=128, is_training=is_training, name="d8")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=2, name="fres9")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=4, name="fres10")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=8, name="fres11")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=16, name="fres12")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=2, name="fres13")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=4, name="fres14")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=8, name="fres15")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=16, name="fres16")

    x = upsample(x, n_filters=64, is_training=is_training, name="up17")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=1, name="fres18")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=1, name="fres19")

    x = upsample(x, n_filters=16, is_training=is_training, name="up20")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=1, name="fres21")
    x = factorized_res_moduleOLD(x, is_training=is_training, dilation=1, name="fres22")

    x = upsample(x, n_filters=n_classes, is_training=is_training, name="up23")
    return x


def erfnetA(X, Y, n_classes, alpha=0.001, dropout=0.3, l2=None, is_training=False):
    """
    """
    # TODO: Use TF repeat for some of these repeating layers
    # TODO: register factorized_res_module and upsample and downsample with arg_scope
    #       and pass dropout, is_training, etc to it.
    # TODO: Add weight decay.
    with tf.variable_scope("preprocess") as scope:
        x = tf.div(X, 255., name="rescaled_inputs")

    x = downsample(x, n_filters=16, is_training=is_training, name="d1")

    x = downsample(x, n_filters=64, is_training=is_training, name="d2")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres3")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres4")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres5")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres6")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres7")

    # TODO: Use dilated convolutions
    x = downsample(x, n_filters=128, is_training=is_training, name="d8")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 2], name="fres9")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 4], name="fres10")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 8], name="fres11")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 16], name="fres12")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 2], name="fres13")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 4], name="fres14")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 8], name="fres15")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 16], name="fres16")

    x = upsample(x, n_filters=64, is_training=is_training, name="up17")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres18")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres19")

    x = upsample(x, n_filters=16, is_training=is_training, name="up20")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres21")
    x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], name="fres22")

    x = upsample(x, n_filters=n_classes, is_training=is_training, name="up23")
    return x


def erfnetB(X, Y, n_classes, alpha=0.001, dropout=0.3, l2=None, is_training=False, encoder=True, decoder=True, decoder_input=None):
    """
    Uses L2 regularization.
    """
    #print("DEBUG: L2 passed on to ERFNETB{}".format(l2))
    # TODO: Use TF repeat for some of these repeating layers
    # TODO: register factorized_res_module and upsample and downsample with arg_scope
    #       and pass dropout, is_training, etc to it.
    # TODO: Add weight decay.

    if encoder:
        with tf.variable_scope("preprocess") as scope:
            x = tf.div(X, 255., name="rescaled_inputs")

        x = downsample(x, n_filters=16, is_training=is_training, l2=l2, name="d1")

        x = downsample(x, n_filters=64, is_training=is_training, l2=l2, name="d2")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres3")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres4")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres5")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres6")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres7")

        # TODO: Use dilated convolutions
        x = downsample(x, n_filters=128, is_training=is_training, l2=l2, name="d8")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 2], l2=l2, name="fres9")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 4], l2=l2, name="fres10")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 8], l2=l2, name="fres11")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 16], l2=l2, name="fres12")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 2], l2=l2, name="fres13")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 4], l2=l2, name="fres14")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 8], l2=l2, name="fres15")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 16], l2=l2, name="fres16")

    if decoder:
        if not encoder:
            x = decoder_input

        x = upsample(x, n_filters=64, is_training=is_training, l2=l2, name="up17")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres18")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres19")

        x = upsample(x, n_filters=16, is_training=is_training, l2=l2, name="up20")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres21")
        x = factorized_res_module(x, is_training=is_training, dilation=[1, 1], l2=l2, name="fres22")

        x = upsample(x, n_filters=n_classes, is_training=is_training, l2=l2, name="up23")

    return x
