# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 23:02:00 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""

import tensorflow as tf
im_size=33
n=3
#layer1 conv
f1=9
stride1=1
n1=32
#layer2 conv
f2=1
stride2=1
n2=64
#layer3 conv
f3=5
stride3=1
n3=3

def inference(input_tensor,regularizer=None):
    with tf.variable_scope('layer1'):
        conv1_weight=tf.get_variable("weight",
                                     [f1,f1,n,n1],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.variance_scaling_initializer())
        conv1_bais=tf.get_variable("bais",
                                   [n1],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0))
        conv1=tf.nn.conv2d(input_tensor,conv1_weight,strides=[1,stride1,stride1,1],padding='VALID')
        Z1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_bais))
    with tf.variable_scope('layer2'):
        conv2_weight=tf.get_variable("weight",
                                     [f2,f2,n1,n2],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.variance_scaling_initializer())
        conv2_bais=tf.get_variable("bais",
                                   [n2],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0))
        conv2=tf.nn.conv2d(Z1,conv2_weight,strides=[1,stride2,stride2,1],padding='VALID')
        Z2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_bais))
    with tf.variable_scope('layer3'):
        conv3_weight=tf.get_variable("weight",
                                     [f3,f3,n2,n3],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.variance_scaling_initializer())
        conv3_bais=tf.get_variable("bais",
                                   [n3],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0))
        conv3=tf.nn.conv2d(Z2,conv3_weight,strides=[1,stride3,stride3,1],padding='VALID')
        y_hat=tf.nn.bias_add(conv3,conv3_bais)
    return y_hat