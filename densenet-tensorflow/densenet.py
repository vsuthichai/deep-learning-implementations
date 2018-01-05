import tensorflow as tf
import numpy as np
from six.moves import xrange
import datetime as dt
import os
import math

def dense_block_layers(x, num_dense_layer, growth_rate, dropout):
    N, H, W, C = x.shape

    with tf.variable_scope("batch_norm_{}".format(num_dense_layer)) as batch_norm_scope:
        mean, var = tf.nn.moments(x, axes=[0, 1, 2]) 
        beta = tf.Variable(tf.zeros((C.value)), name="beta")
        gamma = tf.Variable(tf.ones((C.value)), name="gamma")
        bn = tf.nn.batch_normalization(x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-8, name="bn")
    with tf.variable_scope("relu_{}".format(num_dense_layer)) as relu_scope:
        relu = tf.nn.relu(bn, name="relu")
    with tf.variable_scope("conv2d_{}".format(num_dense_layer)) as conv2d_scope:
        filters = tf.Variable(tf.random_normal((3, 3, C.value, growth_rate), stddev=(2. / (3 * 3 * C.value)) ** .5), name="conv2d_filters")
        conv2d = tf.nn.conv2d(input=relu, filter=filters, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", name="conv2d")
    with tf.variable_scope("dropout_{}".format(num_dense_layer)) as dropout_scope:
        return tf.nn.dropout(conv2d, keep_prob=dropout, name="dropout")

def dense_block(x, num_block, num_dense_layers, growth_rate, keep_prob):
    with tf.variable_scope("dense_block_{}".format(num_block)) as dense_block_scope:
        for num_dense_layer in xrange(num_dense_layers):
            next_x = dense_block_layers(x, num_dense_layer + 1, growth_rate, keep_prob)
            with tf.variable_scope("concat") as concat_scope:
                x = tf.concat([x, next_x], axis=3)
        return x

def transition_layers(x, num_transition_layer, keep_prob):
    with tf.variable_scope("transition_{}".format(num_transition_layer)) as transition_scope:
        N, H, W, C = x.shape
        mean, var = tf.nn.moments(x, axes=[0, 1, 2])
        beta = tf.Variable(tf.zeros((C.value)), name="beta")
        gamma = tf.Variable(tf.ones((C.value)), name="gamma")
        bn = tf.nn.batch_normalization(x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-8, name="batch_norm")
        relu = tf.nn.relu(bn)
        filters = tf.Variable(tf.random_normal((1, 1, C.value, C.value), stddev=(2. / C.value) ** .5), name="conv2d_filters")
        conv2d = tf.nn.conv2d(relu, filter=filters, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", name="conv2d")
        dropout = tf.nn.dropout(conv2d, keep_prob=0.8, name="dropout")
        avg_pool = tf.nn.avg_pool(dropout, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", data_format="NHWC", name="avg_pool")
        return avg_pool

def classification_layer(x, y):
    with tf.variable_scope("fc1") as fc1_scope:
        N, H, W, C = x.shape
        num_features = H.value * W.value * C.value

        mean, var = tf.nn.moments(x, axes=[0, 1, 2])
        beta = tf.Variable(tf.zeros((C.value)), name="beta")
        gamma = tf.Variable(tf.ones((C.value)), name="gamma")
        bn = tf.nn.batch_normalization(x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-8, name="batch_norm")
        relu = tf.nn.relu(bn)
        avg_pool = tf.nn.avg_pool(relu, ksize=(1, H.value, W.value, 1), strides=(1, 1, 1, 1), padding="VALID", data_format="NHWC", name="global_avg_pool")

        features = tf.reshape(avg_pool, shape=(-1, C.value))
        W = tf.Variable(tf.random_normal((C.value, 10), stddev=(1. / C.value) ** .5), name="fc_weights")
        b = tf.Variable(tf.zeros(10), name="fc_bias")
        return tf.matmul(features, W) + b

def densenet_model(growth_rate):
    with tf.variable_scope("placeholders") as placeholder_scope:
        X = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name="input")
        y = tf.placeholder(dtype=tf.int32, shape=(None,), name="labels")
        keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name="keep_prob")
        learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate")
        regularization = tf.placeholder(dtype=tf.float32, shape=(), name="weight_decay")

    with tf.variable_scope("preprocess") as preproces_scope:
        X_mean, X_var = tf.nn.moments(X, axes=[0], name="moments")
        X_normalized = (X - X_mean) / tf.sqrt(X_var)

    with tf.variable_scope("init_conv2d") as init_conv2d_scope:
        init_conv2d_filters = tf.Variable(tf.random_normal((3, 3, 3, 16), stddev=(2. / (3 * 3 * 3)) ** .5), name="init_conv2d_filters")
        init_conv2d = tf.nn.conv2d(input=X_normalized, filter=init_conv2d_filters, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
        
    dense_block_1 = dense_block(init_conv2d, num_block=1, num_dense_layers=12, growth_rate=growth_rate, keep_prob=keep_prob)
    transition_layer_1 = transition_layers(dense_block_1, num_transition_layer=1, keep_prob=keep_prob)
    dense_block_2 = dense_block(transition_layer_1, num_block=2, num_dense_layers=12, growth_rate=growth_rate, keep_prob=keep_prob)
    transition_layer_2 = transition_layers(dense_block_2, num_transition_layer=2, keep_prob=keep_prob)
    dense_block_3 = dense_block(transition_layer_2, num_block=3, num_dense_layers=12, growth_rate=growth_rate, keep_prob=keep_prob)
    logits = classification_layer(dense_block_3, y)

    with tf.variable_scope("loss") as loss_scope:
        softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y, 10), name="softmax")
        avg_softmax_loss = tf.reduce_mean(softmax_loss, name="avg_softmax_loss")

    with tf.variable_scope("optimizer") as optimizer_scope:
        #train_step = tf.train.AdamOptimizer(1e-3).minimize(avg_softmax_loss)
        l2_loss = tf.add_n([ tf.nn.l2_loss(variable) for variable in tf.trainable_variables() ])
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(avg_softmax_loss + (l2_loss * regularization))

    with tf.variable_scope("accuracy") as accuracy_scope:
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32, name="predict")
        correct_predictions = tf.cast(tf.equal(predictions, y), tf.float32)
        count_correct_predictions = tf.reduce_sum(correct_predictions)
        accuracy = tf.reduce_mean(correct_predictions)

    with tf.variable_scope("batch_summary") as summary_scope:
        tf.summary.scalar("avg_softmax_loss", avg_softmax_loss)
        tf.summary.scalar("accuracy", accuracy)
        summaries = tf.summary.merge_all()

    return {
        'X': X, 
        'y': y,
        'keep_prob': keep_prob, 
        'learning_rate': learning_rate, 
        'regularization': regularization, 
        'predictions': correct_predictions,
        'correct_predictions': count_correct_predictions,
        'accuracy': accuracy, 
        'loss': avg_softmax_loss, 
        'train_step': train_step, 
        'summaries': summaries
    }

