# -*- coding: utf-8 -*-
from abc import ABC
import tensorflow as tf


class TextCNN(tf.keras.Model, ABC):
    """
    TextCNN模型
    """
    def __init__(self, num_classes, embedding_dim, vocab_size, embeddings_matrix=None):
        super(TextCNN, self).__init__()
        num_filters = 64
        seq_length = 300
        self.embedding_method = ""
        self.dropout_rate = 0.2
        if self.embedding_method is '':
            self.embedding = tf.keras.layers.Embedding(vocab_size+1, embedding_dim, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, weights=[embeddings_matrix],
                                                       trainable=False)

        self.conv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[2, embedding_dim],
                                            strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[seq_length-2+1, 1], padding='valid')

        self.conv2 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[3, embedding_dim], strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[seq_length-3+1, 1], padding='valid')

        self.conv3 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[4, embedding_dim], strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=[seq_length-4+1, 1], padding='valid')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')
        self.dense = tf.keras.layers.Dense(num_classes,
                                           activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.2),
                                           bias_regularizer=tf.keras.regularizers.l2(0.2),
                                           name='dense')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')

    @tf.function
    def call(self, inputs, training=None):
        inputs = self.embedding(inputs)

        inputs = tf.expand_dims(inputs, -1)
        pooled_output = []
        con1 = self.conv1(inputs)
        pool1 = self.pool1(con1)
        pooled_output.append(pool1)

        con2 = self.conv2(inputs)
        pool2 = self.pool2(con2)
        pooled_output.append(pool2)

        con3 = self.conv3(inputs)
        pool3 = self.pool3(con3)
        pooled_output.append(pool3)

        concat_outputs = tf.keras.layers.concatenate(pooled_output, axis=-1, name='concatenate')
        flatten_outputs = self.flatten(concat_outputs)
        dropout_outputs = self.dropout(flatten_outputs, training)
        outputs = self.dense(dropout_outputs)
        return outputs
