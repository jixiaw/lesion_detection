# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 14:53:12 2019

@author: wjcongyu
"""

import os
import os.path as osp
import tensorflow as tf
from tensorflow import math
from tensorflow import keras
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from tensorflow.keras import losses as KLOSS
from tensorflow.keras import backend as KB
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import losses_utils
import glob
import numpy as np
import sys
from model import CenterNet, CenterNet3d


class Mediastinal_3dcenternet():
    def __init__(self, input_shape, is_training, num_classes, model_dir, config):
        self._is_training = is_training
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.NET_NAME = 'Mediastinal_3dcenternet'
        self.__set_log_dir()  # logging and saving checkpoints
        self.model = self.__build(is_training=is_training)
        # self.model = CenterNet3d(num_class=1)
        # self.model = CenterNet(num_class=1)
    # public functions
    def summary(self):
        '''
        print the network attributes
        :return:
        '''
        return self.model.summary()

    def find_last(self, model_path=None):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        if model_path is None:
            weights_files = glob.glob(osp.join(self.log_dir, self.NET_NAME.lower() + '*.h5'))
        else:
            weights_files = glob.glob(osp.join(model_path, self.NET_NAME.lower(), self.NET_NAME.lower() + '*.h5'))
        if len(weights_files) == 0:
            return ''
        weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(x))
        return weights_files[-1]

    def load_weights(self, filepath, by_name=False, exclude=None):
        '''
        loading weights from checkpoint
        :param filepath:
        :param by_name:
        :param exclude:
        :return:
        '''
        # self.model(tf.ones((1, 320, 320, 320, 1)))  # 初始化模型，不加的话加载模型会报错
        self.model.load_weights(filepath, by_name)

    def train(self, data_provider, test_data_provider, learning_rate, decay_steps, epochs, batch_size,
              augment=None, custom_callbacks=None):
        '''
        Start training the model from specified dataset
        :param train_dataset:
        :param learning_rate:
        :param decay_steps:
        :param epochs:
        :param augment:
        :param custom_callbacks:
        :return:
        '''
        assert self._is_training == True, 'not in training mode'

        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                  decay_steps,
                                                                  decay_rate=0.95,
                                                                  staircase=True)
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        min_cnt_loss = 1000
        min_sze_loss = 10
        min_loss = 1000
        min_sze_model = False
        with self.summary_writer.as_default():
            for self.epoch in range(epochs):
                print('# epoch:' + str(self.epoch + 1) + '/' + str(epochs))
                cnt_losses = []
                sze_losses = []
                for step in range(self.config.STEPS_PER_EPOCH):
                    ims, cnt_gt, sze_gt = data_provider.next_batch(batch_size)
                    # print(ims.shape, cnt_gt.shape, sze_gt.shape)
                    with tf.GradientTape(persistent=False) as tape:
                        cnt_preds, sze_preds = self.model(ims)
                        # print(cnt_preds.shape, sze_preds.shape)
                        cnt_loss = self.__compute_cnt_loss(cnt_gt, cnt_preds)
                        sze_loss = self.__compute_sze_loss(sze_gt, sze_preds)
                        cnt_losses.append(cnt_loss)
                        sze_losses.append(sze_loss)
                        grad = tape.gradient(cnt_loss + sze_loss, self.model.trainable_variables)
                        optimizer.apply_gradients(grads_and_vars=zip(grad, self.model.trainable_variables))
                        self.__draw_progress_bar(step + 1, self.config.STEPS_PER_EPOCH)

                test_cnt_losses = []
                test_sze_losses = []
                for i in range(10):
                    ims, cnt_gt, sze_gt = test_data_provider.next_batch(2)
                    # print(ims.shape)
                    cnt_preds, sze_preds = self.model.predict(ims)
                    cnt_loss = self.__compute_cnt_loss(cnt_gt, cnt_preds)
                    sze_loss = self.__compute_sze_loss(sze_gt, sze_preds)
                    test_cnt_losses.append(cnt_loss)
                    test_sze_losses.append(sze_loss)
                    # print(cnt_loss, sze_loss)

                mean_cnt_loss = tf.reduce_mean(cnt_losses)
                mean_sze_loss = tf.reduce_mean(sze_losses)

                test_mean_cnt_loss = tf.reduce_mean(test_cnt_losses)
                test_mean_sze_loss = tf.reduce_mean(test_sze_losses)

                print('\nCnt Loss:%f; Size Loss:%f; test cnt loss: %f; test size loss: %f; Lr: %f' % (
                mean_cnt_loss, mean_sze_loss, test_mean_cnt_loss, test_mean_sze_loss, KB.eval(optimizer._decayed_lr('float32'))))
                tf.summary.scalar('train_center_loss', mean_cnt_loss, step=(self.epoch + 1))
                tf.summary.scalar('train_size_loss', mean_sze_loss, step=(self.epoch + 1))
                tf.summary.scalar('test_center_loss', test_mean_cnt_loss, step=(self.epoch + 1))
                tf.summary.scalar('test_size_loss', test_mean_sze_loss, step=(self.epoch + 1))

                tf.keras.backend.clear_session()

                if min_loss > test_mean_cnt_loss + test_mean_sze_loss:
                    min_loss = test_mean_sze_loss + test_mean_cnt_loss
                    self.checkpoint_path = osp.join(self.log_dir,
                                                    self.NET_NAME.lower() + "_loss_epoch{0}.h5".format(self.epoch + 1))
                    print('Saving weights to %s' % (self.checkpoint_path))
                    self.model.save_weights(self.checkpoint_path)
                # if not min_sze_model:
                if test_mean_cnt_loss < min_cnt_loss:
                    min_cnt_loss = test_mean_cnt_loss
                    self.checkpoint_path = osp.join(self.log_dir,
                                                    self.NET_NAME.lower() + "cntloss_epoch{0}.h5".format(self.epoch + 1))
                    print('Saving weights to %s' % (self.checkpoint_path))
                    self.model.save_weights(self.checkpoint_path)
                    # self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)
                # else:
                if test_mean_sze_loss < min_sze_loss:
                    min_sze_loss = test_mean_sze_loss
                    self.checkpoint_path = osp.join(self.log_dir,
                                                    self.NET_NAME.lower() + "szeloss_epoch{0}.h5".format(self.epoch + 1))
                    print('Saving weights to %s' % (self.checkpoint_path))
                    self.model.save_weights(self.checkpoint_path)
                    # self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)

    def predict(self, image):
        return self.model.predict(image)

    def predict_on_batch(self, images):
        return self.model.predict_on_batch(images)

    # private functions
    def __set_log_dir(self):
        self.epoch = 0
        self.log_dir = osp.join(self.model_dir, self.NET_NAME.lower())

    def __build(self, is_training):
        # define inputs:[batch_size, Depth, Height, Width, Channels], for keras, you don't need
        # to specify the batch_size
        dtype = tf.float32
        input_image = KL.Input(shape=self.input_shape + [1], dtype=dtype, name='input_image')

        # define backbone network
        # filters = [32, 64, 128, 256]
        filters = [16, 32, 64, 128]
        x1 = KL.Conv3D(filters[0] // 2, (3, 3, 3), (1, 1, 1), padding='same')(input_image)
        x1 = KL.BatchNormalization(axis=-1)(x1, training=is_training)
        x1 = KL.ReLU()(x1)
        x1 = KL.Conv3D(filters[0], (3, 3, 3), (1, 1, 1), padding='same')(x1)
        x1 = KL.BatchNormalization(axis=-1)(x1, training=is_training)
        x1 = KL.ReLU()(x1)
        d1 = KL.MaxPooling3D(pool_size=(2, 2, 2))(x1)

        x2 = KL.Conv3D(filters[1] // 2, (3, 3, 3), (1, 1, 1), padding='same')(d1)
        x2 = KL.BatchNormalization(axis=-1)(x2, training=is_training)
        x2 = KL.ReLU()(x2)
        x2 = KL.Conv3D(filters[1], (3, 3, 3), (1, 1, 1), padding='same')(x2)
        x2 = KL.BatchNormalization(axis=-1)(x2, training=is_training)
        x2 = KL.ReLU()(x2)
        d2 = KL.MaxPooling3D(pool_size=(2, 2, 2))(x2)

        x3 = KL.Conv3D(filters[2] // 2, (3, 3, 3), (1, 1, 1), padding='same')(d2)
        x3 = KL.BatchNormalization(axis=-1)(x3, training=is_training)
        x3 = KL.ReLU()(x3)
        x3 = KL.Conv3D(filters[2], (3, 3, 3), (1, 1, 1), padding='same')(x3)
        x3 = KL.BatchNormalization(axis=-1)(x3, training=is_training)
        x3 = KL.ReLU()(x3)
        d3 = KL.MaxPooling3D(pool_size=(2, 2, 2))(x3)

        x4 = KL.Conv3D(filters[3] // 2, (3, 3, 3), (1, 1, 1), padding='same')(d3)
        x4 = KL.BatchNormalization(axis=-1)(x4, training=is_training)
        x4 = KL.ReLU()(x4)
        x4 = KL.Conv3D(filters[3], (3, 3, 3), (1, 1, 1), padding='same')(x4)
        x4 = KL.BatchNormalization(axis=-1)(x4, training=is_training)
        x4 = KL.ReLU()(x4)
        d4 = KL.MaxPooling3D(pool_size=(2, 2, 2))(x4)

        u5 = KL.UpSampling3D()(d4)
        x5 = KL.Conv3D(filters[3] // 2, (3, 3, 3), (1, 1, 1), padding='same')(u5)
        x5 = KL.BatchNormalization(axis=-1)(x5, training=is_training)
        x5 = KL.ReLU()(x5)
        x5 = KL.Conv3D(filters[3] // 2, (3, 3, 3), (1, 1, 1), padding='same')(x5)
        x5 = KL.BatchNormalization(axis=-1)(x5, training=is_training)
        x5 = KL.ReLU()(x5)
        m5 = KL.Concatenate()([x5, x4])
        x5 = KL.Conv3D(filters[3], (3, 3, 3), (1, 1, 1), padding='same')(m5)
        x5 = KL.BatchNormalization(axis=-1)(x5, training=is_training)
        x5 = KL.ReLU()(x5)

        u6 = KL.UpSampling3D()(x5)
        x6 = KL.Conv3D(filters[2] // 2, (3, 3, 3), (1, 1, 1), padding='same')(u6)
        x6 = KL.BatchNormalization(axis=-1)(x6, training=is_training)
        x6 = KL.ReLU()(x6)
        x6 = KL.Conv3D(filters[2] // 2, (3, 3, 3), (1, 1, 1), padding='same')(x6)
        x6 = KL.BatchNormalization(axis=-1)(x6, training=is_training)
        x6 = KL.ReLU()(x6)
        m6 = KL.Concatenate()([x6, x3])
        x6 = KL.Conv3D(filters[2], (3, 3, 3), (1, 1, 1), padding='same')(m6)
        x6 = KL.BatchNormalization(axis=-1)(x6, training=is_training)
        x6 = KL.ReLU()(x6)

        u7 = KL.UpSampling3D()(x6)
        x7 = KL.Conv3D(filters[1] // 2, (3, 3, 3), (1, 1, 1), padding='same')(u7)
        x7 = KL.BatchNormalization(axis=-1)(x7, training=is_training)
        x7 = KL.ReLU()(x7)
        x7 = KL.Conv3D(filters[1] // 2, (3, 3, 3), (1, 1, 1), padding='same')(x7)
        x7 = KL.BatchNormalization(axis=-1)(x7, training=is_training)
        x7 = KL.ReLU()(x7)
        m7 = KL.Concatenate()([x7, x2])
        x7 = KL.Conv3D(filters[1], (3, 3, 3), (1, 1, 1), padding='same')(m7)
        x7 = KL.BatchNormalization(axis=-1)(x7, training=is_training)
        x7 = KL.ReLU()(x7)

        u8 = KL.UpSampling3D()(x7)
        x8 = KL.Conv3D(filters[0] // 2, (3, 3, 3), (1, 1, 1), padding='same')(u8)
        x8 = KL.BatchNormalization(axis=-1)(x8, training=is_training)
        x8 = KL.ReLU()(x8)
        x8 = KL.Conv3D(filters[0] // 2, (3, 3, 3), (1, 1, 1), padding='same')(x8)
        x8 = KL.BatchNormalization(axis=-1)(x8, training=is_training)
        x8 = KL.ReLU()(x8)
        m8 = KL.Concatenate()([x8, x1])
        x8 = KL.Conv3D(filters[0], (3, 3, 3), (1, 1, 1), padding='same')(m8)
        x8 = KL.BatchNormalization(axis=-1)(x8, training=is_training)
        x8 = KL.ReLU()(x8)

        # define output logits
        cnt_preds = KL.Conv3D(self.num_classes, [3, 3, 3], padding='same', activation='sigmoid', use_bias=False,
                              name='cnt_preds')(x8)

        sze_preds = KL.Conv3D(3, [3, 3, 3], padding='same', activation=None, use_bias=False, name='sze_preds')(x8)

        outputs = [cnt_preds, sze_preds]
        model = KM.Model(input_image, outputs, name=self.NET_NAME.lower())
        return model

    def __compute_cnt_loss(self, cnt_gt, cnt_preds):
        '''
        the loss for center keypoint loss
        :param cnt_gt:
        :param cnt_preds:
        :return:
        '''
        cnt_preds = ops.convert_to_tensor(cnt_preds)
        # cnt_preds = tf.transpose(cnt_preds, [0, 2, 1, 3])
        cnt_gt = math_ops.cast(cnt_gt, cnt_preds.dtype)

        _, d, h, w, c = cnt_preds.get_shape().as_list()

        num_pos = tf.reduce_sum(tf.cast(cnt_gt == 1, tf.float32))
        # print ('num_pos:', num_pos)
        neg_weights = math.pow(1 - cnt_gt, 4)
        pos_weights = tf.ones_like(cnt_preds, dtype=tf.float32)
        weights = tf.where(cnt_gt == 1, pos_weights, neg_weights)
        inverse_preds = tf.where(cnt_gt == 1, cnt_preds, 1 - cnt_preds)

        loss = math.log(inverse_preds + 0.0001) * math.pow(1 - inverse_preds, 2) * weights
        loss = tf.reduce_mean(loss)
        loss = -loss / (num_pos * 1.0 + 1) * 100000
        return loss

    def __compute_sze_loss(self, sze_gt, sze_preds):
        '''
        compute the size loss
        :param sze_preds:
        :param sze_gt:
        :param mask: array masks with keypoints of value 1 and others 0
        :return:
        '''
        # sze_preds = tf.transpose(sze_preds, [0, 2, 1, 3])
        sze_gt = math_ops.cast(sze_gt, sze_preds.dtype)
        mask = tf.where(sze_gt != (0, 0, 0), tf.ones_like(sze_gt), tf.zeros_like(sze_gt))
        fg_num = math_ops.cast(tf.math.count_nonzero(mask), sze_preds.dtype) / 3.0
        # print ('fg_num:',fg_num)
        regr_loss = tf.reduce_sum(tf.abs(sze_gt - sze_preds) * mask) / fg_num
        # regr_loss = tf.sqrt(tf.reduce_sum(tf.square(sze_gt - sze_preds) * mask)) / fg_num

        return regr_loss

    def __delete_old_weights(self, nun_max_keep):
        '''
        keep num_max_keep weight files, the olds are deleted
        :param nun_max_keep:
        :return:
        '''
        weights_files = glob.glob(osp.join(self.log_dir, self.NET_NAME.lower() + '*.h5'))
        if len(weights_files) <= nun_max_keep:
            return

        weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(x))

        weights_files = weights_files[0:len(weights_files) - nun_max_keep]

        for weight_file in weights_files:
            if weight_file != self.checkpoint_path:
                os.remove(weight_file)

    def __draw_progress_bar(self, cur, total, bar_len=50):
        cur_len = int(cur / total * bar_len)
        sys.stdout.write('\r')
        sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
        sys.stdout.flush()
