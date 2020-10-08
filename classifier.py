import os
import math
import os.path as osp
import tensorflow as tf
# from tensorflow import math
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
import torch
from torch import nn
from tensorboardX import SummaryWriter
from pytorch_model.resnet_3D import generate_model, focal_loss

from model import ResNet3D50, ResNet3D18


class Classifier():
    def __init__(self, input_shape, is_training, num_classes, model_dir, config):
        self._is_training = is_training
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.NET_NAME = 'ResNet50'
        self.__set_log_dir()  # logging and saving checkpoints
        self.model = ResNet3D50(num_class=num_classes)

    # public functions
    def summary(self):
        '''
        print the network attributes
        :return:
        '''
        self.model(tf.random.uniform((1, 32, 32, 32, 1)))
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
        self.model(tf.ones((1, 32, 32, 32, 1)))  # 初始化模型，不加的话加载模型会报错
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
        num_scan = math.ceil(batch_size / 8)
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                  decay_steps,
                                                                  decay_rate=0.95,
                                                                  staircase=True)
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        max_acc = 0
        with self.summary_writer.as_default():
            for self.epoch in range(epochs):
                print('# epoch:' + str(self.epoch + 1) + '/' + str(epochs))
                train_loss = []
                for step in range(self.config.STEPS_PER_EPOCH):
                    ims, labels = data_provider.get_bbox_cls_region(num_scan, batch_size)
                    # print(ims.shape, cnt_gt.shape, sze_gt.shape)
                    with tf.GradientTape(persistent=False) as tape:
                        pred = self.model(ims, training=True)
                        # print(cnt_preds.shape, sze_preds.shape)
                        loss = KLOSS.SparseCategoricalCrossentropy()(labels, pred)
                        grad = tape.gradient(loss, self.model.trainable_variables)
                        optimizer.apply_gradients(grads_and_vars=zip(grad, self.model.trainable_variables))
                        train_loss.append(loss)
                        self.__draw_progress_bar(step + 1, self.config.STEPS_PER_EPOCH)

                test_loss = []
                correct = 0
                cnt = 0
                for i in range(int(20 / num_scan)):
                    ims, labels = test_data_provider.get_bbox_cls_region(num_scan, batch_size)
                    # print(ims.shape)
                    pred = self.model(ims, training=False)
                    loss = KLOSS.SparseCategoricalCrossentropy()(labels, pred)
                    test_loss.append(loss)
                    pred = tf.argmax(pred, axis=1).numpy()
                    correct += np.sum(pred == labels)
                    cnt += labels.shape[0]
                    # print(cnt_loss, sze_loss)

                test_acc = correct / cnt
                train_mean_loss = tf.reduce_mean(train_loss)
                test_mean_loss = tf.reduce_mean(test_loss)

                print('\nTrain Loss:%f; test loss: %f; test acc: %f; Lr: %f' % (
                train_mean_loss, test_mean_loss, test_acc, KB.eval(optimizer._decayed_lr('float32'))))
                tf.summary.scalar('train_loss', train_mean_loss, step=(self.epoch + 1))
                tf.summary.scalar('test_loss', test_mean_loss, step=(self.epoch + 1))
                tf.summary.scalar('test_acc', test_acc, step=(self.epoch + 1))

                tf.keras.backend.clear_session()

                if test_acc > max_acc:
                    max_acc = test_acc
                    self.checkpoint_path = osp.join(self.log_dir,
                                                    self.NET_NAME.lower() + "_epoch{0}.h5".format(self.epoch + 1))
                    print('Saving weights to %s' % (self.checkpoint_path))
                    self.model.save_weights(self.checkpoint_path)
                    self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)

    def predict(self, image):
        return self.model.predict(image)

    def predict_on_batch(self, images):
        return self.model.predict_on_batch(images)

    # private functions
    def __set_log_dir(self):
        self.epoch = 0
        self.log_dir = osp.join(self.model_dir, self.NET_NAME.lower())

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


class ClassifierTorch():
    def __init__(self, input_shape, is_training, num_classes, model_dir, config):
        self._is_training = is_training
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.NET_NAME = 'ResNet18'
        self.__set_log_dir()  # logging and saving checkpoints
        self.model = generate_model(18, n_classes=num_classes, no_max_pool=True).cuda()

    # public functions
    def summary(self):
        '''
        print the network attributes
        :return:
        '''
        # self.model(tf.random.uniform((1, 32, 32, 32, 1)))
        return self.model

    def find_last(self, model_path=None):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        if model_path is None:
            weights_files = glob.glob(osp.join(self.log_dir, self.NET_NAME.lower() + '*.pth'))
        else:
            weights_files = glob.glob(osp.join(model_path, self.NET_NAME.lower(), self.NET_NAME.lower() + '*.pth'))
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
        # self.model(tf.ones((1, 32, 32, 32, 1)))  # 初始化模型，不加的话加载模型会报错
        self.model.load_state_dict(torch.load(filepath))

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
        num_scan = 2
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # criterion = nn.CrossEntropyLoss()
        criterion = focal_loss(gamma=3)
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
        max_acc = 0.8
        max_recall = 0.5
        self.model.train()
        for self.epoch in range(epochs):
            print('# epoch:' + str(self.epoch + 1) + '/' + str(epochs))
            train_loss = []
            for step in range(self.config.STEPS_PER_EPOCH):
                # ims, labels = data_provider.get_bbox_cls_region(num_scan, batch_size, channel_first=True)
                ims_all, labels_all = data_provider.get_bbox_cls_region(num_scan, max_num_box=128, channel_first=True)
                if self.num_classes == 2:
                    labels_all[labels_all > 1] = 1
                # print(ims.shape, cnt_gt.shape, sze_gt.shape)
                for i in range(0, ims_all.shape[0], batch_size):
                    ims = torch.tensor(ims_all[i:i+batch_size], dtype=torch.float32).cuda()
                    labels = torch.tensor(labels_all[i:i+batch_size], dtype=torch.long).cuda()
                    pred = self.model(ims)
                    # print(pred.shape, labels.shape)
                    # print(cnt_preds.shape, sze_preds.shape)
                    loss = criterion(pred, labels)
                    loss.backward()
                    train_loss.append(loss.cpu().detach().item())
                optimizer.step()
                optimizer.zero_grad()
                self.__draw_progress_bar(step + 1, self.config.STEPS_PER_EPOCH)

            test_loss = []
            correct = 0
            cnt = 0
            TPs = 0.0
            num_postive = 0.0
            self.model.eval()
            with torch.no_grad():
                for i in range(len(test_data_provider.train_list)):
                    ims_all, labels_all = test_data_provider.get_bbox_cls_region(1, max_num_box=None, channel_first=True)
                    # print(ims.shape)
                    for i in range(0, ims_all.shape[0], batch_size):
                        ims = torch.tensor(ims_all[i:i+batch_size], dtype=torch.float32).cuda()
                        labels = torch.tensor(labels_all[i:i+batch_size], dtype=torch.long).cuda()
                        pred = self.model(ims)
                        # print(cnt_preds.shape, sze_preds.shape)
                        loss = criterion(pred, labels)
                        pred = torch.argmax(pred, dim=1).detach()
                        correct += torch.eq(pred, labels).sum().item()
                        cnt += labels.shape[0]
                        stat = labels > 0
                        TP = torch.eq(pred[stat], labels[stat]).sum().item()
                        N = stat.sum().item()
                        TPs += TP
                        num_postive += N
                        test_loss.append(loss.cpu().item())
                # print(cnt_loss, sze_loss)

            test_acc = correct / cnt
            recall = TPs / num_postive
            # print(TPs, num_postive)
            train_mean_loss = np.mean(train_loss)
            test_mean_loss = np.mean(test_loss)

            print('\nTrain Loss:%f; test loss: %f; recall: %f; test acc: %f' % (
            train_mean_loss, test_mean_loss, recall, test_acc))
            self.summary_writer.add_scalar('train_loss', train_mean_loss, self.epoch + 1)
            self.summary_writer.add_scalar('test_loss', test_mean_loss, self.epoch + 1)
            self.summary_writer.add_scalar('test_acc', test_acc, self.epoch + 1)
            self.summary_writer.add_scalar('recall', recall, self.epoch + 1)

            if recall > max_recall:
                max_recall = recall
                max_acc = test_acc
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{0}.pth".format(self.epoch + 1))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)
            elif recall == max_recall and test_acc > max_acc:
                max_acc = test_acc
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{0}.pth".format(self.epoch + 1))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)

    def predict(self, image):
        return self.model(image)

    def predict_on_batch(self, images):
        return self.model.predict_on_batch(images)

    # private functions
    def __set_log_dir(self):
        self.epoch = 0
        self.log_dir = osp.join(self.model_dir, self.NET_NAME.lower())

    def __delete_old_weights(self, nun_max_keep):
        '''
        keep num_max_keep weight files, the olds are deleted
        :param nun_max_keep:
        :return:
        '''
        weights_files = glob.glob(osp.join(self.log_dir, self.NET_NAME.lower() + '*.pth'))
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


