import os
import math
import logging
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
import time
from tqdm import tqdm
from torch import nn
from tensorboardX import SummaryWriter
from pytorch_model.resnet_3D import generate_model, focal_loss, cls_loss

from model import ResNet3D50, ResNet3D18
from pytorch_model.vgg import vgg16_bn, vgg19_bn
from pytorch_model.densenet import DenseNet3d
logger = logging.getLogger('train centernet')

class Classifier():
    def __init__(self, input_shape, is_training, num_classes, model_dir, config):
        self._is_training = is_training
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.NET_NAME = 'ResNet18'
        self.__set_log_dir()  # logging and saving checkpoints
        self.model = ResNet3D18(num_class=num_classes)

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
    def __init__(self, input_shape, is_training, num_classes, model_dir, config, model_name="vgg16",
                       fold=None, num_classes2=None):
        '''
        :param input_shape: (d, h, w)
        :param is_training:
        :param num_classes:
        :param model_dir:
        :param config:
        :param model_name: vgg16, vgg19, resnet18, seresnet18
        :param model_depth:
        :param senet:
        :param fold:
        :param num_classes2:
        '''
        self._is_training = is_training
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.NET_NAME = model_name
        if model_name == 'vgg16':
            self.model = vgg16_bn(dim=3, num_classes=num_classes, num_classes2=num_classes2).cuda()
        elif model_name == 'vgg19':
            self.model = vgg19_bn(dim=3, num_classes=num_classes, num_classes2=num_classes2).cuda()
        elif model_name == 'resnet18':
            self.model = generate_model(18, senet=False, n_classes=num_classes, no_max_pool=True,
                                        n_classes2=num_classes2).cuda()
        elif model_name == 'seresnet18':
            self.model = generate_model(18, senet=True, n_classes=num_classes, no_max_pool=True,
                                        n_classes2=num_classes2).cuda()
        elif model_name == 'densenet':
            self.model = DenseNet3d(100, 2, num_classes2=num_classes2).cuda()
        if fold is None:
            self.fold = 'fold0'
        else:
            self.fold = fold
        self.__set_log_dir()  # logging and saving checkpoints
        self.num_classes2 = num_classes2
        logger.info(self.NET_NAME)
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

    # 三分类
    def train(self, data_provider, test_data_provider, learning_rate, epochs, batch_size):
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
        sum_scan = len(data_provider.train_list)
        steps = sum_scan // num_scan
        # steps=2
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        # criterion = focal_loss(gamma=3)
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
        max_acc = 0.0
        max_recall = 0.0
        for self.epoch in range(epochs):
            print('# epoch:' + str(self.epoch + 1) + '/' + str(epochs))
            train_loss = []
            self.model.train()
            for step in range(steps):
                # ims, labels = data_provider.get_bbox_cls_region(num_scan, batch_size, channel_first=True)
                ims_all, labels_all = data_provider.get_bbox_cls_region(num_scan, max_num_box=128, channel_first=True, size=(64, 64, 64))
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
                # self.__draw_progress_bar(step + 1, self.config.STEPS_PER_EPOCH)
                if step % 10 == 0:
                    self.log("epoch: {}/{}, step: {}/{}, loss: {:.6f}".format(self.epoch+1, epochs, step, steps, train_loss[-1]))

            test_loss = []
            correct = 0
            cnt = 0
            TPs = 0.0
            num_postive = 0.0
            num_test = len(test_data_provider.train_list)
            # num_test = 2
            self.model.eval()
            with torch.no_grad():
                for i in tqdm(range(num_test)):
                    ims_all, labels_all = test_data_provider.get_bbox_cls_region(1, max_num_box=128, channel_first=True, size=(64, 64, 64))
                    # print(ims.shape)
                    for i in range(0, ims_all.shape[0], batch_size):
                        ims = torch.tensor(ims_all[i:i+batch_size], dtype=torch.float32).cuda()
                        labels = torch.tensor(labels_all[i:i+batch_size], dtype=torch.long).cuda()
                        pred = self.model(ims)
                        # print(cnt_preds.shape, sze_preds.shape)
                        loss = criterion(pred, labels)
                        pred = torch.argmax(pred, dim=1).detach()
                        # correct += torch.eq(pred, labels).sum().item()
                        # cnt += labels.shape[0]
                        stat = labels > 0
                        TP = torch.sum(pred[stat] > 0).item()
                        # TP = torch.eq(pred[stat], labels[stat]).sum().item()
                        correct += torch.eq(pred[stat], labels[stat]).sum().item()
                        N = stat.sum().item()
                        cnt += N
                        TPs += TP
                        num_postive += N
                        test_loss.append(loss.cpu().item())
                # print(cnt_loss, sze_loss)

            test_acc = correct / cnt
            recall = TPs / num_postive
            # print(TPs, num_postive)
            train_mean_loss = np.mean(train_loss)
            test_mean_loss = np.mean(test_loss)

            self.log('Train Loss:%f; test loss: %f; recall: %f; test acc: %f' % (
            train_mean_loss, test_mean_loss, recall, test_acc))
            self.summary_writer.add_scalar('train_loss', train_mean_loss, self.epoch + 1)
            self.summary_writer.add_scalar('test_loss', test_mean_loss, self.epoch + 1)
            self.summary_writer.add_scalar('test_acc', test_acc, self.epoch + 1)
            self.summary_writer.add_scalar('recall', recall, self.epoch + 1)

            if recall > max_recall:
                max_recall = recall
                max_acc = test_acc
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}.pth".format(self.epoch + 1, recall, test_acc))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)
            elif recall == max_recall and test_acc > max_acc:
                max_acc = test_acc
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}.pth".format(self.epoch + 1, recall, test_acc))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)

            if self.epoch+1 >= 50 and self.epoch+1 % 10 == 0:
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}.pth".format(self.epoch + 1, recall, test_acc))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
            if recall > 0.9:
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}.pth".format(
                                                    self.epoch + 1, recall, test_acc))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)

    def train4(self, dataloader_train, dataloader_test, learning_rate, epochs, start_epoch=0):
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
        steps = len(dataloader_train)
        # steps=2
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        # criterion = focal_loss(gamma=3)
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
        max_acc = 0.0
        max_recall = 0.0
        max_cls_acc = 0.0
        for self.epoch in range(start_epoch, epochs):
            print('# epoch:' + str(self.epoch + 1) + '/' + str(epochs))
            train_loss1 = []
            # train_loss2 = []

            self.model.train()
            for step, (ims, labels) in enumerate(dataloader_train):
                ims = ims.cuda()
                labels = labels.cuda()
                pred = self.model(ims)
                # print(pred.shape, labels.shape)
                # print(cnt_preds.shape, sze_preds.shape)
                loss1 = criterion(pred, labels)
                loss1.backward()
                train_loss1.append(loss1.cpu().detach().item())
                # train_loss2.append(loss2.cpu().detach().item())

                optimizer.step()
                optimizer.zero_grad()
                # self.__draw_progress_bar(step + 1, self.config.STEPS_PER_EPOCH)
                if step % 10 == 0:
                    logger.info(
                        "epoch: {}/{}, step: {}/{}, loss1: {:.6f}".format(self.epoch + 1, epochs, step,
                                                                                         steps,
                                                                                         train_loss1[-1]))

            test_loss1 = []
            # test_loss2 = []
            correct = 0       # 良恶性分类正确个数
            cnt = 0             # 良恶性个数
            TPs = 0.0           # 良恶性个数
            num_postive = 0.0
            correct_malignant = 0
            true_malignant = 0
            # num_test = 2
            self.model.eval()
            with torch.no_grad():
                for ims, labels in tqdm(dataloader_test):
                    ims = ims.cuda()
                    labels = labels.cuda()
                    pred1 = self.model(ims)
                    # print(cnt_preds.shape, sze_preds.shape)
                    loss1 = criterion(pred1, labels)

                    pred = torch.argmax(pred1, dim=1).detach()

                    stat = labels > 0
                    stat_pred = pred > 0
                    TPs += torch.sum(stat_pred[stat]).item()
                    cnt += stat.sum().item()

                    correct += torch.sum(labels[stat] == pred[stat]).item()
                    num_postive += stat.sum().item()

                    stat1 = labels > 1
                    stat1_pred = pred > 1
                    correct_malignant += stat1_pred[stat1].sum().item()
                    true_malignant += stat1.sum().item()

                    test_loss1.append(loss1.cpu().item())

                # print(cnt_loss, sze_loss)

            test_acc = correct / cnt
            recall = TPs / num_postive
            recall_malignant = correct_malignant / true_malignant
            # print(TPs, num_postive)
            train_mean_loss1 = np.mean(train_loss1)
            # train_mean_loss2 = np.mean(train_loss2)

            test_mean_loss1 = np.mean(test_loss1)
            # test_mean_loss2 = np.mean(test_loss2)

            logger.info(
                'Train Loss:%f; test loss: %f; recall: %f; test acc: %f; test recall: %f' % (
                    train_mean_loss1, test_mean_loss1, recall, test_acc,
                    recall_malignant))
            self.summary_writer.add_scalar('train_loss', train_mean_loss1, self.epoch + 1)
            self.summary_writer.add_scalar('test_loss', test_mean_loss1, self.epoch + 1)
            self.summary_writer.add_scalar('test_acc', test_acc, self.epoch + 1)
            self.summary_writer.add_scalar('recall', recall, self.epoch + 1)

            if recall > max_recall:
                max_recall = recall
                max_acc = test_acc
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}.pth".format(self.epoch + 1, recall, test_acc))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)
            elif recall == max_recall and test_acc > max_acc:
                max_acc = test_acc
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}.pth".format(self.epoch + 1, recall, test_acc))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)

            if self.epoch+1 >= 50 and self.epoch+1 % 10 == 0:
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}.pth".format(self.epoch + 1, recall, test_acc))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
            if recall > 0.9:
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}.pth".format(
                                                    self.epoch + 1, recall, test_acc))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)

    def train2(self, data_provider, test_data_provider, learning_rate, epochs, batch_size, start_epoch=0):
        assert self._is_training == True, 'not in training mode'
        assert self.num_classes2 is not None, 'not in two cls mode'

        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)
        num_scan = 2
        sum_scan = len(data_provider.train_list)
        steps = sum_scan // num_scan
        # steps=2
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # criterion = nn.CrossEntropyLoss()
        loss_FP = cls_loss(extra_id=2)
        loss_cls = cls_loss(extra_id=0)
        # criterion = focal_loss(gamma=3)
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
        max_acc = 0.0
        max_recall = 0.0
        max_cls_acc = 0.0
        for self.epoch in range(start_epoch, epochs):
            print('# epoch:' + str(self.epoch + 1) + '/' + str(epochs))
            train_loss1 = []
            train_loss2 = []

            self.model.train()
            for step in range(steps):
                # ims, labels = data_provider.get_bbox_cls_region(num_scan, batch_size, channel_first=True)
                ims_all, labels_all = data_provider.get_bbox_cls_region(num_scan, max_num_box=128, channel_first=True,
                                                                        size=(64, 64, 64))
                # print(ims.shape, cnt_gt.shape, sze_gt.shape)
                for i in range(0, ims_all.shape[0], batch_size):
                    ims = torch.tensor(ims_all[i:i + batch_size], dtype=torch.float32).cuda()
                    labels = torch.tensor(labels_all[i:i + batch_size], dtype=torch.long).cuda()
                    pred1, pred2 = self.model(ims)
                    # print(pred.shape, labels.shape)
                    # print(cnt_preds.shape, sze_preds.shape)
                    loss1 = loss_FP(pred1, labels)
                    loss2 = loss_cls(pred2, labels)
                    loss = loss1 + loss2
                    loss.backward()
                    train_loss1.append(loss1.cpu().detach().item())
                    train_loss2.append(loss2.cpu().detach().item())
                optimizer.step()
                optimizer.zero_grad()
                # self.__draw_progress_bar(step + 1, self.config.STEPS_PER_EPOCH)
                if step % 10 == 0:
                    logger.info("epoch: {}/{}, step: {}/{}, loss1: {:.6f}, loss2: {:.6f}".format(self.epoch + 1, epochs, step, steps,
                                                                              train_loss1[-1], train_loss2[-1]))

            test_loss1 = []
            test_loss2 = []
            correct = 0
            cnt = 0
            TPs = 0.0
            num_postive = 0.0
            correct_malignant = 0
            true_malignant = 0
            num_test = len(test_data_provider.train_list)
            # num_test = 2
            self.model.eval()
            with torch.no_grad():
                for ii in tqdm(range(num_test)):
                    ims_all, labels_all = test_data_provider.get_bbox_cls_region(1, max_num_box=16, channel_first=True,
                                                                                 size=(64, 64, 64))
                    # print(ims.shape)
                    for i in range(0, ims_all.shape[0], batch_size):
                        ims = torch.tensor(ims_all[i:i + batch_size], dtype=torch.float32).cuda()
                        labels = torch.tensor(labels_all[i:i + batch_size], dtype=torch.long).cuda()
                        pred1, pred2 = self.model(ims)
                        # print(cnt_preds.shape, sze_preds.shape)
                        loss1 = loss_FP(pred1, labels)
                        loss2 = loss_cls(pred2, labels)

                        tp, n = self.get_TP(pred1, labels)
                        co, n2 = self.get_acc(pred2, labels)
                        acc_tp, acc_n = self.get_acc_tp(pred2, labels)
                        correct_malignant += acc_tp
                        true_malignant += acc_n
                        correct += co
                        cnt += n2
                        TPs += tp
                        num_postive += n
                        test_loss1.append(loss1.cpu().item())
                        test_loss2.append(loss2.cpu().item())

                # print(cnt_loss, sze_loss)

            test_acc = correct / cnt
            recall = TPs / num_postive
            recall_malignant = correct_malignant / true_malignant
            # print(TPs, num_postive)
            train_mean_loss1 = np.mean(train_loss1)
            train_mean_loss2 = np.mean(train_loss2)

            test_mean_loss1 = np.mean(test_loss1)
            test_mean_loss2 = np.mean(test_loss2)

            logger.info('Train FP Loss:%f; train cls loss: %f; test FP loss: %f; test cls loss: %f; recall: %f; test acc: %f; test recall: %f' % (
                train_mean_loss1,train_mean_loss2, test_mean_loss1,test_mean_loss2, recall, test_acc, recall_malignant))
            self.summary_writer.add_scalar('train_loss', train_mean_loss1+train_mean_loss2, self.epoch + 1)
            self.summary_writer.add_scalar('test_loss', test_mean_loss1+test_mean_loss2, self.epoch + 1)
            self.summary_writer.add_scalar('test_acc', test_acc, self.epoch + 1)
            self.summary_writer.add_scalar('recall', recall, self.epoch + 1)

            if recall > max_recall:
                max_recall = recall
                max_acc = test_acc
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}.pth".format(
                                                    self.epoch + 1, recall, test_acc))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)
            elif recall == max_recall and test_acc > max_acc:
                max_acc = test_acc
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}.pth".format(
                                                    self.epoch + 1, recall, test_acc))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)

            if recall > 0.9:
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}.pth".format(
                                                    self.epoch + 1, recall, test_acc))
                logger.info('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)

    def train3(self, dataloader_train, dataloader_test, learning_rate=0.0001, epochs=20, start_epoch=0):
        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)
        steps = len(dataloader_train)
        # steps=2
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
        # criterion = nn.CrossEntropyLoss()
        loss_FP = cls_loss(extra_id=2)
        loss_cls = cls_loss(extra_id=0)
        # criterion = focal_loss(gamma=3)
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
        max_acc = 0.0
        max_recall = 0.0
        max_cls_acc = 0.0
        epochs += start_epoch
        for self.epoch in range(start_epoch, epochs):
            print('# epoch:' + str(self.epoch + 1) + '/' + str(epochs))
            train_loss1 = []
            train_loss2 = []

            self.model.train()
            for step, (ims, labels) in enumerate(dataloader_train):
                ims = ims.cuda()
                labels = labels.cuda()
                pred1, pred2 = self.model(ims)
                # print(pred.shape, labels.shape)
                # print(cnt_preds.shape, sze_preds.shape)
                loss1 = loss_FP(pred1, labels)
                loss2 = loss_cls(pred2, labels)
                loss = loss1 + loss2

                optimizer.zero_grad()
                loss.backward()
                train_loss1.append(loss1.cpu().detach().item())
                train_loss2.append(loss2.cpu().detach().item())

                optimizer.step()
                scheduler.step()
                # self.__draw_progress_bar(step + 1, self.config.STEPS_PER_EPOCH)
                if step % 10 == 0:
                    logger.info("epoch: {}/{}, step: {}/{}, loss1: {:.6f}, loss2: {:.6f}".format(self.epoch + 1, epochs, step, steps,
                                                                              train_loss1[-1], train_loss2[-1]))

            test_loss1 = []
            test_loss2 = []
            correct = 0
            cnt = 0
            TPs = 0.0
            num_postive = 0.0
            correct_malignant = 0
            true_malignant = 0
            # num_test = 2
            self.model.eval()
            with torch.no_grad():
                for ims, labels in tqdm(dataloader_test):
                    ims = ims.cuda()
                    labels = labels.cuda()
                    pred1, pred2 = self.model(ims)
                    # print(cnt_preds.shape, sze_preds.shape)
                    loss1 = loss_FP(pred1, labels)
                    loss2 = loss_cls(pred2, labels)

                    tp, n = self.get_TP(pred1, labels)
                    co, n2 = self.get_acc(pred2, labels)
                    acc_tp, acc_n = self.get_acc_tp(pred2, labels)
                    correct_malignant += acc_tp
                    true_malignant += acc_n
                    correct += co
                    cnt += n2
                    TPs += tp
                    num_postive += n
                    test_loss1.append(loss1.cpu().item())
                    test_loss2.append(loss2.cpu().item())

                # print(cnt_loss, sze_loss)

            test_acc = correct / cnt
            recall = TPs / num_postive
            recall_malignant = correct_malignant / true_malignant
            # print(TPs, num_postive)
            train_mean_loss1 = np.mean(train_loss1)
            train_mean_loss2 = np.mean(train_loss2)

            test_mean_loss1 = np.mean(test_loss1)
            test_mean_loss2 = np.mean(test_loss2)

            logger.info('Train FP Loss:%f; train cls loss: %f; test FP loss: %f; test cls loss: %f; recall: %f; test acc: %f; test recall: %f' % (
                train_mean_loss1,train_mean_loss2, test_mean_loss1,test_mean_loss2, recall, test_acc, recall_malignant))
            self.summary_writer.add_scalar('train_loss', train_mean_loss1+train_mean_loss2, self.epoch + 1)
            self.summary_writer.add_scalar('test_loss', test_mean_loss1+test_mean_loss2, self.epoch + 1)
            self.summary_writer.add_scalar('test_acc', test_acc, self.epoch + 1)
            self.summary_writer.add_scalar('recall', recall, self.epoch + 1)
            if recall > max_recall:
                max_recall = recall
                max_acc = test_acc
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}_{:.2f}.pth".format(
                                                    self.epoch + 1, recall, test_acc, recall_malignant))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)
            elif recall == max_recall and test_acc > max_acc:
                max_acc = test_acc
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}_{:.2f}.pth".format(
                                                    self.epoch + 1, recall, test_acc, recall_malignant))
                print('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self.__delete_old_weights(self.config.MAX_KEEPS_CHECKPOINTS)

            if recall > 0.9:
                self.checkpoint_path = osp.join(self.log_dir,
                                                self.NET_NAME.lower() + "_epoch{}_{:.2f}_{:.2f}_{:.2f}.pth".format(
                                                    self.epoch + 1, recall, test_acc, recall_malignant))
                logger.info('Saving weights to %s' % (self.checkpoint_path))
                torch.save(self.model.state_dict(), self.checkpoint_path)

    def get_TP(self, pred, labels):
        # 总tp
        # labels[labels > 0] = 1
        label_new = labels.clone()
        label_new[label_new > 1] = 1
        pred = torch.argmax(pred, dim=1)
        stat = label_new > 0
        TP = torch.eq(pred[stat], label_new[stat]).sum().item()
        N = stat.sum().item()
        return TP, N

    def get_acc(self, pred, labels):
        # 良恶性 acc
        stats = labels > 0
        N = stats.sum().item()
        if N == 0:
            return 0, 0
        pred = torch.argmax(pred, dim=1)
        correct = torch.eq(pred[stats], labels[stats]-1).sum().item()
        return correct, N

    def get_acc_tp(self, pred, labels):
        # 良恶性 tp
        stats = labels > 1
        N = stats.sum().item()
        if N == 0:
            return 0, 0
        pred = torch.argmax(pred, dim=1)
        TP = torch.eq(pred[stats], labels[stats]-1).sum().item()
        return TP, N

    def predict(self, image):
        return self.model(image)

    def predict_on_batch(self, images):
        return self.model.predict_on_batch(images)

    def log(self, info):
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())), info)

    # private functions
    def __set_log_dir(self):
        self.epoch = 0
        self.log_dir = osp.join(self.model_dir, self.fold)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

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


if __name__ == '__main__':
    model = generate_model(18, senet=True, n_classes=2, no_max_pool=True, n_classes2=2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
    # criterion = nn.CrossEntropyLoss()
    loss_FP = cls_loss(extra_id=2)
    loss_cls = cls_loss(extra_id=0)
    batch_size = 4
    for i in range(10):
        train_loss1 = []
        train_loss2 = []
        model.train()
        for j in range(10):
            ims = torch.rand((batch_size, 1, 96, 96, 96)).cuda()
            labels = torch.randint(3, (batch_size,)).cuda()
            pred1, pred2 = model(ims)
            loss1 = loss_FP(pred1, labels)
            loss2 = loss_cls(pred2, labels)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            train_loss1.append(loss1.cpu().detach().item())
            train_loss2.append(loss2.cpu().detach().item())
            optimizer.step()
            scheduler.step()
            print(i, j, train_loss1[-1], train_loss2[-1])
        model.eval()
        with torch.no_grad():
            for j in range(5):
                ims = torch.rand((batch_size, 1, 96, 96, 96)).cuda()
                labels = torch.randint(3, (batch_size,)).cuda()
                pred1, pred2 = model(ims)
                loss1 = loss_FP(pred1, labels)
                loss2 = loss_cls(pred2, labels)
                loss = loss1 + loss2
                print("test ", j, loss1.cpu().detach().item(), loss2.cpu().detach().item())
