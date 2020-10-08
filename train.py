import os
import tensorflow as tf

from config import cfg
from data_generator import DataGenerator
from centernet3D import Mediastinal_3dcenternet
from classifier import Classifier, ClassifierTorch


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train_detect():
    # datagenerator = DataGenerator(cfg, training=True)
    datagenerator = DataGenerator(cfg, training=True, mode='detect', data_root=cfg.DATA_ROOT,
                                  annotation_file=cfg.train_anno_file, results_file=None, label_file=None)
    # test_datagenerator = DataGenerator(cfg, training=False)
    test_datagenerator = DataGenerator(cfg, training=False, mode='detect', data_root=cfg.TEST_DATA_ROOT,
                                  annotation_file=cfg.test_anno_file, results_file=None, label_file=None)

    load_pretrained = True
    if not os.path.exists(cfg.CHECKPOINTS_ROOT):
        os.mkdir(cfg.CHECKPOINTS_ROOT)

    model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, 'centernet_96_128_norm_size_1_all_3')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model = Mediastinal_3dcenternet(cfg.INPUT_SHAPE, is_training=True, num_classes=1, model_dir=model_dir, config=cfg)
    print(tf.test.is_gpu_available())
    # model.model(tf.ones((1, 192, 192, 192, 1)))
    if load_pretrained:
        checkpoint_file = model.find_last()
        if not os.path.exists(checkpoint_file):
            print ('no pretrained weight file found...')
        else:
            print ('loading pretrained from ', checkpoint_file)
            model.load_weights(checkpoint_file, by_name=False)
            # if continue_train:
            #     current_epoch = int(checkpoint_file.split('.')[-2].split('epoch')[-1])

    model.train(datagenerator, test_datagenerator, learning_rate=0.001, decay_steps=1000, epochs=600, batch_size=2)
    # print(img.shape, cnt.shape, sze.shape)
    # from sklearn.model_selection import train_test_split


def train_cls():
    datagenerator = DataGenerator(cfg, training=True, mode='cls')
    test_datagenerator = DataGenerator(cfg, training=False, mode='cls')

    load_pretrained = True
    if not os.path.exists(cfg.CHECKPOINTS_ROOT):
        os.mkdir(cfg.CHECKPOINTS_ROOT)

    model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, 'resnet18_cls3_sapcing1')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model = Classifier(cfg.INPUT_SHAPE, is_training=True, num_classes=3, model_dir=model_dir, config=cfg)
    print(tf.test.is_gpu_available())
    # model.model(tf.ones((1, 192, 192, 192, 1)))
    if load_pretrained:
        checkpoint_file = model.find_last()
        if not os.path.exists(checkpoint_file):
            print('no pretrained weight file found...')
        else:
            print('loading pretrained from ', checkpoint_file)
            model.load_weights(checkpoint_file, by_name=False)
            # if continue_train:
            #     current_epoch = int(checkpoint_file.split('.')[-2].split('epoch')[-1])

    model.train(datagenerator, test_datagenerator, learning_rate=0.001, decay_steps=10000, epochs=600, batch_size=8)
    #


def train_cls_torch():
    datagenerator = DataGenerator(cfg, training=True, mode='cls', data_root=cfg.DATA_ROOT,
                                  annotation_file=cfg.train_anno_file, results_file=cfg.train_results_file, label_file=None)
    test_datagenerator = DataGenerator(cfg, training=False, mode='cls', data_root=cfg.TEST_DATA_ROOT,
                                  annotation_file=cfg.test_anno_file, results_file=cfg.test_results_file, label_file=None)

    load_pretrained = True
    if not os.path.exists(cfg.CHECKPOINTS_ROOT):
        os.mkdir(cfg.CHECKPOINTS_ROOT)

    model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, 'resnet18_cls2_torch_spacing1_focal_3')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model = ClassifierTorch(cfg.INPUT_SHAPE, is_training=True, num_classes=2, model_dir=model_dir, config=cfg)
    # print(tf.test.is_gpu_available())
    # model.model(tf.ones((1, 192, 192, 192, 1)))
    if load_pretrained:
        checkpoint_file = model.find_last()
        if not os.path.exists(checkpoint_file):
            print('no pretrained weight file found...')
        else:
            print('loading pretrained from ', checkpoint_file)
            model.load_weights(checkpoint_file, by_name=False)
            # if continue_train:
            #     current_epoch = int(checkpoint_file.split('.')[-2].split('epoch')[-1])

    model.train(datagenerator, test_datagenerator, learning_rate=0.0002, decay_steps=10000, epochs=600, batch_size=32)
    #

if __name__ == '__main__':
    train_cls_torch()
    # train_cls()
    # train_detect()