import os
import tensorflow as tf
import json
import logging
from config import cfg
from torch.utils.data import DataLoader, Dataset
from data_generator import DataGenerator, MyDataset
from centernet3D import Mediastinal_3dcenternet
from classifier import Classifier, ClassifierTorch

logger = logging.getLogger('train centernet')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)
handler.setLevel(0)
logger.addHandler(handler)

file_handler = logging.FileHandler('checkpoints/resnet.log', mode='a')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
    with open(cfg.cross_validation, 'r') as f:
        cv = json.load(f)
    datagenerator = DataGenerator(cfg, training=True, mode='cls', data_root=cfg.DATA_ROOT,
                                  annotation_file=cfg.anno_file, results_file=cfg.train_results_file,
                                  label_file=None, cross_validation=cv['fold0'])
    test_datagenerator = DataGenerator(cfg, training=False, mode='cls', data_root=cfg.DATA_ROOT,
                                  annotation_file=cfg.anno_file, results_file=cfg.train_results_file,
                                  label_file=None, cross_validation=cv['fold0'])

    load_pretrained = True
    if not os.path.exists(cfg.CHECKPOINTS_ROOT):
        os.mkdir(cfg.CHECKPOINTS_ROOT)

    model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, 'resnet18_cls2_2_sapcing1')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model = Classifier(cfg.INPUT_SHAPE, is_training=True, num_classes=2, model_dir=model_dir, config=cfg)
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


def train_cls_torch(cv, fold):

    # datagenerator = DataGenerator(cfg, training=True, mode='cls', data_root=cfg.DATA_ROOT,
    #                               annotation_file=cfg.anno_file, results_file=cfg.train_results_file,
    #                               label_file=cfg.label_file, cross_validation=cv[fold])
    # test_datagenerator = DataGenerator(cfg, training=False, mode='cls', data_root=cfg.DATA_ROOT,
    #                               annotation_file=cfg.anno_file, results_file=cfg.train_results_file,
    #                               label_file=cfg.label_file, cross_validation=cv[fold])
    load_pretrained = True
    if not os.path.exists(cfg.CHECKPOINTS_ROOT):
        os.mkdir(cfg.CHECKPOINTS_ROOT)

    model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, 'densenet_final_cls2_2')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model = ClassifierTorch(cfg.INPUT_SHAPE, model_name='densenet', is_training=True, num_classes=2, num_classes2=2,
                            model_dir=model_dir, config=cfg, fold=fold)
    if load_pretrained:
        checkpoint_file = model.find_last()
        if not os.path.exists(checkpoint_file):
            print('no pretrained weight file found...')
        else:
            print('loading pretrained from ', checkpoint_file)
            model.load_weights(checkpoint_file, by_name=False)
            # if continue_train:
            #     current_epoch = int(checkpoint_file.split('.')[-2].split('epoch')[-1])
    # start_epoch = 50
    # if fold == "fold3":
    #     start_epoch = 17
    # model.train2(datagenerator, test_datagenerator, learning_rate=0.0001, epochs=20, batch_size=8)
    dataset_train = MyDataset(is_train=True, cv=cv[fold], label_file=cfg.label_file)
    dataset_test = MyDataset(is_train=False, cv=cv[fold], label_file=cfg.label_file)

    dataloader_train = DataLoader(dataset_train, 8, True)
    dataloader_test = DataLoader(dataset_test, 8, False)
    model.train3(dataloader_train, dataloader_test, learning_rate=0.0001, epochs=20, start_epoch=0)
    # model.train4(dataloader_train, dataloader_test, learning_rate=0.0001, epochs=10)


if __name__ == '__main__':
    with open(cfg.cross_validation, 'r') as f:
        cv = json.load(f)
    # fold = 'fold0'
    for fold in ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']:
    # for fold in ['fold3', 'fold4']:
        # fold = 'fold3'
        print(fold)
        logger.info(fold)
        train_cls_torch(cv, fold)
    # train_cls()
    # train_detect()

