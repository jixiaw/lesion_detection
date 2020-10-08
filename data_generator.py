# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:05:27 2019

@author: wjcongyu
"""

import numpy as np
import os
import os.path as osp
import pandas as pd
import cv2
from PIL import Image
import imageio
import json

from data_processor import generate_gaussian_mask_3d_tf, hu2gray, resize, generate_gaussian_mask_3d
from utils import get_bbox_region, IOU_3d, load_raw_data, draw_results, dist_3d, get_crop_region, IOU_3d_max, dist_3d_max
from config import cfg


class DataGenerator(object):
    def __init__(self, cfg, training=True, mode='detect', data_root=None, annotation_file=None, results_file=None, label_file=None):
        '''
        :param cfg: 各种参数
        :param training: 是否训练
        :param mode: 模式有检测 detect 和分类 cls
        :param data_root: 数据文件夹
        :param annotation_file: 用于检测的标签文件
        :param results_file: 检测的结果文件，用于分类
        :param label_file: 分类的标签文件
        '''
        if annotation_file is None:
            if training:
                self.annotation_file = cfg.train_anno_file
            else:
                self.annotation_file = cfg.test_anno_file
        else:
            self.annotation_file = annotation_file
        self.mode = mode
        self.data_root_dir = data_root
        self.cfg = cfg
        self.training = training
        print('loading annotations from ', self.annotation_file)
        self.train_list, self.annotations = self.load_annotations(self.annotation_file)
        if self.mode == 'cls':
            self.labels = self.load_labels(label_file)
            self.results_file = results_file
            self.detect_results = self.load_results(self.results_file)
        # np.random.shuffle(self.file_list)
        # self.train_list = self.file_list
        print('load annotation done')
        if training:
            print('found train images:', len(self.train_list))
        else:
            print('found test images:', len(self.train_list))
        # print ('found test images:', len(self.test_list))

        self.train_file_idx = 0
        # self.test_file_idx = 0

    def load_annotations(self, anno_file):
        with open(anno_file, 'r') as f:
            annos = json.load(f)
        annotations = {}
        train_list = []
        for filename in annos.keys():
            annoboxs = annos[filename]
            img_path = osp.join(self.data_root_dir, filename, filename + '_128.npy')
            if not osp.exists(img_path):
                # print(img_path)
                continue
            if filename not in annotations:
                annotations[filename] = []
                train_list.append(filename)
            if isinstance(annoboxs, list):
                for anno in annoboxs:
                    x, y, z, w, h, d = anno['x'], anno['y'], anno['z'], anno['w'], anno['h'], anno['d']
                    annotations[filename].append([x, y, z, w, h, d])
            else:
                anno = annoboxs
                x, y, z, w, h, d = anno['x'], anno['y'], anno['z'], anno['w'], anno['h'], anno['d']
                annotations[filename].append([x, y, z, w, h, d])
        np.random.shuffle(train_list)
        return train_list, annotations

    def load_results(self, results_file):
        with open(results_file, 'r') as f:
            res = json.load(f)
        return res

    def load_labels(self, label_file_path):
        res = {}
        if label_file_path is None:
            for name in self.train_list:
                res[name] = 0
        else:
            label_file = pd.read_excel(label_file_path)
            for i in range(len(label_file)):
                ID = '%07d' % label_file['ID'].iloc[i]
                res[ID] = label_file['label_1'].iloc[i]
        return res

    def next_batch(self, batch_size, channel_first=False, return_names=False):
        ims = []
        cnt_targets = []
        sze_targets = []
        filenames = []
        for i in range(batch_size):
            file_name = self.train_list[int((self.train_file_idx + i) % len(self.train_list))]
            filenames.append(file_name)
            img_path = osp.join(self.data_root_dir, file_name, file_name + '_128.npy')
            im = self.load_image(img_path)
            annotation = self.annotations[file_name]
            # print(file_name)
            if self.training:
                im, cnt_target, sze_target = self.create_feed_ims_and_targets(im, annotation, random_crop=True,
                                                                              norm_size=True)
            else:
                im, cnt_target, sze_target = self.create_feed_ims_and_targets(im, annotation, random_crop=False,
                                                                              norm_size=True)

            ims.append(im)
            cnt_targets.append(cnt_target)
            sze_targets.append(sze_target)
           
        ims = np.array(ims)
        cnt_targets = np.array(cnt_targets)
        sze_targets = np.array(sze_targets)

        # cycle the batches
        self.train_file_idx += batch_size
        if self.train_file_idx >= len(self.train_list):
            self.train_file_idx = 0
            if self.training:
                np.random.shuffle(self.train_list)
        cnt_targets = np.expand_dims(cnt_targets, axis=-1)
        if channel_first:
            ims = np.transpose(ims, (0, 4, 1, 2, 3))
            cnt_targets = np.transpose(cnt_targets, (0, 4, 1, 2, 3))
            sze_targets = np.transpose(sze_targets, (0, 4, 1, 2, 3))
        #print ('ims shape:', ims.shape)
        if return_names:
            return ims, cnt_targets, sze_targets, filenames
        else:
            return ims, cnt_targets, sze_targets

    def next_test_batch(self, batch_size, channel_first=False):
        ims = []
        cnt_targets = []
        sze_targets = []
        for i in range(batch_size):
            file_name = self.test_list[int((self.test_file_idx + i) % len(self.test_list))]

            img_path = osp.join(self.data_root_dir, file_name, file_name + '_128.npy')
            im = self.load_image(img_path)
            annotation = self.annotations[file_name]
            # print (file_name)
            im, cnt_target, sze_target = self.create_feed_ims_and_targets(im, annotation, self.cfg, resized=False)

            ims.append(im)
            cnt_targets.append(cnt_target)
            sze_targets.append(sze_target)

        ims = np.array(ims)
        cnt_targets = np.array(cnt_targets)
        sze_targets = np.array(sze_targets)

        # cycle the batches
        self.test_file_idx += batch_size
        if self.test_file_idx >= len(self.test_list):
            self.test_file_idx = 0
            # np.random.shuffle(self.file_list)

        cnt_targets = np.expand_dims(cnt_targets, axis=-1)
        if channel_first:
            ims = np.transpose(ims, (0, 4, 1, 2, 3))
            cnt_targets = np.transpose(cnt_targets, (0, 4, 1, 2, 3))
            sze_targets = np.transpose(sze_targets, (0, 4, 1, 2, 3))
        # print ('ims shape:', ims.shape)
        return ims, cnt_targets, sze_targets

    def create_feed_ims_and_targets(self, im, annotation, crop=True, random_crop=False, norm_size=False):
        # H, W, D = im.shape
        origin_shape = np.array(im.shape)
        bbox = np.array(annotation)  # shape (n, 6)
        im_gray = hu2gray(im, WL=40, WW=500)
        # Normalization
        im_gray = np.expand_dims(im_gray, -1).astype(np.float32) / 255.0
        im_gray = (im_gray - self.cfg.mean) / self.cfg.std
        # im_gray = (im_gray - 0.196) / 0.278
        target_shape = np.int32(origin_shape)

        # compute guassian target for center point loss
        bbox = bbox.astype(np.int)
        cnt_target = generate_gaussian_mask_3d(target_shape, bbox)
        # print ('num_pos:', np.sum(cnt_target==1))
        sze_target = np.zeros((target_shape[0], target_shape[1], target_shape[2], 3))
        if not norm_size:
            for i in range(0, bbox.shape[0]):
                sze_target[bbox[i, 0], bbox[i, 1], bbox[i, 2]] = (bbox[i, 3], bbox[i, 4], bbox[i, 5])
        else:
            for i in range(0, bbox.shape[0]):
                sze_target[bbox[i, 0], bbox[i, 1], bbox[i, 2]] = (bbox[i, 3] / 128.0, bbox[i, 4] / 96.0, bbox[i, 5] / 128.0)

        # Crop && Data Augmentation
        if crop:
            crop_size2 = self.cfg.INPUT_SHAPE[1]
            if random_crop:
                random_seed = np.random.randint(11)
                start = 10 + random_seed
            else:
                start = 16
            end = start + crop_size2

            return im_gray[:, start:end, :], cnt_target[:, start:end, :], sze_target[:, start:end, :]
        else:
            return im_gray, cnt_target, sze_target

    def load_image(self, im_file, dtype=np.float32):
        im = np.load(im_file).astype(dtype)
        return im

    def get_img_from_name(self, name, crop=True, channel_first=False):
        if not isinstance(name, list):
            name = [name]
        ims = []
        cnt_targets = []
        sze_targets = []
        for file_name in name:
            img_path = osp.join(self.data_root_dir, file_name, file_name + '_128.npy')
            im = self.load_image(img_path)
            annotation = self.annotations[file_name]
            # print(file_name)
            if self.training:
                im, cnt_target, sze_target = self.create_feed_ims_and_targets(im, annotation, crop=crop,
                                                                              random_crop=True, norm_size=False)
            else:
                im, cnt_target, sze_target = self.create_feed_ims_and_targets(im, annotation, crop=crop,
                                                                              random_crop=False, norm_size=False)
            ims.append(im)
            cnt_targets.append(cnt_target)
            sze_targets.append(sze_target)

        ims = np.array(ims)
        cnt_targets = np.array(cnt_targets)
        sze_targets = np.array(sze_targets)

        cnt_targets = np.expand_dims(cnt_targets, axis=-1)
        if channel_first:
            ims = np.transpose(ims, (0, 4, 1, 2, 3))
            cnt_targets = np.transpose(cnt_targets, (0, 4, 1, 2, 3))
            sze_targets = np.transpose(sze_targets, (0, 4, 1, 2, 3))
        # print ('ims shape:', ims.shape)
        return ims, cnt_targets, sze_targets

    def get_test_img_from_name(self, name, crop=True, channel_first=False, return_box=False):
        if not isinstance(name, list):
            name = [name]
        ims = []
        annos = []
        for file_name in name:
            if return_box:
                im, anno = self.get_img_bbox(file_name)
                annos.append(anno)
            else:
                img_path = osp.join(self.data_root_dir, file_name, file_name + '_128.npy')
                im = self.load_image(img_path)
            im_gray = hu2gray(im, WL=40, WW=500)
            # Normalization
            im_gray = np.expand_dims(im_gray, -1).astype(np.float32) / 255.0
            im_gray = (im_gray - self.cfg.mean) / self.cfg.std
            if crop:
                start = 16
                end = start + cfg.INPUT_SHAPE[1]
                im_gray = im_gray[:, start:end, :]
            ims.append(im_gray)
        ims = np.array(ims)
        if channel_first:
            ims = np.transpose(ims, (0, 4, 1, 2, 3))
        if return_box:
            return ims, annos
        else:
            return ims

    def get_img_bbox(self, file_name):
        '''
        :param file_name: string
        :return: img of numpy array (w, h, d), bbox of numpy array [x, y, z, w, h, d]
        '''
        img_path = osp.join(self.data_root_dir, file_name, file_name + '_128.npy')
        im = self.load_image(img_path)
        annotation = self.annotations[file_name]
        return im, np.array(annotation)

    def get_img(self, file_name, gray=True):
        img_path = osp.join(self.data_root_dir, file_name, file_name + '_.npy')
        if os.path.exists(img_path):
            im = self.load_image(img_path)
        else:
            im, bbox = load_raw_data(self.data_root_dir, file_name, resize_same=True)
        if gray:
            im = hu2gray(im, WL=40, WW=350)
        return im, None

    def save_raw_bbox(self):
        for file_name in self.train_list:
            im, bbox = self.get_img_bbox(file_name)
            bbox_pred = np.array(self.detect_results[file_name]['bbox_pred'], dtype=np.float32)
            bbox_gt = np.array(self.detect_results[file_name]['bbox_gt'], dtype=np.float32)
            bbox_pred[:, :3] += bbox[:3] - bbox_gt[:3]
            bbox_gt_raw = bbox / 128.0
            bbox_pred_raw = bbox_pred / 128.0
            self.detect_results[file_name]['bbox_pred_raw'] = bbox_pred_raw.tolist()
            self.detect_results[file_name]['bbox_gt_raw'] = bbox_gt_raw.tolist()
        with open(self.results_file, 'w') as f:
            json.dump(self.detect_results, f)

    def get_all_bbox(self, file_name, channel_first=False, crop=False):
        im, bbox = self.get_img(file_name)
        bbox_pred = np.array(self.detect_results[file_name]['bbox_pred'], dtype=np.float32)
        bbox_gt = np.array(self.detect_results[file_name]['bbox_gt'], dtype=np.float32)
        bbox_pred *= np.array([im.shape[0], im.shape[1], im.shape[2], im.shape[0], im.shape[1], im.shape[2]])
        bbox_gt *= np.array([im.shape[0], im.shape[1], im.shape[2], im.shape[0], im.shape[1], im.shape[2]])
        if crop:
            dist = dist_3d_max(bbox_pred, bbox_gt)
            bbox_volume = get_crop_region(im, bbox_pred[:, :3], random_crop=False)
            bbox_volume = [hu2gray(v, WL=40, WW=500) for v in bbox_volume]
            label = [self.labels[file_name] + 1 if e else 0 for e in dist]
        else:
            iou = IOU_3d_max(bbox_pred, bbox_gt)
            bbox_volume = get_bbox_region(im, bbox_pred)
            bbox_volume = [hu2gray(resize(v, (32, 32, 32)), WL=40, WW=500) for v in bbox_volume]
            label = [0 if e < 0.3 else self.labels[file_name] + 1 for e in iou]
        bbox_volume = np.array(bbox_volume, dtype=np.float32) / 255.0
        bbox_volume = (bbox_volume - self.cfg.mean) / self.cfg.std
        if channel_first:
            return np.expand_dims(bbox_volume, 1), label
        else:
            return np.expand_dims(bbox_volume, -1), label

    def get_bbox_cls_region(self, num_scans, max_num_box=None, channel_first=False):
        # def resize_32(v):
        #     return resize(v, (32, 32, 32))
        labels = []
        volumes = []
        for i in range(num_scans):
            file_name = self.train_list[int((self.train_file_idx + i) % len(self.train_list))]
            # im, bbox = self.get_img_bbox(file_name)
            # im, bbox = load_raw_data(self.data_root_dir, file_name)
            im, bbox = self.get_img(file_name)
            if self.training:
                bbox_pred = np.array(self.detect_results[file_name]['bbox_pred_raw'], dtype=np.float32)
                bbox_gt = np.array(self.detect_results[file_name]['bbox_gt_raw'], dtype=np.float32)
            else:
                bbox_pred = np.array(self.detect_results[file_name]['bbox_pred'], dtype=np.float32)
                bbox_gt = np.array(self.detect_results[file_name]['bbox_gt'], dtype=np.float32)
            bbox_pred *= np.array([im.shape[0], im.shape[1], im.shape[2], im.shape[0], im.shape[1], im.shape[2]])
            bbox_gt *= np.array([im.shape[0], im.shape[1], im.shape[2], im.shape[0], im.shape[1], im.shape[2]])
            # print(im.shape)
            # print(bbox)
            # print(bbox_gt)
            # print(bbox_pred)
            if not self.training:
                # bbox_ = bbox_pred[:8]
                # print(bbox_, bbox)
                iou = IOU_3d_max(bbox_pred, bbox_gt)
                # print(iou)
                bbox_volume = get_bbox_region(im, bbox_pred)

                bbox_volume = [hu2gray(resize(v, (32, 32, 32)), WL=40, WW=500) for v in bbox_volume]

                volumes += bbox_volume
                label = [0 if e < 0.3 else self.labels[file_name] + 1 for e in iou]
                labels += label
            else:
                iou = IOU_3d_max(bbox_pred, bbox_gt)
                # print(iou)
                bbox_tp = bbox_pred[iou > 0.3]
                bbox_tp = np.vstack((bbox_gt, bbox_tp))
                bbox_fp = bbox_pred[iou < 0.1]

                # print(bbox_tp.shape, bbox_fp.shape)

                bbox_volume_tp = get_bbox_region(im, bbox_tp, random_crop=self.training)
                bbox_volume_fp = get_bbox_region(im, bbox_fp)

                valid_fp_id = []
                for idx, v in enumerate(bbox_volume_fp):
                    h, w, d = v.shape
                    if h < 3 or w < 3 or d < 3:
                        continue
                    else:
                        valid_fp_id.append(idx)
                # num_tps = len(bbox_volume_tp)
                # if num_tps > 4:
                #     choice_tp_id = np.random.choice(np.arange(0, num_tps), 4, replace=False)
                #     choice_fp_id = np.random.choice(valid_fp_id, 4, replace=False)
                # else:
                #     choice_tp_id = np.arange(0, num_tps)
                #     choice_fp_id = np.random.choice(valid_fp_id, 8 - num_tps, replace=False)
                bbox_volume_fp = [bbox_volume_fp[idx] for idx in valid_fp_id]
                # bbox_volume_tp = [bbox_volume_tp[idx] for idx in choice_tp_id]

                # bbox_volume_tp = list(map(resize_32, bbox_volume_tp))
                bbox_volume_tp = [hu2gray(resize(v, (32, 32, 32)), WL=40, WW=500) for v in bbox_volume_tp]
                bbox_volume_fp = [hu2gray(resize(v, (32, 32, 32)), WL=40, WW=500) for v in bbox_volume_fp]
                label = [self.labels[file_name] + 1] * len(bbox_volume_tp) + [0] * len(bbox_volume_fp)
                volumes += bbox_volume_tp
                volumes += bbox_volume_fp
                labels += label

        self.train_file_idx += num_scans
        if self.train_file_idx >= len(self.train_list):
            self.train_file_idx = 0

        volumes = np.array(volumes)
        labels = np.array(labels)
        # print(volumes.shape, labels.shape)

        idx = np.arange(0, volumes.shape[0])
        if self.training:
            np.random.shuffle(idx)
        if max_num_box is None:
            max_num_box = len(idx)
        volumes = volumes[idx[:max_num_box]]
        labels = labels[idx[:max_num_box]]
        volumes = volumes.astype(np.float32) / 255.0
        volumes = (volumes - self.cfg.mean) / self.cfg.std
        if channel_first:
            return np.expand_dims(volumes, 1), labels
        else:
            return np.expand_dims(volumes, -1), labels

    def get_bbox_cls_region_crop(self, num_scans, max_num_box=None, channel_first=False):
        labels = []
        volumes = []
        for i in range(num_scans):
            file_name = self.train_list[int((self.train_file_idx + i) % len(self.train_list))]
            # im, bbox = self.get_img_bbox(file_name)
            im, bbox = load_raw_data(self.data_root_dir, file_name)
            # im, bbox = self.get_img(file_name)
            bbox_pred = np.array(self.detect_results[file_name]['bbox_pred'], dtype=np.float32)
            bbox_gt = np.array(self.detect_results[file_name]['bbox_gt'], dtype=np.float32)
            bbox_pred *= np.array([im.shape[0], im.shape[1], im.shape[2], im.shape[0], im.shape[1], im.shape[2]])
            bbox_gt *= np.array([im.shape[0], im.shape[1], im.shape[2], im.shape[0], im.shape[1], im.shape[2]])
            if not self.training:
                dist = dist_3d_max(bbox_pred, bbox_gt, weight=0.5)
                print(dist[:10], file_name)
                bbox_volume = get_crop_region(im, bbox_pred[:, :3], random_crop=False)

                bbox_volume = [hu2gray(v, WL=40, WW=500) for v in bbox_volume]

                volumes += bbox_volume
                label = [self.labels[file_name] + 1 if e else 0 for e in dist]
                labels += label
            else:
                bbox_pred = np.vstack((bbox_gt, bbox_pred))
                dist = dist_3d_max(bbox_pred, bbox_gt, weight=0.8)
                bbox_volume = get_crop_region(im, bbox_pred[:, :3], random_crop=True)
                bbox_volume = [hu2gray(v, WL=40, WW=500) for v in bbox_volume]

                label = [self.labels[file_name] + 1 if e else 0 for e in dist]
                volumes += bbox_volume
                labels += label

        self.train_file_idx += num_scans
        if self.train_file_idx >= len(self.train_list):
            self.train_file_idx = 0

        volumes = np.array(volumes)
        labels = np.array(labels)
        # print(volumes.shape, labels.shape)

        idx = np.arange(0, volumes.shape[0])

        if self.training:
            np.random.shuffle(idx)
        if max_num_box is None:
            max_num_box = len(idx)
        volumes = volumes[idx[:max_num_box]]
        labels = labels[idx[:max_num_box]]
        volumes = volumes.astype(np.float32) / 255.0
        volumes = (volumes - self.cfg.mean) / self.cfg.std
        if channel_first:
            return np.expand_dims(volumes, 1), labels
        else:
            return np.expand_dims(volumes, -1), labels


if __name__ == '__main__':
    datagenerator = DataGenerator(cfg, training=False, mode='cls', data_root=cfg.TEST_DATA_ROOT,
                                  annotation_file=cfg.test_anno_file, results_file=cfg.test_results_file, label_file=None)
    import time
    t1 = time.time()
    v, l = datagenerator.get_bbox_cls_region(1, max_num_box=32)

    # v, l = datagenerator.get_bbox_cls_region_crop(2, 16)
    # t3 = time.time()
    print(v.shape, l)
    # datagenerator.save_raw_bbox(
    # im, cnt, sze = datagenerator.next_batch(2)
    # print(im.shape, cnt.shape, sze.shape)
    t2 = time.time()
    print(t2-t1)




