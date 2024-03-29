#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
Example Data Loader for the LIDC data set. This dataloader expects preprocessed data in .npy or .npz files per patient and
a pandas dataframe in the same directory containing the meta-info e.g. file paths, labels, foregound slice-ids.
'''


import numpy as np
import os
from collections import OrderedDict
import pandas as pd
import pickle
import time
import subprocess
# import utils.dataloader_utils as dutils
import json

# # batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
# from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
# from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
# from batchgenerators.transforms.abstract_transforms import Compose
# from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
# from batchgenerators.dataloading import SingleThreadedAugmenter
# from batchgenerators.transforms.spatial_transforms import SpatialTransform
# from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
# from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates



class MyDataloader:
    def __init__(self, cf, train=True):
        self.training = train
        self.data_root_dir = cf.root_dir
        self.annotation_file = cf.anno_file
        self.train_list, self.annotations = self.load_annotations(self.annotation_file)
        with open(cf.cv_file, 'r') as f:
            self.cv = json.load(f)
        self.fold = cf.fold
        self.train_list = self.cv[self.fold]['train']
        self.test_list = self.cv[self.fold]['val']
        self.train_file_idx = 0
        self.test_file_idx = 0
        self.BATCH_SIZE = cf.batch_size
        self.train_steps = len(self.train_list) // self.BATCH_SIZE
        self.test_steps = len(self.test_list) // self.BATCH_SIZE

    def load_annotations(self, anno_file):
        with open(anno_file, 'r') as f:
            annos = json.load(f)
        annotations = {}
        train_list = []
        for filename in annos.keys():
            annoboxs = annos[filename]
            img_path = os.path.join(self.data_root_dir, filename, filename + '_128.npy')
            if not os.path.exists(img_path):
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
            annotations[filename] = self.transform_bbox(annotations[filename])
        np.random.shuffle(train_list)
        return train_list, annotations

    def transform_bbox(self, bbox, maxlen=128):
        if isinstance(bbox, list):
            bbox = np.array(bbox)
        if bbox.ndim == 1:
            bbox = np.expand_dims(bbox, 0)
        bbox_new = np.zeros_like(bbox)
        bbox_new[:, 0] = bbox[:, 1] - bbox[:, 4] / 2    # x1
        bbox_new[:, 2] = bbox[:, 1] + bbox[:, 4] / 2    # x2
        bbox_new[:, 1] = bbox[:, 2] - bbox[:, 5] / 2    # y1
        bbox_new[:, 3] = bbox[:, 2] + bbox[:, 5] / 2    # y2
        bbox_new[:, 4] = bbox[:, 0] - bbox[:, 3] / 2    # z1
        bbox_new[:, 5] = bbox[:, 0] + bbox[:, 3] / 2    # z2
        bbox_new = np.maximum(bbox_new, 0)
        bbox_new = np.minimum(bbox_new, maxlen-1)
        return bbox_new

    def load_image(self, im_file, dtype=np.float32):
        im = np.load(im_file).astype(dtype)
        return im

    @staticmethod
    def hu2gray(volume, WL=40, WW=500):
        '''
        convert HU value to gray scale[0,255] using lung-window(WL/WW=-500/1200)
        '''
        low = WL - 0.5 * WW
        volume = (volume - low) / WW * 255.0
        volume[volume > 255] = 255
        volume[volume < 0] = 0
        volume = np.uint8(volume)
        return volume

    def next_train_batch(self):
        res = self.next_batch(self.train_list, self.train_file_idx)
        # {'data': data, 'seg': seg, 'pid': batch_pids, 'class_target': class_target}
        # cycle the batches
        self.train_file_idx += self.BATCH_SIZE
        if self.train_file_idx >= len(self.train_list):
            self.train_file_idx = 0
            if self.training:
                np.random.shuffle(self.train_list)
        return res

    def next_test_batch(self):
        res = self.next_batch(self.test_list, self.test_file_idx)
        self.test_file_idx += self.BATCH_SIZE
        if self.test_file_idx >= len(self.test_list):
            self.test_file_idx = 0
            if self.training:
                np.random.shuffle(self.test_list)
        return res

    def next_batch(self, train_list, idx):
        # batch data(b, c, x, y, (z)) / seg(b, 1, x, y, (z)) / pids / class_target
        ims = []
        annos = []
        filenames = []
        class_target = []
        roi_masks = []
        segs = []
        for i in range(self.BATCH_SIZE):
            file_name = train_list[int((idx + i) % len(train_list))]
            filenames.append(file_name)
            img_path = os.path.join(self.data_root_dir, file_name, file_name + '_128.npy')
            im = self.load_image(img_path)
            im = self.hu2gray(im).astype(np.float32) / 255.0
            # print("im: ", np.sum(im))
            ims.append(im)
            annotation = np.array(self.annotations[file_name])
            annos.append(annotation)
            class_target.append(np.array([1 for i in range(len(annotation))]))
            roi_masks.append(np.zeros((annotation.shape[0], 1, im.shape[0], im.shape[1], im.shape[2])))
            seg = np.zeros_like(im).astype(np.int)
            for i in range(annotation.shape[0]):
                x1, y1, x2, y2, z1, z2 = annotation[i].astype(np.int)
                seg[x1:x2, y1:y2, z1:z2] = 1
            segs.append(seg)
            # print(file_name)
        ims = np.expand_dims(np.array(ims), 1)
        segs = np.expand_dims(np.array(segs), 1)
        roi_masks = np.array(roi_masks)

        return {'data': ims, 'class_target': class_target, 'pid': filenames, 'bb_target': annos,
                'roi_masks': roi_masks, 'seg': segs}

    def next_patient(self, mode='test'):
        # ims = []
        # annos = []
        # filenames = []
        # class_target = []
        # roi_masks = []
        # segs = []
        if mode != 'test':
            train_list = self.train_list
            self.train_file_idx += 1
            idx = self.train_file_idx
        else:
            train_list = self.test_list
            self.test_file_idx += 1
            idx = self.test_file_idx
        # for i in range(self.BATCH_SIZE):
        file_name = train_list[int(idx % len(train_list))]
        img_path = os.path.join(self.data_root_dir, file_name, file_name + '_128.npy')
        im = self.load_image(img_path)
        im = self.hu2gray(im).astype(np.float32) / 255.0
        # print("im: ", np.sum(im))
        annotation = np.array(self.annotations[file_name])
        class_target = np.array([1 for i in range(len(annotation))])
        roi_masks = np.zeros((annotation.shape[0], 1, im.shape[0], im.shape[1], im.shape[2]))
        seg = np.zeros_like(im).astype(np.int)
        for i in range(annotation.shape[0]):
            x1, y1, x2, y2, z1, z2 = annotation[i].astype(np.int)
            seg[x1:x2, y1:y2, z1:z2] = 1
            # print(file_name)
        im = np.expand_dims(np.expand_dims(im, 0), 0)
        seg = np.expand_dims(np.expand_dims(seg, 0), 0)
        # print((file_name))
        return {'data': im, 'patient_roi_labels': class_target, 'pid': file_name, 'patient_bb_target': annotation,
                'roi_masks': roi_masks, 'seg': seg, 'original_img_shape': (1, 1, 128, 128, 128)}

def get_train_generators(cf, logger):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators.
    selects patients according to cv folds (generated by first run/fold of experiment):
    splits the data into n-folds, where 1 split is used for val, 1 split for testing and the rest for training. (inner loop test set)
    If cf.hold_out_test_set is True, adds the test split to the training data.
    """
    # img = batch['data']
    # gt_class_ids = batch['roi_labels']
    # gt_boxes = batch['bb_target']
    #
    # all_data = load_dataset(cf, logger)
    # all_pids_list = np.unique([v['pid'] for (k, v) in all_data.items()])
    #
    # if not cf.created_fold_id_pickle:
    #     fg = dutils.fold_generator(seed=cf.seed, n_splits=cf.n_cv_splits, len_data=len(all_pids_list)).get_fold_names()
    #     with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'wb') as handle:
    #         pickle.dump(fg, handle)
    #     cf.created_fold_id_pickle = True
    # else:
    #     with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'rb') as handle:
    #         fg = pickle.load(handle)
    #
    # train_ix, val_ix, test_ix, _ = fg[cf.fold]
    #
    # train_pids = [all_pids_list[ix] for ix in train_ix]
    # val_pids = [all_pids_list[ix] for ix in val_ix]
    #
    # if cf.hold_out_test_set:
    #     train_pids += [all_pids_list[ix] for ix in test_ix]
    #
    # train_data = {k: v for (k, v) in all_data.items() if any(p == v['pid'] for p in train_pids)}
    # val_data = {k: v for (k, v) in all_data.items() if any(p == v['pid'] for p in val_pids)}
    #
    # logger.info("data set loaded with: {} train / {} val / {} test patients".format(len(train_ix), len(val_ix), len(test_ix)))
    # batch_gen = {}
    # batch_gen['train'] = create_data_gen_pipeline(train_data, cf=cf, is_training=True)
    # batch_gen['val_sampling'] = create_data_gen_pipeline(val_data, cf=cf, is_training=False)
    # if cf.val_mode == 'val_patient':
    #     batch_gen['val_patient'] = PatientBatchIterator(val_data, cf=cf)
    #     batch_gen['n_val'] = len(val_ix) if cf.max_val_patients is None else min(len(val_ix), cf.max_val_patients)
    # else:
    #     batch_gen['n_val'] = cf.num_val_batches

    return MyDataloader(cf)
#
#
# def get_test_generator(cf, logger):
#     """
#     wrapper function for creating the test batch generator pipeline.
#     selects patients according to cv folds (generated by first run/fold of experiment)
#     If cf.hold_out_test_set is True, gets the data from an external folder instead.
#     """
#     if cf.hold_out_test_set:
#         pp_name = cf.pp_test_name
#         test_ix = None
#     else:
#         pp_name = None
#         with open(os.path.join(cf.exp_dir, 'fold_ids.pickle'), 'rb') as handle:
#             fold_list = pickle.load(handle)
#         _, _, test_ix, _ = fold_list[cf.fold]
#         # warnings.warn('WARNING: using validation set for testing!!!')
#
#     test_data = load_dataset(cf, logger, test_ix, pp_data_path=cf.pp_test_data_path, pp_name=pp_name)
#     logger.info("data set loaded with: {} test patients".format(len(test_ix)))
#     batch_gen = {}
#     batch_gen['test'] = PatientBatchIterator(test_data, cf=cf)
#     batch_gen['n_test'] = len(test_ix) if cf.max_test_patients=="all" else \
#         min(cf.max_test_patients, len(test_ix))
#     return batch_gen
#
#
#
# def load_dataset(cf, logger, subset_ixs=None, pp_data_path=None, pp_name=None):
#     """
#     loads the dataset. if deployed in cloud also copies and unpacks the data to the working directory.
#     :param subset_ixs: subset indices to be loaded from the dataset. used e.g. for testing to only load the test folds.
#     :return: data: dictionary with one entry per patient (in this case per patient-breast, since they are treated as
#     individual images for training) each entry is a dictionary containing respective meta-info as well as paths to the preprocessed
#     numpy arrays to be loaded during batch-generation
#     """
#     if pp_data_path is None:
#         pp_data_path = cf.pp_data_path
#     if pp_name is None:
#         pp_name = cf.pp_name
#     if cf.server_env:
#         copy_data = True
#         target_dir = os.path.join(cf.data_dest, pp_name)
#         if not os.path.exists(target_dir):
#             cf.data_source_dir = pp_data_path
#             os.makedirs(target_dir)
#             subprocess.call('rsync -av {} {}'.format(
#                 os.path.join(cf.data_source_dir, cf.input_df_name), os.path.join(target_dir, cf.input_df_name)), shell=True)
#             logger.info('created target dir and info df at {}'.format(os.path.join(target_dir, cf.input_df_name)))
#
#         elif subset_ixs is None:
#             copy_data = False
#
#         pp_data_path = target_dir
#
#
#     p_df = pd.read_pickle(os.path.join(pp_data_path, cf.input_df_name))
#
#     if cf.select_prototype_subset is not None:
#         prototype_pids = p_df.pid.tolist()[:cf.select_prototype_subset]
#         p_df = p_df[p_df.pid.isin(prototype_pids)]
#         logger.warning('WARNING: using prototyping data subset!!!')
#
#     if subset_ixs is not None:
#         subset_pids = [np.unique(p_df.pid.tolist())[ix] for ix in subset_ixs]
#         p_df = p_df[p_df.pid.isin(subset_pids)]
#         logger.info('subset: selected {} instances from df'.format(len(p_df)))
#
#     if cf.server_env:
#         if copy_data:
#             copy_and_unpack_data(logger, p_df.pid.tolist(), cf.fold_dir, cf.data_source_dir, target_dir)
#
#     class_targets = p_df['class_target'].tolist()
#     pids = p_df.pid.tolist()
#     imgs = [os.path.join(pp_data_path, '{}_img.npy'.format(pid)) for pid in pids]
#     segs = [os.path.join(pp_data_path,'{}_rois.npy'.format(pid)) for pid in pids]
#
#     data = OrderedDict()
#     for ix, pid in enumerate(pids):
#         # for the experiment conducted here, malignancy scores are binarized: (benign: 1-2, malignant: 3-5)
#         targets = [1 if ii >= 3 else 0 for ii in class_targets[ix]]
#         data[pid] = {'data': imgs[ix], 'seg': segs[ix], 'pid': pid, 'class_target': targets}
#         data[pid]['fg_slices'] = p_df['fg_slices'].tolist()[ix]
#
#     return data
#
#
#
# def create_data_gen_pipeline(patient_data, cf, is_training=True):
#     """
#     create mutli-threaded train/val/test batch generation and augmentation pipeline.
#     :param patient_data: dictionary containing one dictionary per patient in the train/test subset.
#     :param is_training: (optional) whether to perform data augmentation (training) or not (validation/testing)
#     :return: multithreaded_generator
#     """
#
#     # create instance of batch generator as first element in pipeline.
#     data_gen = BatchGenerator(patient_data, batch_size=cf.batch_size, cf=cf)
#
#     # add transformations to pipeline.
#     my_transforms = []
#     if is_training:
#         mirror_transform = Mirror(axes=np.arange(cf.dim))
#         my_transforms.append(mirror_transform)
#         spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
#                                              patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
#                                              do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
#                                              alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
#                                              do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
#                                              angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
#                                              do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
#                                              random_crop=cf.da_kwargs['random_crop'])
#
#         my_transforms.append(spatial_transform)
#     else:
#         my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))
#
#     my_transforms.append(ConvertSegToBoundingBoxCoordinates(cf.dim, get_rois_from_seg_flag=False, class_specific_seg_flag=cf.class_specific_seg_flag))
#     all_transforms = Compose(my_transforms)
#     # multithreaded_generator = SingleThreadedAugmenter(data_gen, all_transforms)
#     multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers, seeds=range(cf.n_workers))
#     return multithreaded_generator
#
#
# class BatchGenerator(SlimDataLoaderBase):
#     """
#     creates the training/validation batch generator. Samples n_batch_size patients (draws a slice from each patient if 2D)
#     from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
#     Actual patch_size is obtained after data augmentation.
#     :param data: data dictionary as provided by 'load_dataset'.
#     :param batch_size: number of patients to sample for the batch
#     :return dictionary containing the batch data (b, c, x, y, (z)) / seg (b, 1, x, y, (z)) / pids / class_target
#     """
#     def __init__(self, data, batch_size, cf):
#         super(BatchGenerator, self).__init__(data, batch_size)
#
#         self.cf = cf
#         self.crop_margin = np.array(self.cf.patch_size)/8. #min distance of ROI center to edge of cropped_patch.
#         self.p_fg = 0.5
#
#     def generate_train_batch(self):
#
#         batch_data, batch_segs, batch_pids, batch_targets, batch_patient_labels = [], [], [], [], []
#         class_targets_list =  [v['class_target'] for (k, v) in self._data.items()]
#
#         if self.cf.head_classes > 2:
#             # samples patients towards equilibrium of foreground classes on a roi-level (after randomly sampling the ratio "batch_sample_slack).
#             batch_ixs = dutils.get_class_balanced_patients(
#                 class_targets_list, self.batch_size, self.cf.head_classes - 1, slack_factor=self.cf.batch_sample_slack)
#         else:
#             batch_ixs = np.random.choice(len(class_targets_list), self.batch_size)
#
#         patients = list(self._data.items())
#
#         for b in batch_ixs:
#             patient = patients[b][1]
#
#             data = np.transpose(np.load(patient['data'], mmap_mode='r'), axes=(1, 2, 0))[np.newaxis] # (c, y, x, z)
#             seg = np.transpose(np.load(patient['seg'], mmap_mode='r'), axes=(1, 2, 0))
#             batch_pids.append(patient['pid'])
#             batch_targets.append(patient['class_target'])
#
#             if self.cf.dim == 2:
#                 # draw random slice from patient while oversampling slices containing foreground objects with p_fg.
#                 if len(patient['fg_slices']) > 0:
#                     fg_prob = self.p_fg / len(patient['fg_slices'])
#                     bg_prob = (1 - self.p_fg) / (data.shape[3] - len(patient['fg_slices']))
#                     slices_prob = [fg_prob if ix in patient['fg_slices'] else bg_prob for ix in range(data.shape[3])]
#                     slice_id = np.random.choice(data.shape[3], p=slices_prob)
#                 else:
#                     slice_id = np.random.choice(data.shape[3])
#
#                 # if set to not None, add neighbouring slices to each selected slice in channel dimension.
#                 if self.cf.n_3D_context is not None:
#                     padded_data = dutils.pad_nd_image(data[0], [(data.shape[-1] + (self.cf.n_3D_context*2))], mode='constant')
#                     padded_slice_id = slice_id + self.cf.n_3D_context
#                     data = (np.concatenate([padded_data[..., ii][np.newaxis] for ii in range(
#                         padded_slice_id - self.cf.n_3D_context, padded_slice_id + self.cf.n_3D_context + 1)], axis=0))
#                 else:
#                     data = data[..., slice_id]
#                 seg = seg[..., slice_id]
#
#             # pad data if smaller than pre_crop_size.
#             if np.any([data.shape[dim + 1] < ps for dim, ps in enumerate(self.cf.pre_crop_size)]):
#                 new_shape = [np.max([data.shape[dim + 1], ps]) for dim, ps in enumerate(self.cf.pre_crop_size)]
#                 data = dutils.pad_nd_image(data, new_shape, mode='constant')
#                 seg = dutils.pad_nd_image(seg, new_shape, mode='constant')
#
#             # crop patches of size pre_crop_size, while sampling patches containing foreground with p_fg.
#             crop_dims = [dim for dim, ps in enumerate(self.cf.pre_crop_size) if data.shape[dim + 1] > ps]
#             if len(crop_dims) > 0:
#                 fg_prob_sample = np.random.rand(1)
#                 # with p_fg: sample random pixel from random ROI and shift center by random value.
#                 if fg_prob_sample < self.p_fg and np.sum(seg) > 0:
#                     seg_ixs = np.argwhere(seg == np.random.choice(np.unique(seg)[1:], 1))
#                     roi_anchor_pixel = seg_ixs[np.random.choice(seg_ixs.shape[0], 1)][0]
#                     assert seg[tuple(roi_anchor_pixel)] > 0
#                     # sample the patch center coords. constrained by edges of images - pre_crop_size /2. And by
#                     # distance to the desired ROI < patch_size /2.
#                     # (here final patch size to account for center_crop after data augmentation).
#                     sample_seg_center = {}
#                     for ii in crop_dims:
#                         low = np.max((self.cf.pre_crop_size[ii]//2, roi_anchor_pixel[ii] - (self.cf.patch_size[ii]//2 - self.crop_margin[ii])))
#                         high = np.min((data.shape[ii + 1] - self.cf.pre_crop_size[ii]//2,
#                                        roi_anchor_pixel[ii] + (self.cf.patch_size[ii]//2 - self.crop_margin[ii])))
#                         # happens if lesion on the edge of the image. dont care about roi anymore,
#                         # just make sure pre-crop is inside image.
#                         if low >= high:
#                             low = data.shape[ii + 1] // 2 - (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
#                             high = data.shape[ii + 1] // 2 + (data.shape[ii + 1] // 2 - self.cf.pre_crop_size[ii] // 2)
#                         sample_seg_center[ii] = np.random.randint(low=low, high=high)
#
#                 else:
#                     # not guaranteed to be empty. probability of emptiness depends on the data.
#                     sample_seg_center = {ii: np.random.randint(low=self.cf.pre_crop_size[ii]//2,
#                                                            high=data.shape[ii + 1] - self.cf.pre_crop_size[ii]//2) for ii in crop_dims}
#
#                 for ii in crop_dims:
#                     min_crop = int(sample_seg_center[ii] - self.cf.pre_crop_size[ii] // 2)
#                     max_crop = int(sample_seg_center[ii] + self.cf.pre_crop_size[ii] // 2)
#                     data = np.take(data, indices=range(min_crop, max_crop), axis=ii + 1)
#                     seg = np.take(seg, indices=range(min_crop, max_crop), axis=ii)
#
#             batch_data.append(data)
#             batch_segs.append(seg[np.newaxis])
#
#         data = np.array(batch_data)
#         seg = np.array(batch_segs).astype(np.uint8)
#         class_target = np.array(batch_targets)
#         return {'data': data, 'seg': seg, 'pid': batch_pids, 'class_target': class_target}
#
#
#
# class PatientBatchIterator(SlimDataLoaderBase):
#     """
#     creates a test generator that iterates over entire given dataset returning 1 patient per batch.
#     Can be used for monitoring if cf.val_mode = 'patient_val' for a monitoring closer to actualy evaluation (done in 3D),
#     if willing to accept speed-loss during training.
#     :return: out_batch: dictionary containing one patient with batch_size = n_3D_patches in 3D or
#     batch_size = n_2D_patches in 2D .
#     """
#     def __init__(self, data, cf): #threads in augmenter
#         super(PatientBatchIterator, self).__init__(data, 0)
#         self.cf = cf
#         self.patient_ix = 0
#         self.dataset_pids = [v['pid'] for (k, v) in data.items()]
#         self.patch_size = cf.patch_size
#         if len(self.patch_size) == 2:
#             self.patch_size = self.patch_size + [1]
#
#
#     def generate_train_batch(self):
#
#
#         pid = self.dataset_pids[self.patient_ix]
#         patient = self._data[pid]
#         data = np.transpose(np.load(patient['data'], mmap_mode='r'), axes=(1, 2, 0))[np.newaxis] # (c, y, x, z)
#         seg = np.transpose(np.load(patient['seg'], mmap_mode='r'), axes=(1, 2, 0))
#         batch_class_targets = np.array([patient['class_target']])
#
#         # pad data if smaller than patch_size seen during training.
#         if np.any([data.shape[dim + 1] < ps for dim, ps in enumerate(self.patch_size)]):
#             new_shape = [data.shape[0]] + [np.max([data.shape[dim + 1], self.patch_size[dim]]) for dim, ps in enumerate(self.patch_size)]
#             data = dutils.pad_nd_image(data, new_shape) # use 'return_slicer' to crop image back to original shape.
#             seg = dutils.pad_nd_image(seg, new_shape)
#
#         # get 3D targets for evaluation, even if network operates in 2D. 2D predictions will be merged to 3D in predictor.
#         if self.cf.dim == 3 or self.cf.merge_2D_to_3D_preds:
#             out_data = data[np.newaxis]
#             out_seg = seg[np.newaxis, np.newaxis]
#             out_targets = batch_class_targets
#
#             batch_3D = {'data': out_data, 'seg': out_seg, 'class_target': out_targets, 'pid': pid}
#             converter = ConvertSegToBoundingBoxCoordinates(dim=3, get_rois_from_seg_flag=False, class_specific_seg_flag=self.cf.class_specific_seg_flag)
#             batch_3D = converter(**batch_3D)
#             batch_3D.update({'patient_bb_target': batch_3D['bb_target'],
#                                   'patient_roi_labels': batch_3D['roi_labels'],
#                                   'original_img_shape': out_data.shape})
#
#         if self.cf.dim == 2:
#             out_data = np.transpose(data, axes=(3, 0, 1, 2))  # (z, c, x, y )
#             out_seg = np.transpose(seg, axes=(2, 0, 1))[:, np.newaxis]
#             out_targets = np.array(np.repeat(batch_class_targets, out_data.shape[0], axis=0))
#
#             # if set to not None, add neighbouring slices to each selected slice in channel dimension.
#             if self.cf.n_3D_context is not None:
#                 slice_range = range(self.cf.n_3D_context, out_data.shape[0] + self.cf.n_3D_context)
#                 out_data = np.pad(out_data, ((self.cf.n_3D_context, self.cf.n_3D_context), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
#                 out_data = np.array(
#                     [np.concatenate([out_data[ii] for ii in range(
#                         slice_id - self.cf.n_3D_context, slice_id + self.cf.n_3D_context + 1)], axis=0) for slice_id in
#                      slice_range])
#
#             batch_2D = {'data': out_data, 'seg': out_seg, 'class_target': out_targets, 'pid': pid}
#             converter = ConvertSegToBoundingBoxCoordinates(dim=2, get_rois_from_seg_flag=False, class_specific_seg_flag=self.cf.class_specific_seg_flag)
#             batch_2D = converter(**batch_2D)
#
#             if self.cf.merge_2D_to_3D_preds:
#                 batch_2D.update({'patient_bb_target': batch_3D['patient_bb_target'],
#                                       'patient_roi_labels': batch_3D['patient_roi_labels'],
#                                       'original_img_shape': out_data.shape})
#             else:
#                 batch_2D.update({'patient_bb_target': batch_2D['bb_target'],
#                                  'patient_roi_labels': batch_2D['roi_labels'],
#                                  'original_img_shape': out_data.shape})
#
#         out_batch = batch_3D if self.cf.dim == 3 else batch_2D
#         patient_batch = out_batch
#
#         # crop patient-volume to patches of patch_size used during training. stack patches up in batch dimension.
#         # in this case, 2D is treated as a special case of 3D with patch_size[z] = 1.
#         if np.any([data.shape[dim + 1] > self.patch_size[dim] for dim in range(3)]):
#             patch_crop_coords_list = dutils.get_patch_crop_coords(data[0], self.patch_size)
#             new_img_batch, new_seg_batch, new_class_targets_batch = [], [], []
#
#             for cix, c in enumerate(patch_crop_coords_list):
#
#                 seg_patch = seg[c[0]:c[1], c[2]: c[3], c[4]:c[5]]
#                 new_seg_batch.append(seg_patch)
#
#                 # if set to not None, add neighbouring slices to each selected slice in channel dimension.
#                 # correct patch_crop coordinates by added slices of 3D context.
#                 if self.cf.dim == 2 and self.cf.n_3D_context is not None:
#                     tmp_c_5 = c[5] + (self.cf.n_3D_context * 2)
#                     if cix == 0:
#                         data = np.pad(data, ((0, 0), (0, 0), (0, 0), (self.cf.n_3D_context, self.cf.n_3D_context)), 'constant', constant_values=0)
#                 else:
#                     tmp_c_5 = c[5]
#
#                 new_img_batch.append(data[:, c[0]:c[1], c[2]:c[3], c[4]:tmp_c_5])
#
#             data = np.array(new_img_batch) # (n_patches, c, x, y, z)
#             seg = np.array(new_seg_batch)[:, np.newaxis]  # (n_patches, 1, x, y, z)
#             batch_class_targets = np.repeat(batch_class_targets, len(patch_crop_coords_list), axis=0)
#
#             if self.cf.dim == 2:
#                 if self.cf.n_3D_context is not None:
#                     data = np.transpose(data[:, 0], axes=(0, 3, 1, 2))
#                 else:
#                     # all patches have z dimension 1 (slices). discard dimension
#                     data = data[..., 0]
#                 seg = seg[..., 0]
#
#             patch_batch = {'data': data, 'seg': seg, 'class_target': batch_class_targets, 'pid': pid}
#             patch_batch['patch_crop_coords'] = np.array(patch_crop_coords_list)
#             patch_batch['patient_bb_target'] = patient_batch['patient_bb_target']
#             patch_batch['patient_roi_labels'] = patient_batch['patient_roi_labels']
#             patch_batch['original_img_shape'] = patient_batch['original_img_shape']
#
#             converter = ConvertSegToBoundingBoxCoordinates(self.cf.dim, get_rois_from_seg_flag=False, class_specific_seg_flag=self.cf.class_specific_seg_flag)
#             patch_batch = converter(**patch_batch)
#             out_batch = patch_batch
#
#         self.patient_ix += 1
#         if self.patient_ix == len(self.dataset_pids):
#             self.patient_ix = 0
#
#         return out_batch
#
#
#
# def copy_and_unpack_data(logger, pids, fold_dir, source_dir, target_dir):
#
#
#     start_time = time.time()
#     with open(os.path.join(fold_dir, 'file_list.txt'), 'w') as handle:
#         for pid in pids:
#             handle.write('{}_img.npz\n'.format(pid))
#             handle.write('{}_rois.npz\n'.format(pid))
#
#     subprocess.call('rsync -av --files-from {} {} {}'.format(os.path.join(fold_dir, 'file_list.txt'),
#         source_dir, target_dir), shell=True)
#     dutils.unpack_dataset(target_dir, threads=16)
#     copied_files = os.listdir(target_dir)
#     logger.info("copying and unpacking data set finsihed : {} files in target dir: {}. took {} sec".format(
#         len(copied_files), target_dir, np.round(time.time() - start_time, 0)))

