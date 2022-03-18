import SimpleITK as sitk
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import cv2
from scipy import ndimage
import json
import imageio
import torch

from make_db import read_annotations, read_bboxes, scale2newspacing
from data_processor import resize, hu2gray, interpolate_volume


def get_imgs_mean_std(datagenerator):
    mean = []
    std = []
    for i in tqdm(len(datagenerator.train_list)):
        im, _, _ = datagenerator.next_batch(1)
        mi = np.mean(im)
        mean.append(mi)
        std.append(np.sqrt(np.mean((im - mi) ** 2)))
    return np.mean(mean), np.mean(std)


def process_results(sze_gt, cnt_pred, sze_pred):
    pos_gt = np.where(sze_gt[:, :, :, :, 0] != 0)
    pos_sze = sze_gt[pos_gt]
    if np.max(pos_sze) <= 1:
        bboxs_gt = np.array([pos_gt[1][0], pos_gt[2][0], pos_gt[3][0],
                             pos_sze[0, 0] * 128, pos_sze[0, 1] * 96, pos_sze[0, 2] * 128])
    else:
        bboxs_gt = np.array([pos_gt[1][0], pos_gt[2][0], pos_gt[3][0],
                             pos_sze[0, 0], pos_sze[0, 1], pos_sze[0, 2]])
    # bbox['bbox_gt'] = bboxs_gt.tolist()
    peaks = ndimage.maximum_filter(cnt_pred[0], size=(3, 3, 3, 1))
    idx = np.where(peaks == cnt_pred)
    num = np.sum(cnt_pred[idx] > 0.01)
    # nums.append(num)
    sorted_id = np.argsort(cnt_pred[idx])
    max_id = sorted_id[::-1][:num]
    pos = (idx[0][max_id], idx[1][max_id], idx[2][max_id], idx[3][max_id])
    sze = sze_pred[pos]
    if np.max(sze) <= 1:
        sze[:, 0] *= 128
        sze[:, 1] *= 96
        sze[:, 2] *= 128
    sze = sze.astype(np.int)
    score = cnt_pred[pos]
    pos = np.array((pos[1], pos[2], pos[3])).T.astype(np.int)
    return bboxs_gt, pos, sze, score


def get_results(datagenerator, model):
    train_res = {}
    for name in tqdm(datagenerator.train_list):
        bbox = {}
        im, cnt_gt, sze_gt = datagenerator.get_img_from_name(name)
        cnt_pred, sze_pred = model.predict(im)
        bboxs_gt, pos, sze, score = process_results(sze_gt, cnt_pred, sze_pred)
        bboxs_pred = np.hstack((pos, sze))
        bbox['bbox_gt'] = bboxs_gt.tolist()
        bbox['bbox_pred'] = bboxs_pred.tolist()
        bbox['score'] = np.squeeze(score).tolist()
        train_res[name] = bbox
    return train_res


def get_results_torch(datagenerator, model, th=0.01):
    train_res = {}
    device = torch.device('cuda:0')
    model.eval()
    with torch.no_grad():
        for name in tqdm(datagenerator.train_list):
            bbox = {}
            im, anno = datagenerator.get_test_img_from_name(name, crop=False, return_box=True, channel_first=False)

            # im, cnt_gt, sze_gt = datagenerator.get_img_from_name(name)
            im = torch.from_numpy(np.transpose(im, (0, 4, 1, 2, 3))).to(device)
            cnt_pred, sze_pred = model(im)
            cnt_pred = cnt_pred.detach().cpu().numpy()
            sze_pred = sze_pred.detach().cpu().numpy()
            cnt_pred = np.transpose(cnt_pred, (0, 2, 3, 4, 1))
            sze_pred = np.transpose(sze_pred, (0, 2, 3, 4, 1))

            pred_bboxs = generate_bbox_from_pred(cnt_pred, sze_pred, th, (0, 0, 0))
            pos, sze, score = pred_bboxs[0]
            bbox_gt = anno[0] / 128.0
            bbox_pred = np.hstack((pos, sze))
            bbox['bbox_gt'] = bbox_gt.tolist()
            bbox['bbox_pred'] = bbox_pred.tolist()
            bbox['score'] = score.tolist()
            train_res[name] = bbox
    return train_res

def get_results_torch_FPN(datagenerator, model, th=0.01):
    train_res = {}
    device = torch.device('cuda:0')
    model.eval()
    with torch.no_grad():
        for name in tqdm(datagenerator.train_list):
            bbox = {}
            im, anno = datagenerator.get_test_img_from_name(name, return_box=True, channel_first=False)

            # im, cnt_gt, sze_gt = datagenerator.get_img_from_name(name)
            im = torch.from_numpy(np.transpose(im, (0, 4, 1, 2, 3))).to(device)
            preds = model(im)
            bbox['bbox_gt'] = []
            bbox['bbox_pred'] = []
            bbox['score'] = []
            # cnt_pred, sze_pred = model(im)
            for cnt_pred, sze_pred in preds:
                cnt_pred = cnt_pred.detach().cpu().numpy()
                sze_pred = sze_pred.detach().cpu().numpy()
                cnt_pred = np.transpose(cnt_pred, (0, 2, 3, 4, 1))
                sze_pred = np.transpose(sze_pred, (0, 2, 3, 4, 1))

                pred_bboxs = generate_bbox_from_pred(cnt_pred, sze_pred, th, (0, 0, 0))
                pos, sze, score = pred_bboxs[0]
                bbox_gt = anno[0] / 128.0
                bbox_pred = np.hstack((pos, sze))
                bbox['bbox_gt'] += bbox_gt.tolist()
                bbox['bbox_pred'] += bbox_pred.tolist()
                bbox['score'] += score.tolist()
            train_res[name] = bbox
    return train_res


def get_all_results(datagenerator, model):
    res = {}
    for name in tqdm(datagenerator.train_list):
        bbox = {}
        im, anno = datagenerator.get_test_img_from_name(name, return_box=True)
        cnt_pred, sze_pred = model.predict(im)
        pred_bboxs = generate_bbox_from_pred(cnt_pred, sze_pred)
        pos, sze, score = pred_bboxs[0]
        bbox_gt = anno[0] / 128.0
        bbox_pred = np.hstack((pos, sze))
        bbox['bbox_gt'] = bbox_gt.tolist()
        bbox['bbox_pred'] = bbox_pred.tolist()
        bbox['score'] = score.tolist()
        res[name] = bbox
    return res


def generate_bbox_from_pred(cnt_pred, sze_pred, th=0.01, offset=(0, 16, 0)):
    '''
    :param cnt_pred: (n, w, h, d, 1)
    :param sze_pred: (n, w, h, d, 3)
    :param th: 阈值
    :param offset: 预测的是裁剪后的图，需要恢复到原图
    :return: list: 每个bbox包含(位置、大小、分数), 均归一化
    '''
    bboxs = []
    n = cnt_pred.shape[0]
    cnt_pred = np.squeeze(cnt_pred, -1)
    for i in range(n):
        cnt_pred_temp = cnt_pred[i]
        sze_pred_temp = sze_pred[i]
        peaks = ndimage.maximum_filter(cnt_pred_temp, size=(3, 3, 3))
        idx = np.where(peaks == cnt_pred_temp)
        num = np.sum(cnt_pred_temp[idx] > th)
        # nums.append(num)
        sorted_id = np.argsort(cnt_pred_temp[idx])
        max_id = sorted_id[::-1][:num]
        pos = (idx[0][max_id], idx[1][max_id], idx[2][max_id])
        sze = sze_pred_temp[pos]
        # if np.max(sze) <= 1:
            # sze[:, 0] *= 128
            # sze[:, 1] *= 96.0 / 128
            # sze[:, 2] *= 128
        # sze = sze
        score = cnt_pred_temp[pos]
        pos = (np.array((pos[0], pos[1], pos[2])).T + np.array(offset)) / 128
        bboxs.append([pos, sze, score])
    return bboxs


def IOU_3d(bbox_pred, bbox_gt, iobb=False, offset=1.0/128):
    if isinstance(bbox_pred, list):
        bbox_pred = np.array(bbox_pred)
    if isinstance(bbox_gt, list):
        bbox_gt = np.array(bbox_gt)
    if bbox_pred.shape[1] == 0:
        return np.zeros(bbox_gt.shape[0])
    xmin = bbox_pred[:, 0] - bbox_pred[:, 3] / 2
    xmax = bbox_pred[:, 0] + bbox_pred[:, 3] / 2
    ymin = bbox_pred[:, 1] - bbox_pred[:, 4] / 2
    ymax = bbox_pred[:, 1] + bbox_pred[:, 4] / 2
    zmin = bbox_pred[:, 2] - bbox_pred[:, 5] / 2
    zmax = bbox_pred[:, 2] + bbox_pred[:, 5] / 2
    xleft = np.maximum(xmin, bbox_gt[0] - bbox_gt[3] / 2)
    xright = np.minimum(xmax, bbox_gt[0] + bbox_gt[3] / 2)
    yleft = np.maximum(ymin, bbox_gt[1] - bbox_gt[4] / 2)
    yright = np.minimum(ymax, bbox_gt[1] + bbox_gt[4] / 2)
    zleft = np.maximum(zmin, bbox_gt[2] - bbox_gt[5] / 2)
    zright = np.minimum(zmax, bbox_gt[2] + bbox_gt[5] / 2)
    area_gt = bbox_gt[3] * bbox_gt[4] * bbox_gt[5]
    area_pred = bbox_pred[:, 3] * bbox_pred[:, 4] * bbox_pred[:, 5]
    x = np.maximum(xright - xleft + offset, 0)
    y = np.maximum(yright - yleft + offset, 0)
    z = np.maximum(zright - zleft + offset, 0)
    inter_area = x * y * z
    if iobb:
        IOU = inter_area / area_gt
    else:
        IOU = inter_area / (area_gt + area_pred - inter_area)

    return IOU


def dist_3d(bbox_pred, bbox_gt, weight=0.5):
    if isinstance(bbox_pred, list):
        bbox_pred = np.array(bbox_pred)
    if isinstance(bbox_gt, list):
        bbox_gt = np.array(bbox_gt)
    dist = np.mean((bbox_pred[:, :3] - bbox_gt[:3]) ** 2, axis=1) ** 0.5
    min_dist = np.mean(bbox_gt[3:]) * weight
#     print(dist, min_dist)
    return dist < min_dist


def dist_3d_max(bbox_pred, bbox_gt, weight=0.5):
    if bbox_pred.ndim == 1:
        bbox_pred = np.expand_dims(bbox_pred, 0)
    if bbox_gt.ndim == 1:
        bbox_gt = np.expand_dims(bbox_gt, 0)
    dist = np.zeros(bbox_pred.shape[0])
    for i in range(0, bbox_gt.shape[0]):
        dist = np.maximum(dist, dist_3d(bbox_pred, bbox_gt[i], weight=weight))
    return dist


def IOU_3d_max(bbox_pred, bbox_gt, offset=1.0 / 128):
    '''
    :param bbox_pred: (n1, 6)
    :param bbox_gt: (n2, 6)
    :param offset:
    :return: (n1,) 预测框与真实框最大 IOU
    '''
    if bbox_pred.ndim == 1:
        bbox_pred = np.expand_dims(bbox_pred, 0)
    if bbox_gt.ndim == 1:
        bbox_gt = np.expand_dims(bbox_gt, 0)
    iou = np.zeros(bbox_pred.shape[0])
    for i in range(0, bbox_gt.shape[0]):
        iou = np.maximum(iou, IOU_3d(bbox_pred, bbox_gt[i], offset=offset))
    return iou


def nms_3d(bbox, score=None, threshod=0.3):
    if isinstance(bbox, list):
        bbox = np.array(bbox)
    if isinstance(score, list):
        score = np.array(score)
    x1 = bbox[:, 0] - bbox[:, 3] / 2
    x2 = bbox[:, 0] + bbox[:, 3] / 2
    y1 = bbox[:, 1] - bbox[:, 4] / 2
    y2 = bbox[:, 1] + bbox[:, 4] / 2
    z1 = bbox[:, 2] - bbox[:, 5] / 2
    z2 = bbox[:, 2] + bbox[:, 5] / 2
    if score is None:
        score = bbox[:, 6]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)  # 面积
    ordered_id = np.argsort(score)[::-1]

    keep = []
    while (ordered_id.size > 0):
        i = ordered_id[0]
        keep.append(i)
        # 相交区域
        xx1 = np.maximum(x1[i], x1[ordered_id[1:]])
        yy1 = np.maximum(y1[i], y1[ordered_id[1:]])
        xx2 = np.minimum(x2[i], x2[ordered_id[1:]])
        yy2 = np.minimum(y2[i], y2[ordered_id[1:]])
        zz1 = np.minimum(z1[i], z1[ordered_id[1:]])
        zz2 = np.minimum(z2[i], z2[ordered_id[1:]])
        # 计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        d = np.maximum(0.0, zz2 - zz1 + 1)
        inter = w * h * d

        iou = inter / (areas[i] + areas[ordered_id[1:]] - inter)
        idx = np.where(iou <= threshod)[0]
        ordered_id = ordered_id[idx + 1]
    return keep


def froc(train_res, method='iou', mesh=np.arange(0.01, 0.5, 0.01), iou_threshold=0.3, nms=False):
    train_fps = []
    train_tps = []
    for name in tqdm(train_res.keys()):
        bbox_pred = np.array(train_res[name]['bbox_pred'])
        if bbox_pred.ndim == 1:
            bbox_pred = np.expand_dims(bbox_pred, 0)
        bbox_gt = np.array(train_res[name]['bbox_gt'])
        if bbox_gt.ndim == 1:
            bbox_gt = np.expand_dims(bbox_gt, 0)
        score = np.array(train_res[name]['score'])

        if nms:
            keep = nms_3d(bbox_pred, score, threshod=0.5)
            bbox_pred = bbox_pred[keep]
            score = score[keep]
        if method == 'iou':
            iou = np.zeros(bbox_pred.shape[0])
            for i in range(0, bbox_gt.shape[0]):
                iou = np.maximum(iou, IOU_3d(bbox_pred, bbox_gt[i]))
        elif method == 'iobb':
            iobb = np.zeros(bbox_pred.shape[0])
            for i in range(0, bbox_gt.shape[0]):
                iobb = np.maximum(iobb, IOU_3d(bbox_pred, bbox_gt[i], iobb=True))
        else:
            dists = np.zeros(bbox_pred.shape[0])
            for i in range(0, bbox_gt.shape[0]):
                dists = np.maximum(dists, dist_3d(bbox_pred, bbox_gt[i]))
        fps = []
        tps = []
        for th in mesh:
            idx = np.where(np.array(score) > th)
            num = idx[0].shape[0]
            if method == 'iou':
                fp = np.sum(iou[idx] <= iou_threshold)
            elif method == 'iobb':
                fp = np.sum(iobb[idx] <= iou_threshold)
            else:
                fp = num - np.sum(dists[idx])
            tps.append(num - fp)
            fps.append(fp)
        train_fps.append(fps)
        train_tps.append(tps)
    train_fps = np.array(train_fps)
    train_tps = np.array(train_tps)
    fps = np.mean(train_fps, axis=0)
    train_tps[train_tps > 1] = 1
    tps = np.mean(train_tps, axis=0)
    return fps, tps, train_tps


def draw_pred_results(im, bbox_pred, bbox_gt, num_bbox=None, dir='./results/img/'):
    '''
    :param im: numpy array (w, h, d)
    :param bbox_pred: numpy array (n, 6)  已归一化
    :param bbox_gt:  numpy array (n, 6)  已归一化
    :param num_bbox:  max num of bboxes to draw
    :return:
    '''
    if bbox_gt.ndim == 1:
        bbox_gt = np.expand_dims(bbox_gt, 0)
    if bbox_pred.ndim == 1:
        bbox_pred = np.expand_dims(bbox_pred, 0)
    n = bbox_pred.shape[0]
    if num_bbox is None:
        num_bbox = n
    w, h, t = im.shape
    bbox_pred = (bbox_pred * np.array([w, h, t, w, h, t]))
    bbox_gt = (bbox_gt * np.array([w, h, t, w, h, t]))
    for j in range(w):
        vis_im = np.zeros((h, t, 3))
        # print(vis_im.shape)
        vis_im[..., 0] = im[j, :, :]
        vis_im[..., 1] = im[j, :, :]
        vis_im[..., 2] = im[j, :, :]
        for i in range(bbox_gt.shape[0]):
            a, b = int(bbox_gt[i, 1] - bbox_gt[i, 4] / 2), int(bbox_gt[i, 2] - bbox_gt[i, 5] / 2)
            c, d = int(bbox_gt[i, 1] + bbox_gt[i, 4] / 2), int(bbox_gt[i, 2] + bbox_gt[i, 5] / 2)
            if bbox_gt[i, 0] - bbox_gt[i, 3] / 2 <= j <= bbox_gt[i, 0] + bbox_gt[i, 3] / 2:
                cv2.rectangle(vis_im, (b, a), (d, c), (0, 255, 0), 2)
        for i in range(min(n, num_bbox)):
            a, b = int(bbox_pred[i, 1] - bbox_pred[i, 4] / 2), int(bbox_pred[i, 2] - bbox_pred[i, 5] / 2)
            c, d = int(bbox_pred[i, 1] + bbox_pred[i, 4] / 2), int(bbox_pred[i, 2] + bbox_pred[i, 5] / 2)
            if bbox_pred[i, 0] - bbox_pred[i, 3] / 2 <= j <= bbox_pred[i, 0] + bbox_pred[i, 3] / 2:
                cv2.rectangle(vis_im, (b, a), (d, c), (255, 0, 0), 2)
        imageio.imsave(os.path.join(dir, '{}.png'.format(j)), np.uint8(vis_im))


def draw_results(im, pos_gt, size_gt, pos, sze, num_bbox=20):
    n, _ = pos.shape
    _, w, h, t, _ = im.shape
    # print(w, h, t)
    # print(im[0, 1, :, :, 0].shape)
    for j in range(w):
        vis_im = np.zeros((h, t, 3))
        # print(vis_im.shape)
        vis_im[...,0] = im[0, j, :, :, 0]
        vis_im[...,1] = im[0, j, :, :, 0]
        vis_im[...,2] = im[0, j, :, :, 0]
        vis_im = np.uint8(255.0 * (vis_im * 0.278 + 0.196))
        if pos_gt[0] - size_gt[0] / 2 <= j <= pos_gt[0] + size_gt[0] / 2:
            a, b = int(pos_gt[1] - size_gt[1] / 2), int(pos_gt[2] - size_gt[2] / 2)
            c, d = int(pos_gt[1] + size_gt[1] / 2), int(pos_gt[2] + size_gt[2] / 2)
            cv2.rectangle(vis_im, (b, a), (d, c), (0, 255, 0), 1)
        for i in range(min(n, num_bbox)):
            a, b = int(pos[i][1] - sze[i][1] / 2), int(pos[i][2] - sze[i][2] / 2)
            c, d = int(pos[i][1] + sze[i][1] / 2), int(pos[i][2] + sze[i][2] / 2)
            if pos[i][0] - sze[i][0] / 2 <= j <= pos[i][0] + sze[i][0] / 2:
                cv2.rectangle(vis_im, (b, a), (d, c), (255, 0, 0), 1)
        imageio.imsave('./results/img1/{}.png'.format(j), np.uint8(vis_im))


def draw_result(datagenerator, model):
    name = datagenerator.train_list[19]
    im, cnt_gt, sze_gt = datagenerator.get_img_from_name(name)
    cnt_pred, sze_pred = model.predict(im)

    peaks = ndimage.maximum_filter(cnt_pred[0], size=(3, 3, 3, 1))
    idx = np.where(peaks == cnt_pred)
    sorted_id = np.argsort(cnt_pred[idx])
    max_id = sorted_id[::-1][:20]
    pos = (idx[0][max_id], idx[1][max_id], idx[2][max_id], idx[3][max_id])
    pos_gt = np.where(sze_gt[:, :, :, :, 0] != 0)
    size_gt = sze_gt[pos_gt][0]
    sze = sze_pred[0, pos[1], pos[2], pos[3]]

    if np.max(sze) < 1:
        sze[:, 0] = sze[:, 0] * 128
        sze[:, 1] = sze[:, 1] * 96
        sze[:, 2] = sze[:, 2] * 128

    for j in range(128):
        vis_im = np.zeros((96, 128, 3))
        vis_im[..., 0] = im[0, j, :, :, 0]
        vis_im[..., 1] = im[0, j, :, :, 0]
        vis_im[..., 2] = im[0, j, :, :, 0]
        vis_im = np.uint8(255.0 * (vis_im * 0.278 + 0.196))
        if pos_gt[1][0] - size_gt[0] / 2 <= j <= pos_gt[1][0] + size_gt[0] / 2:
            a, b = int(pos_gt[2][0] - size_gt[1] / 2), int(pos_gt[3][0] - size_gt[2] / 2)
            c, d = int(pos_gt[2][0] + size_gt[1] / 2), int(pos_gt[3][0] + size_gt[2] / 2)
            cv2.rectangle(vis_im, (b, a), (d, c), (0, 255, 0), 1)
        for i in range(10):
            a, b = int(pos[2][i] - sze[i][1] / 2), int(pos[3][i] - sze[i][2] / 2)
            c, d = int(pos[2][i] + sze[i][1] / 2), int(pos[3][i] + sze[i][2] / 2)
            if pos[1][i] - sze[i][0] / 2 <= j <= pos[1][i] + sze[i][0] / 2:
                cv2.rectangle(vis_im, (b, a), (d, c), (255, 0, 0), 1)
        imageio.imsave('./results/img/{}.png'.format(j), np.uint8(vis_im))


def load_raw_data(data_path, data_name, resize_same=False, new_spacing=None):
    raw_data = sitk.ReadImage(os.path.join(data_path, data_name, data_name + '.nii'))
    np_data = sitk.GetArrayFromImage(raw_data)
    origin = np.array(raw_data.GetOrigin())
    spacing = np.array(raw_data.GetSpacing())
    if resize_same:
        if new_spacing:
            new_spacing = np.array(new_spacing)
        else:
            new_spacing = np.array([1.0, 1.0, 1.0])
        new_data = interpolate_volume(np_data, spacing, new_spacing)
        new_data = new_data.astype(np.int16)
        return new_data, None
    annofile = glob.glob(os.path.join(data_path, data_name, '*.acsv'))
    if len(annofile) > 0:
        bbox = read_bboxes(annofile[0], org=origin, spacing=spacing)
    else:
        print(data_name)
        return np_data, None
    # new_bbox = scale2newspacing(bbox, org_spacing=new_size, new_spacing=np.array(raw_data.GetSize()))
    if len(bbox) == 1:
        return np_data, [
            bbox[0][2],
            bbox[0][1],
            bbox[0][0],
            bbox[0][5],
            bbox[0][4],
            bbox[0][3],
        ]
    else:
        print(data_name, "errors")
        return np_data, None


def get_bbox_region(volume, bbox, ext=(1, 1, 1), random_crop=False, crop_large=False, scale=1.0):
    '''
    :param crop_large: 为 True 表示按最长边进行裁剪
    :param random_crop:
    :param ext:
    :param volume: numpy array, shape like (w, h, d, ...)
    :param bbox:  list or numpy array, shape like (num_bboxs, 6 [x, y, z, w, h, d])
    :return:
    '''
    if isinstance(bbox, list):
        bbox = np.array(bbox)
    # print(bbox.shape)
    w = volume.shape[0]
    h = volume.shape[1]
    d = volume.shape[2]
    dim = bbox.ndim
    if dim == 1:
        bbox = np.expand_dims(bbox, 0)
    if random_crop:
        bbox_2 = bbox.copy()
        bbox_2[:, :3] += np.random.randint(-2, 3, (bbox.shape[0], 3))
        bbox_2[:, 3:6] += np.random.randint(-1, 2, (bbox.shape[0], 3))
        bbox = np.vstack((bbox, bbox_2))
        # print(bbox)
    bbox[bbox < 0] = 0
    if crop_large:
        max_size = np.max(bbox[:, 3:5], axis=1)
        bbox[:, 3] = max_size * scale
        bbox[:, 4] = max_size * scale
        bbox[:, 5] = 32
    x1 = np.maximum(0, bbox[:, 0] - bbox[:, 3] / 2 - ext[0]).astype(np.int)
    x2 = np.minimum(w, bbox[:, 0] + bbox[:, 3] / 2 + 1 + ext[0]).astype(np.int)
    y1 = np.maximum(0, bbox[:, 1] - bbox[:, 4] / 2 - ext[1]).astype(np.int)
    y2 = np.minimum(h, bbox[:, 1] + bbox[:, 4] / 2 + 1 + ext[1]).astype(np.int)
    z1 = np.maximum(0, bbox[:, 2] - bbox[:, 5] / 2 - ext[2]).astype(np.int)
    z2 = np.minimum(d, bbox[:, 2] + bbox[:, 5] / 2 + 1 + ext[2]).astype(np.int)
    results = []
    for i in range(bbox.shape[0]):
        results.append(volume[x1[i]:x2[i], y1[i]:y2[i], z1[i]:z2[i]])

    return results


def get_crop_region(volume, centers, size=(32, 32, 32), random_crop=False, ext=2):
    '''
    :param volume: numpy array, shape like (w, h, d, ...)
    :param centers: list or numpy array, shape like (num_bboxs, 3 [x, y, z])
    :param size:
    :param random_crop:
    :return:
    '''
    if isinstance(centers, list):
        centers = np.array(centers, dtype=np.int)
    if isinstance(size, int):
        size = (size, size, size)
    if centers.ndim == 1:
        centers = np.expand_dims(centers, axis=0)
    if not random_crop:
        ext = 0
    ext_size = np.array(size) + ext
    centers += ext_size
    volume = np.pad(volume, ((ext_size[0], ext_size[0]), (ext_size[1], ext_size[1]), (ext_size[2], ext_size[2])), 'minimum')
    # w = volume.shape[0]
    # h = volume.shape[1]
    # d = volume.shape[2]
    left_point = centers - ext_size / 2
    left_point = left_point.astype(np.int)
    if random_crop:
        offset = np.random.randint(0, ext * 2, left_point.shape)
        left_point += offset
    results = []
    for i in range(left_point.shape[0]):
        results.append(volume[left_point[i, 0]:left_point[i, 0]+size[0], left_point[i, 1]:left_point[i, 1]+size[1], left_point[i, 2]:left_point[i, 2]+size[2]])
    return results


def resize_same_spacing(data_path, data_names, new_spacing=None):
    if new_spacing is None:
        new_spacing = [1.0, 1.0, 1.0]
    # bboxs = []
    for data_name in tqdm(data_names):
        raw_data = sitk.ReadImage(os.path.join(data_path, data_name, data_name + '.nii'))
        np_data = sitk.GetArrayFromImage(raw_data)
        origin = np.array(raw_data.GetOrigin())
        spacing = np.array(raw_data.GetSpacing())
        # bbox = read_bboxes(os.path.join(data_path, data_name, 'R.acsv'), org=origin, spacing=spacing)
        new_data = interpolate_volume(np_data, spacing, new_spacing)
        new_data = new_data.astype(np.int16)
        # new_bbox = scale2newspacing(bbox, org_spacing=spacing, new_spacing=new_spacing)
        # if len(new_bbox) == 1:
        #     bboxs.append(np.array(new_bbox[0]))
        # else:
        #     print("errors")
        np.save(os.path.join(data_path, data_name, data_name + '_.npy'), new_data)


def resize_same_shape(data_path, data_names, new_size=(128, 128, 128)):
    new_size = np.array(new_size)
    results = {}
    for data_name in tqdm(data_names):
        raw_data = sitk.ReadImage(os.path.join(data_path, data_name, data_name + '.nii'))
        np_data_name = os.path.join(data_path, data_name, data_name + '_128.npy')
        if not os.path.exists(np_data_name):
            np_data = sitk.GetArrayFromImage(raw_data)
            new_data = resize(np_data, new_size)
            np.save(os.path.join(data_path, data_name, data_name + '_128.npy'), new_data)
        origin = np.array(raw_data.GetOrigin())
        spacing = np.array(raw_data.GetSpacing())
        annofile = glob.glob(os.path.join(data_path, data_name, '*.acsv'))
        bbox = []
        if len(annofile) > 0:
            for anno in annofile:
                bbox += read_bboxes(anno, org=origin, spacing=spacing)
        else:
            print(data_name)
            continue
        new_bboxs = scale2newspacing(bbox, org_spacing=new_size, new_spacing=np.array(raw_data.GetSize()))
        results[data_name] = []
        for new_bbox in new_bboxs:
            results[data_name].append({
                'x': new_bbox[2],
                'y': new_bbox[1],
                'z': new_bbox[0],
                'w': new_bbox[5],
                'h': new_bbox[4],
                'd': new_bbox[3],
            })
        # else:
        #     print(data_name, "errors")
    return results

if __name__ == '__main__':
    import os, subprocess
    p = subprocess.Popen("ls", stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    p.wait()
    print(p.stdout.read().decode())
    # with os.popen("ls") as f:
    #     print(f.read())
    print("hello")
