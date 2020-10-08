# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:37:32 2019

@author: wjcongyu
"""
import cv2
import operator
import numpy as np
import SimpleITK as sitk
from skimage import measure
from scipy import ndimage
from scipy import signal
from skimage import morphology
import math
import fnmatch
import os
import csv
import tensorflow as tf


def find_scans_oswalk(base_path, file_filter='*.dcm'):
    scan_dirs = {}
    for root, dirnames, filenames in os.walk(base_path):
        for file in fnmatch.filter(filenames, file_filter):
            if not root in scan_dirs:
                if '.dcm' in file_filter:
                    scan_dirs[root] = get_series_uids(root)
                else:
                    scan_dirs[root] = []
            if '.dcm' not in file_filter:
                scan_dirs[root].append(file)

    return scan_dirs


def resize(volume, target_shape):
    '''
    resize volume to specified shape
    '''
    if target_shape[0] <= 0:
        target_shape[0] = volume.shape[0]
    if target_shape[1] <= 0:
        target_shape[1] = volume.shape[1]
    if target_shape[2] <= 0:
        target_shape[2] = volume.shape[2]

    D, H, W = volume.shape
    # cv2 can not process image with channels > 512
    if W <= 512:
        res = cv2.resize(np.float32(volume), dsize=(target_shape[1], target_shape[0]))
    else:
        N = 512
        results = []
        for i in range(0, int(W / N + 1)):
            l = i * N
            r = min((i + 1) * N, W)
            patch = volume[:, :, l:r]
            resized_patch = cv2.resize(np.float32(patch), dsize=(target_shape[1], target_shape[0]))
            if len(resized_patch.shape) == 2:
                resized_patch = np.expand_dims(resized_patch, axis=-1)
            results.append(resized_patch)

        res = np.concatenate(results, axis=-1)

    res = np.transpose(res, (2, 1, 0))
    D, H, W = res.shape
    if W <= 512:
        res = cv2.resize(np.float32(res), dsize=(target_shape[1], target_shape[2]))
    else:
        N = 512
        results = []
        for i in range(0, int(W / N + 1)):
            l = i * N
            r = min((i + 1) * N, W)
            patch = res[:, :, l:r]
            resized_patch = cv2.resize(np.float32(patch), dsize=(target_shape[1], target_shape[2]))
            if len(resized_patch.shape) == 2:
                resized_patch = np.expand_dims(resized_patch, axis=-1)
            results.append(resized_patch)

        res = np.concatenate(results, axis=-1)

    res = np.transpose(res, (2, 1, 0))
    return res


def generate_gaussian_mask_3d(mask_shape, bbox_gt):
    H, W, D = mask_shape
    h = range(H)
    w = range(W)
    d = range(D)
    gt_x, gt_y, gt_z, gt_h, gt_w, gt_d = bbox_gt[:, 0], bbox_gt[:, 1], bbox_gt[:, 2], bbox_gt[:, 3], bbox_gt[:,
                                                                                                     4], bbox_gt[:, 5]
    [meshgrid_x, meshgrid_y, meshgrid_z] = np.meshgrid(h, w, d, indexing='ij')
    meshgrid_x = meshgrid_x.astype(np.float32)
    meshgrid_y = meshgrid_y.astype(np.float32)
    meshgrid_z = meshgrid_z.astype(np.float32)
    gt_y = np.reshape(gt_y, [-1, 1, 1, 1]).astype(np.float32)
    gt_x = np.reshape(gt_x, [-1, 1, 1, 1]).astype(np.float32)
    gt_z = np.reshape(gt_z, [-1, 1, 1, 1]).astype(np.float32)

    #     print(gt_y, meshgrid_x)
    sigma = compute_gaussian_radius(gt_h, gt_w, min_overlap=0.7)
    # print('sigma:', sigma, gt_h, gt_w)
    sigma = np.reshape(sigma, [-1, 1, 1, 1]).astype(np.float32)
    gau = np.exp(-((gt_x - meshgrid_x) ** 2 + (gt_y - meshgrid_y) ** 2 + (gt_z - meshgrid_z) ** 2) / (2 * sigma ** 2))
    target = np.zeros(mask_shape)
    for i in range(gau.shape[0]):
        f = gau[i, ...]
        target = np.where(f > target, f, target)
    return target


def generate_gaussian_mask_3d_tf(mask_shape, bbox_gt):
    H, W, D = mask_shape
    h = range(H)
    w = range(W)
    d = range(D)
    gt_x, gt_y, gt_z, gt_h, gt_w, gt_d = bbox_gt[:, 0], bbox_gt[:, 1], bbox_gt[:, 2], bbox_gt[:, 3], bbox_gt[:, 4], bbox_gt[:, 5]
    [meshgrid_x, meshgrid_y, meshgrid_z] = tf.meshgrid(h, w, d, indexing='ij')
    meshgrid_x = tf.cast(meshgrid_x, tf.float32)
    meshgrid_y = tf.cast(meshgrid_y, tf.float32)
    meshgrid_z = tf.cast(meshgrid_z, tf.float32)
    gt_y = tf.cast(tf.reshape(gt_y, [-1, 1, 1, 1]), tf.float32)
    gt_x = tf.cast(tf.reshape(gt_x, [-1, 1, 1, 1]), tf.float32)
    gt_z = tf.cast(tf.reshape(gt_z, [-1, 1, 1, 1]), tf.float32)

    #     print(gt_y, meshgrid_x)
    sigma = compute_gaussian_radius(gt_h, gt_w, min_overlap=0.7)
    # print('sigma:', sigma, gt_h, gt_w)
    sigma = tf.cast(tf.reshape(sigma, [-1, 1, 1, 1]), tf.float32)
    gau = tf.exp(-((gt_x - meshgrid_x) ** 2 + (gt_y - meshgrid_y) ** 2 + (gt_z - meshgrid_z) ** 2) / (2 * sigma ** 2))
    target = tf.zeros(mask_shape)
    for i in range(gau.shape[0]):
        f = gau[i, ...]
        target = tf.where(f > target, f, target)
    return target.numpy()


def generate_gaussian_mask(mask_shape, gt_y, gt_x, gt_h, gt_w):
    H, W = mask_shape

    h = range(0, H)
    w = range(0, W)
    [meshgrid_y, meshgrid_x] = np.meshgrid(h, w, indexing='ij')

    gt_y = np.reshape(gt_y, [-1, 1, 1])
    gt_x = np.reshape(gt_x, [-1, 1, 1])

    sigma = compute_gaussian_radius(gt_h, gt_w, min_overlap=0.5)
    # print('sigma:', sigma)
    sigma = np.reshape(sigma, [-1, 1, 1])
    gau = np.exp(-((gt_x - meshgrid_x) ** 2 + (gt_y - meshgrid_y) ** 2) / (2 * sigma ** 2))
    target = np.zeros(mask_shape)
    for i in range(gau.shape[0]):
        f = gau[i, ...]
        target = np.where(f > target, f, target)
    return target


def compute_gaussian_radius(height, width, min_overlap=0.3):
    a1 = 1.
    b1 = (height + width)
    c1 = width * height * (1. - min_overlap) / (1. + min_overlap)
    sq1 = np.sqrt(b1 ** 2. - 4. * a1 * c1)
    r1 = (b1 + sq1) / 2.
    a2 = 4.
    b2 = 2. * (height + width)
    c2 = (1. - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2. - 4. * a2 * c2)
    r2 = (b2 + sq2) / 2. / a2
    a3 = 4. * min_overlap
    b3 = -2. * min_overlap * (height + width)
    c3 = (min_overlap - 1.) * width * height
    sq3 = np.sqrt(b3 ** 2. - 4. * a3 * c3)
    r3 = (b3 + sq3) / 2. / a3
    return np.min([r1, r2, r3], axis=0)


def hu2gray(volume, WL=40, WW=350):
    '''
    convert HU value to gray scale[0,255] using lung-window(WL/WW=-500/1200)
    '''
    low = WL - 0.5 * WW
    volume = (volume - low) / WW * 255.0
    volume[volume > 255] = 255
    volume[volume < 0] = 0
    volume = np.uint8(volume)
    return volume


####################################################################

def load_series_volume_mhd(mhd_file):
    im_3D = sitk.ReadImage(mhd_file)
    volume = sitk.GetArrayFromImage(im_3D)
    org = np.array(im_3D.GetOrigin())
    spacing = np.array(im_3D.GetSpacing())
    return volume, org, spacing


# data loader
def load_series_volume(base_path, series_uid):
    """
	This Func load 3D volume of series_uid from the base_path
	Params:
		base_path: the directory of series images
		series_uid:the sereis uid to load
	Returns:
		np.array of the volume data, dim:z,y,x
	"""
    sitk.ImageSeriesReader.GlobalWarningDisplayOff()
    series_file_names, series_file_ids = get_sitk_dcm_files(base_path, series_uid)
    if len(series_file_names) < 10:
        return None, [0, 0, 0], [0, 0, 0], 0, []

    try:

        # start load the volume data
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)

        im_3D = series_reader.Execute()

        # image array:z,y,x; origin and spacing:x,y,z
        volume = sitk.GetArrayFromImage(im_3D)
        org = np.array(im_3D.GetOrigin())
        spacing = np.array(im_3D.GetSpacing())

        im_reader = sitk.ImageFileReader()
        im_reader.LoadPrivateTagsOn()

        slice_spacing = 0
        slice_spacing_tag = '0018|0088'
        im_reader.SetFileName(series_file_names[0])
        dcm_im = im_reader.Execute()

        if slice_spacing_tag in dcm_im.GetMetaDataKeys():
            slice_spacing = float(dcm_im.GetMetaData(slice_spacing_tag))

        return volume, org, spacing, slice_spacing, series_file_ids
    except (OSError, TypeError) as reason:
        print(str(reason))
        return None, np.array([0, 0, 0]), np.array([0, 0, 0, 0]), 0, []


def interpolate_volume(volume, org_spacing, expect_spacing):
    D, H, W = volume.shape
    scale = np.array(org_spacing) / np.array(expect_spacing)
    nW, nH, nD = np.int32(np.array([W, H, D]) * scale)
    new_volume = resize(volume, [nD, nH, nW])
    return new_volume


def get_sitk_dcm_files(base_path, series_uid):
    '''
    search the file names of a series at specified directory
    '''
    sitk.ImageSeriesReader.GlobalWarningDisplayOff()
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(base_path, series_uid)
    if len(series_file_names) < 2:
        return series_file_names, None

    # sort dcm files according to ascending order of the image number
    im_reader = sitk.ImageFileReader()
    im_reader.LoadPrivateTagsOn()
    im_numbers = {}
    im_number_tag = '0020|0013'
    for file_name in series_file_names:
        im_reader.SetFileName(file_name)
        dcm_im = im_reader.Execute()
        if im_number_tag in dcm_im.GetMetaDataKeys():
            if not int(dcm_im.GetMetaData(im_number_tag)) in im_numbers:
                im_numbers[int(dcm_im.GetMetaData(im_number_tag))] = file_name

    im_numbers = sorted(im_numbers.items(), key=operator.itemgetter(0))
    series_file_names = [item[1] for item in im_numbers]
    series_file_ids = [item[0] for item in im_numbers]
    return series_file_names, series_file_ids


def get_series_uids(base_path):
    '''
    find the series uids at a specified directory
    '''
    sitk.ImageSeriesReader.GlobalWarningDisplayOff()
    series_uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(base_path)
    return series_uids


def get_series_dcm_nums(base_path, series_uid):
    '''
    get the number of dicom images of a specified series
    '''
    sitk.ImageSeriesReader.GlobalWarningDisplayOff()
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(base_path, series_uid)
    return len(series_file_names)


def get_chest_roi(volume):
    try:
        blurred_volume = cv2.blur(volume.copy(), (3, 3))
    except:
        blurred_volume = volume.copy()

    body = hu2gray(blurred_volume.copy())
    body[body >= 200] = 255
    body[body < 200] = 0

    body_labels = measure.label(body, connectivity=2)
    body_props = measure.regionprops(body_labels)
    if len(body_props) == 0:
        return [0, 0, 0, 0, 0, 0], volume

    body_prop = sorted(body_props, key=operator.itemgetter('area'), reverse=True)[0]
    bz1, by1, bx1, bz2, by2, bx2 = body_prop.bbox
    body[body_labels != body_prop.label] = 0
    for z in range(body.shape[0]):
        body[z] = morphology.dilation(body[z], morphology.square(3))

    body_filled = body.copy()
    body_filled[body_filled == 255] = 1
    for z in range(body_filled.shape[0]):
        body_filled[z] = ndimage.binary_fill_holes(body_filled[z])

    body_filled[body_filled == 1] = 255

    lung = np.bitwise_xor(body_filled, body)
    lung_labels = measure.label(lung, connectivity=2)
    lung_props = measure.regionprops(lung_labels)
    if len(lung_props) == 0:
        return [bx1, by1, bz1, bx2, by2, bz2], body

    lung_props = sorted(lung_props, key=operator.itemgetter('area'), reverse=True)
    lung = np.zeros_like(lung)
    lung[lung_labels == lung_props[0].label] = 255
    N, H, W = lung.shape
    if len(lung_props) > 1 and lung_props[1].area > 0.003 * (N * H * W):
        lung[lung_labels == lung_props[1].label] = 255

    lung_bbox = get_convex_bbox_frm_3dmask(lung)
    if lung_bbox is None:
        return [bx1, by1, bz1, bx2, by2, bz2], body

    return lung_bbox, lung


def get_convex_bbox(bboxes):
    xs = []
    ys = []
    zs = []
    for z in bboxes:
        for bbox in bboxes[z]:
            xs.append(bbox[0])
            xs.append(bbox[2])
            ys.append(bbox[1])
            ys.append(bbox[3])
            zs.append(z)
    x1 = int(min(xs))
    x2 = int(max(xs))
    y1 = int(min(ys))
    y2 = int(max(ys))
    z1 = int(min(zs))
    z2 = int(max(zs))

    return [x1, y1, z1, x2, y2, z2]


def get_convex_bbox_frm_2dmask(mask_label, ext_size=[0, 0]):
    H, W = mask_label.shape
    mask = mask_label.copy()
    mask[mask > 0] = 255

    labels = measure.label(mask, connectivity=2)
    props = measure.regionprops(labels)
    if len(props) == 0:
        return [0, 0, 0, 0]

    xs = []
    ys = []
    for prop in props:
        by1, bx1, by2, bx2 = prop.bbox
        xs.append(bx1)
        xs.append(bx2)
        ys.append(by1)
        ys.append(by2)

    x1 = int(max(0, np.min(xs) - ext_size[0]))
    x2 = int(min(W, np.max(xs) + ext_size[0]))
    y1 = int(max(0, np.min(ys) - ext_size[1]))
    y2 = int(min(H, np.max(ys) + ext_size[1]))

    return [x1, y1, x2, y2]


def get_convex_bbox_frm_3dmask(volume, ext_size=[8, 8, 1]):
    N, H, W = volume.shape
    mask = np.zeros_like(volume)
    mask[volume > 0] = 255
    labels = measure.label(mask, connectivity=2)
    props = measure.regionprops(labels)
    if len(props) == 0:
        return None

    xs = []
    ys = []
    zs = []
    for prop in props:
        bz1, by1, bx1, bz2, by2, bx2 = prop.bbox
        xs.append(bx1)
        xs.append(bx2)
        ys.append(by1)
        ys.append(by2)
        zs.append(bz1)
        zs.append(bz2)
    x1 = int(max(0, np.min(xs) - ext_size[0]))
    x2 = int(min(W, np.max(xs) + ext_size[0]))
    y1 = int(max(0, np.min(ys) - ext_size[1]))
    y2 = int(min(H, np.max(ys) + ext_size[1]))
    z1 = int(max(0, np.min(zs) - ext_size[2]))
    z2 = int(min(N, np.max(zs) + ext_size[2]))

    return [x1, y1, z1, x2, y2, z2]


def keep_max_object(volume):
    labels = measure.label(volume, connectivity=2)
    props = measure.regionprops(labels)
    if len(props) > 0:
        max_prop = sorted(props, key=operator.itemgetter('area'), reverse=True)[0]
        volume[labels != max_prop.label] = 0
    return volume


def fill_gaps3d(mask_volume):
    mask3d = np.uint8(mask_volume.copy())
    mask3d[mask3d != 0] = 255
    for z in range(mask3d.shape[0]):
        mask = np.uint8(mask3d[z, :, :])
        mask = fill_gaps(np.uint8(mask))
        mask = signal.medfilt2d(mask, kernel_size=3)
        mask3d[z, :, :] = mask

    return mask3d


def fill_gaps(lung_mask):
    (contours, _) = cv2.findContours(lung_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        if defects is None:
            continue
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            start_new = find_closet_pt(lung_mask, start)
            if start_new is None:
                continue

            end_new = find_closet_pt(lung_mask, end)
            if end_new is None:
                continue

            arc_len = math.sqrt(pow(end_new[0] - start_new[0], 2) + pow(end_new[1] - start_new[1], 2))
            # print ('arc_len:',arc_len)
            if arc_len > 25:
                continue
            cv2.line(lung_mask, start_new, end_new, [255, 255, 255], 2)

    lung_mask[lung_mask == 255] = 1
    lung_mask = ndimage.binary_fill_holes(lung_mask)
    lung_mask = ndimage.binary_closing(lung_mask, structure=np.ones((5, 5)))
    lung_mask[lung_mask > 0.5] = 255
    return np.uint8(lung_mask)


def find_closet_pt(lung_mask, pt):
    k = 20
    pts = np.argwhere(lung_mask[pt[1] - k:pt[1] + k + 1, pt[0] - k:pt[0] + k + 1] > 0)

    dis = np.sqrt(np.square(pts[:, 0] - k) + np.square(pts[:, 1] - k))
    if len(dis) == 0:
        return None
    pt_ind = np.argmin(dis)
    x = pts[pt_ind][0]
    y = pts[pt_ind][1]

    return (y - k + pt[0], x - k + pt[1])


def voxel2world_coord(voxel_coord, origin, spacing):
    world_coord = (voxel_coord * spacing)
    if origin[0] > 0:
        x = origin[0] - np.absolute(world_coord)[0]
    else:
        x = origin[0] + np.absolute(world_coord)[0]

    if origin[1] > 0:
        y = origin[1] - np.absolute(world_coord)[1]
    else:
        y = origin[1] + np.absolute(world_coord)[1]
    z = origin[2] + np.absolute(world_coord)[2]
    # world_coord = origin - np.absolute(world_coord)
    return [x, y, z]

    return world_coord


def world2voxel_coord(world_coord, origin, spacing):
    voxel_coord = np.absolute(world_coord - origin)
    return voxel_coord / spacing


def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines


def extractCube(scan, xyz, cube_extract_size_mm, spacing, cube_target_size=-1):
    # Extract cube of cube_size^3 voxels and world dimensions of cube_size_mm^3 mm from scan at image coordinates xyz
    xyz = np.array([xyz[i] for i in [2, 1, 0]], np.int)
    spacing = np.array([spacing[i] for i in [2, 1, 0]])
    scan_halfcube_size = np.array(cube_extract_size_mm / spacing / 2, np.int)
    if np.any(xyz < scan_halfcube_size) or np.any(
            xyz + scan_halfcube_size > scan.shape):  # check if padding is necessary
        maxsize = max(scan_halfcube_size)
        scan = np.pad(scan, ((maxsize, maxsize), (maxsize, maxsize), (maxsize, maxsize)), 'constant',
                      constant_values=-1000)
        xyz = xyz + maxsize

    scancube = scan[xyz[0] - scan_halfcube_size[0]:xyz[0] + scan_halfcube_size[0],  # extract cube from scan at xyz
               xyz[1] - scan_halfcube_size[1]:xyz[1] + scan_halfcube_size[1],
               xyz[2] - scan_halfcube_size[2]:xyz[2] + scan_halfcube_size[2]]

    if cube_target_size > 0:
        scancube = resize(scancube, [cube_target_size, cube_target_size, cube_target_size])  # resample for cube_size

    return scancube
