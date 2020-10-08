import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-anno_file', '--anno_file', help='the nodule annotation file', type=str,
                    default='/media/wjcy/AICC-2T-1/label_list.xlsx')
parser.add_argument('-scan_path', '--scan_path', help='the path of scans', type=str,
                    default='/media/wjcy/AICC-2T-1/wnb')
parser.add_argument('-save_path', '--save_path', help='the path for saving result', type=str,
                    default='/home/wjcy/data/tf_ggn_invasive_db/')

import xlrd


def read_annotations(anno_file):
    scans = []
    workbook = xlrd.open_workbook(anno_file)
    worksheet = workbook.sheet_by_index(0)
    for row in range(1, worksheet.nrows):
        scan_id = str(worksheet.cell(row, 0).value).split('.')[0]
        cls_id = int(worksheet.cell(row, 1).value)
        if not scan_id is None and len(scan_id) > 0:
            scans.append([scan_id, cls_id])

    return scans


import csv


def read_bboxes(bbox_file, org, spacing):
    roi_ctrs = []
    roi_sizes = []
    hit_id = 0
    with open(bbox_file, 'r') as f:
        csvr = csv.reader(f)
        for row in csvr:
            if 'point|' in row[0]:
                hit_id += 1
                itms = row[0].split('|')
                if hit_id % 2 != 0:
                    roi_ctrs.append([-float(itms[1]), -float(itms[2]), float(itms[3])])
                else:
                    roi_sizes.append([float(itms[1]), float(itms[2]), float(itms[3])])

    bboxes = []
    if len(roi_ctrs) == len(roi_sizes):
        for i in range(len(roi_ctrs)):
            x, y, z = np.int32(np.absolute(np.asanyarray(roi_ctrs[i]) - org) / spacing)
            w, h, d = roi_sizes[i] / spacing * 2.1
            # D = 0.5 * (w + h)
            bboxes.append([x, y, z, w, h, d])
    return bboxes


def scale2newspacing(candidates, org_spacing, new_spacing):
    rsts = []

    scales = np.array(org_spacing) / np.array(new_spacing)
    scale_x, scale_y, scale_z = scales
    for n in candidates:
        X, Y, Z, w, h, d = n[0:6]

        X *= scale_x
        Y *= scale_y
        Z *= scale_z
        # D *= scale_x
        w *= scale_x
        h *= scale_y
        d *= scale_z

        item = [X, Y, Z, w, h, d]
        item.extend(n[6:])
        rsts.append(item)
    return rsts


import os
import os.path as osp
import SimpleITK as sitk
import numpy as np
import glob
from data_processor import extractCube, interpolate_volume
from scipy import ndimage
from skimage import measure
import operator


def hu2gray(volume, WL=-600, WW=900):
    '''
    convert HU value to gray scale[0,255] using lung-window(WL/WW=-500/1200)
    '''
    low = WL - 0.5 * WW
    volume = (volume - low) / WW * 255.0
    volume[volume > 255] = 255
    volume[volume < 0] = 0
    volume = np.uint8(volume)
    return volume


def get_body_bbox(volume):
    body = hu2gray(volume.copy(), WL=-500, WW=800)
    body[body >= 50] = 255
    body[body < 50] = 0
    kernel = np.ones((1, 7, 7), np.uint8)
    body = ndimage.morphology.binary_erosion(body, structure=kernel, iterations=1)

    body_labels = measure.label(body, connectivity=2)
    body_props = measure.regionprops(body_labels)
    if len(body_props) > 0:
        body_prop = sorted(body_props, key=operator.itemgetter('area'), reverse=True)[0]
        z1, y1, x1, z2, y2, x2 = body_prop.bbox
        w = x2 - x1
        h = y2 - y1
        extent_x = 50
        extent_y = 80
        return [x1 + extent_x, y1 + extent_y, z1 + 10, x2 - extent_x, y2 - extent_y, z2 - 10]
    else:
        return [0, 0, 0, 0, 0, 0]


if __name__ == '__main__':
    args = parser.parse_args()
    dst = args.save_path
    '''if osp.exists(dst):
        shutil.rmtree(dst)'''
    if not osp.exists(dst):
        os.mkdir(dst)

    scans = read_annotations(args.anno_file)
    total = len(scans)
    done = 0
    all_bboxes = []
    for scan in scans:
        print(done, total, scan[0])
        done += 1
        scan_id, cls_id = scan

        scan_files = glob.glob(osp.join(args.scan_path, scan_id, '*.nii'))
        if len(scan_files) < 1:
            print('no scan found at:', osp.join(args.scan_path, scan_id))
            continue

        bbox_files = glob.glob(osp.join(args.scan_path, scan_id, '*.acsv'))
        if len(bbox_files) < 1:
            print('no bbox file found at:', osp.join(args.scan_path, scan_id))
            continue

        scan_file = scan_files[0]

        try:
            im = sitk.ReadImage(scan_file)
            volume = sitk.GetArrayFromImage(im)
            org = np.array(im.GetOrigin())
            spacing = np.array(im.GetSpacing())

            new_spacing = [1.0, 1.0, 1.0]
            new_volume = interpolate_volume(volume, spacing, new_spacing)

        except:
            continue

        bbox_file = bbox_files[0]
        bboxes = read_bboxes(bbox_file, org, spacing)
        bboxes = scale2newspacing(bboxes, spacing, new_spacing)
        ignore_mask = np.zeros_like(new_volume)
        for bbox in bboxes:

            X, Y, Z, D = bbox
            try:
                ignore_mask[int(Z - 8):int(Z + 8), int(Y - 10):int(Y + 10), int(X - 10):int(X + 10)] = 1
            except:
                print(ignore_mask.shape, Z, Y, X)
                print('outrange')

            crop_d_mm = D * 2 * new_spacing[0]
            cube = extractCube(new_volume, [X, Y, Z], crop_d_mm, new_spacing, -1)
            save_dir = osp.join(dst, str(cls_id))
            if not osp.exists(save_dir):
                os.mkdir(save_dir)

            np.save(osp.join(save_dir,
                             '{0}_{1}_{2}_{3}_{4}_{5}.npy'.format(scan_id, int(X), int(Y), int(Z), int(D), 'feike')),
                    np.int16(cube))

        # generate negatives
        save_dir = osp.join(dst, '0')
        if not osp.exists(save_dir):
            os.mkdir(save_dir)

        body_bbox = get_body_bbox(new_volume)
        bx1, by1, bz1, bx2, by2, bz2 = body_bbox
        if bx2 == 0 and by2 == 0 and bz2 == 0:
            print('invalid body bbox')
            continue

        body = new_volume[bz1:bz2, by1:by2, bx1:bx2]
        body_mask = ignore_mask[bz1:bz2, by1:by2, bx1:bx2]
        zs, ys, xs = np.where(body < -850)
        select_pxs = np.random.choice([i for i in range(len(zs))], size=15)

        D, H, W = body.shape
        neg_count = 0
        for i in select_pxs:
            z = zs[i]
            y = ys[i]
            x = xs[i]
            if body_mask[z, y, x] == 1:
                continue
            for bbox in bboxes:
                X, Y, Z, D = bbox

                crop_d_mm = D * 2 * new_spacing[0]
                cube = extractCube(body, [x, y, z], crop_d_mm, new_spacing, -1)

                if np.mean(cube) > -500:
                    continue
                np.save(osp.join(save_dir, scan_id + '_' + str(neg_count) + '.npy'), cube)
                neg_count += 1
            '''x1 = int(max(0, x - 0.8*w))
            x2 = int(min(W-1, x + 0.8*w + 1))
            y1 = int(max(0, y - 0.8*h))
            y2 = int(min(H-1, y + 0.8*h +1))
            z1 = int(max(0, z - 0.8*d))
            z2 = int(min(D-1, z + 0.8*d+1))
            all_bboxes.append([x1,y1,z1,x2,y2,z2])
            patch = volume[z1:z2,y1:y2,x1:x2]
            volume_mask[z1:z2,y1:y2,x1:x2] = 0
            save_dir = osp.join(dst,str(cls_id))
            if not osp.exists(save_dir):
                os.mkdir(save_dir)
                
            np.save(osp.join(save_dir,scan_id+'.npy'), patch)'''

        # generate negatives
        '''save_dir = osp.join(dst,'negatives')
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
                
        body_bbox = get_body_bbox(volume)
        bx1,by1,bz1,bx2,by2,bz2 = body_bbox
        if bx2==0 and by2==0 and bz2==0:
            print ('invalid body bbox')
            continue
        
        body = volume[bz1:bz2, by1:by2, bx1:bx2]
        body_mask = volume_mask[bz1:bz2,by1:by2,bx1:bx2]
        zs, ys, xs = np.where(body<-850)
        select_pxs = np.random.choice([i for i in range(len(zs))], size=15)
       
        D, H, W = body.shape
        neg_count= 0
        for i in select_pxs:
            z = zs[i]
            y = ys[i]
            x = xs[i]
            if body_mask[z,y,x] == 0:
                continue
            for bbox in all_bboxes:
                x1,y1,z1,x2,y2,z2 = bbox
                w = 0.5*(x2-x1)
                h = 0.5*(y2-y1)
                d = 0.5*(z2-z1)
                cx1 = int(max(0,x-w))
                cx2 = int(min(W-1, x+w))
                cy1 = int(max(0,y-h))
                cy2 = int(min(H-1, y+h))
                cz1 = int(max(0,z-d))
                cz2 = int(min(D-1, z+d))
                max_l = max(cy2-cy1,cx2-cx1)
                min_l = min(cy2-cy1,cx2-cx1)
                
                if max_l*1.0/min_l>=1.5:
                    continue
                neg_count+=1
                patch = body[cz1:cz2, cy1:cy2, cx1:cx2]
                if np.mean(patch)>-500:
                    continue
                np.save(osp.join(save_dir,scan_id+'_'+ str(neg_count) +'.npy'), patch)'''
