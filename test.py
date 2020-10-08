import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import cv2
from scipy import ndimage
import torch
import json
import imageio

from data_processor import voxel2world_coord, world2voxel_coord, resize, hu2gray, interpolate_volume, generate_gaussian_mask_3d
from make_db import read_annotations, read_bboxes, scale2newspacing
from data_generator import DataGenerator
from centernet3D import Mediastinal_3dcenternet
from config import cfg
from utils import process_results, get_results, froc, draw_results, get_all_results, IOU_3d, draw_pred_results, get_results_torch
from pytorch_model.models import CenterNet3d, CenterLoss, SizeLoss, unet_CT_dsv_3D


datagenerator = DataGenerator(cfg,training=False,  data_root=cfg.TEST_DATA_ROOT2,
                                  annotation_file=cfg.test_anno_file, results_file=None, label_file=None)

datagenerator_test = DataGenerator(cfg, training=False, data_root=cfg.TEST_DATA_ROOT,
                                  annotation_file=cfg.test_anno_file, results_file=None, label_file=None)

device = torch.device('cuda:0')
# model = CenterNet3d(outpooling=True).to(device)
model = unet_CT_dsv_3D(n_classes=1, in_channels=1, is_dsv=False).to(device)
model.load_state_dict(torch.load('./checkpoints/centernet_torch_sumloss/centernet_53_1.55.pth'))

train_res = get_results_torch(datagenerator, model)

fps, tps, _ = froc(train_res)
plt.plot(fps, tps)
plt.show()