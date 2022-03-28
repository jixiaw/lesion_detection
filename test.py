import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import torch
import json
import glob

from data_generator import DataGenerator
from classifier import ClassifierTorch
from config import cfg
from utils import froc, get_results_torch
from pytorch_model.models import unet_CT_dsv_3D


def find_last(log_dir):
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Returns:
        The path of the last checkpoint file
    """
    weights_files = glob.glob(os.path.join(log_dir, '*.pth'))
    if len(weights_files) == 0:
        return ''
    weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(x))
    return weights_files[-1]


def test_detect():
    with open(cfg.cross_validation, 'r') as f:
        cv = json.load(f)
    checkpoint = 'pytorch_model/checkpoints/centernet_final_aug'
    out_file = 'results2/detection_aug.json'
    device = torch.device('cuda:0')
    model = unet_CT_dsv_3D(n_classes=1, in_channels=1, is_dsv=False).to(device)
    res = {}
    for fold in cv.keys():
        print(fold)
        datagenerator = DataGenerator(cfg, training=False, data_root=cfg.DATA_ROOT,
                                      annotation_file=cfg.anno_file, results_file=None,
                                      label_file=None, cross_validation=cv[fold])

        model_path = find_last(os.path.join(checkpoint, fold))

        model.load_state_dict(torch.load(model_path))
        temp_res = get_results_torch(datagenerator, model)
        res.update(temp_res)

    with open(out_file, 'w') as f:
        json.dump(res, f)
    fps, tps, _ = froc(res)
    plt.plot(fps, tps)
    plt.show()


def test_cls():
    with open(cfg.cross_validation, 'r') as f:
        cv = json.load(f)
    checkpoint = 'checkpoints/seresnet18_final_cls2_2'

    for fold_id in cv.keys():
        model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, 'seresnet18_final_cls2_2')
        model = ClassifierTorch(cfg.INPUT_SHAPE, model_name='seresnet18', is_training=True, num_classes=2,
                                model_dir=model_dir, config=cfg, fold=fold_id, num_classes2=2)
        model_path = find_last(os.path.join(checkpoint, fold_id))

        model.load_weights(model_path)

        train_datagenerator = DataGenerator(cfg, training=False, mode='cls', data_root=cfg.DATA_ROOT,
                                            annotation_file=cfg.train_anno_file,
                                            results_file='./results2/detection.json',
                                            label_file=cfg.label_file, cross_validation=cv[fold_id])
        _, _, res_dict = test2(train_datagenerator, model, dump=True)
        with open('./results2/%s_fp_se.json' % fold_id, 'w') as f:
            json.dump(res_dict, f)


def test2(datagenerator, model, dump=True):
    batch_size = 16
    res_dict = {}
    res = []
    y_true = []
    model.model.eval()
    with torch.no_grad():
        for name in tqdm(datagenerator.train_list):
            temp = {}
            ims_all, labels_all = datagenerator.get_all_bbox(name, channel_first=True, size=(64, 64, 64))
            y_true.append(labels_all)
            # print(ims.shape)
            preds1, preds2 = [], []
            for i in range(0, ims_all.shape[0], batch_size):
                ims = torch.tensor(ims_all[i:i+batch_size], dtype=torch.float32).cuda()
                labels = torch.tensor(labels_all[i:i+batch_size], dtype=torch.long).cuda()
                pred1, pred2 = model.model(ims)
                pred1 = torch.softmax(pred1, dim=1)
                pred2 = torch.softmax(pred2, dim=1)
                preds1.append(pred1.detach().cpu())
                preds2.append(pred2.detach().cpu())
            res.append((torch.cat(preds1, 0), torch.cat(preds2, 0)))
            if dump:
                temp['y_true'] = np.array(labels_all, dtype=np.int32).tolist()
                temp['y_pred1'] = res[-1][0].numpy().tolist()
                temp['y_pred2'] = res[-1][1].numpy().tolist()
                res_dict[name] = temp
    if dump:
        return y_true, res, res_dict
    else:
        return y_true, res


if __name__ == '__main__':
    test_detect()
    # test_cls()