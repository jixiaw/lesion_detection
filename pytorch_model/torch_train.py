import torch
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import os
import glob
import time
import json
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from config import cfg
from data_generator import DataGenerator
from pytorch_model.models import CenterNet3d, CenterLoss, SizeLoss, unet_CT_dsv_3D, unet_CT_dsv_3D_FPN, unet_CT_dsv_2D
from data_processor import generate_gaussian_mask_3d, hu2gray

logger = logging.getLogger('train centernet')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)
handler.setLevel(0)
logger.addHandler(handler)

file_handler = logging.FileHandler('centernet.log', mode='a')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class myDataset(Dataset):
    def __init__(self, cfg, cv, training=True):
        self.annotation_file = cfg.crop_anno_file
        self.data_root_dir = cfg.crop_data_root
        self.cfg = cfg
        self.training = training
        print('loading annotations from ', self.annotation_file)
        with open(self.annotation_file, 'r') as f:
            self.anno = json.load(f)
        if training:
            self.name_list = cv['train']
        else:
            self.name_list = cv['val']
        print("find {} images".format(len(self.name_list)))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        box = np.array(self.anno[name]['bbox_gt']).astype(np.int)
        dmin, dmax = self.anno[name]['slice_range']
        img = np.load(os.path.join(self.data_root_dir, name + '.npy'))
        img = img[dmin:dmax]
        img = hu2gray(img, WL=40, WW=350)
        if dmax - dmin < 32:
            img = np.pad(img, ((0, 32 - dmax + dmin), (0, 0), (0, 0)), 'constant')
        img = (img.astype(np.float32) / 255.0 - cfg.mean) / cfg.std
        cnt_gt = generate_gaussian_mask_3d(img.shape, box)
        sze_gt = np.zeros((3, img.shape[0], img.shape[1], img.shape[2]))
        for i in range(len(box)):
            x, y, z, w, h, d = box[i]
            sze_gt[0, x, y, z] = np.minimum(1, w / 32.0)
            sze_gt[1, x, y, z] = np.minimum(1, h / 128.0)
            sze_gt[2, x, y, z] = np.minimum(1, d / 128.0)

        x, y, z = img.shape
        xlen = x - 32
        ylen = (y - 128) // 2
        zlen = z - 128
        if self.training:
            xmin = np.random.randint(xlen+1)
            ymin = (y - 128) // 4 + np.random.randint(ylen+1)
            zmin = np.random.randint(zlen+1)
        else:
            xmin = xlen // 2
            ymin = ylen
            zmin = zlen // 2
        img = img[xmin:xmin+32, ymin:ymin+128, zmin:zmin+128]
        img = np.expand_dims(img, axis=0)
        cnt_gt = cnt_gt[xmin:xmin+32, ymin:ymin+128, zmin:zmin+128]
        cnt_gt = np.expand_dims(cnt_gt, axis=0)
        sze_gt = sze_gt[:, xmin:xmin+32, ymin:ymin+128, zmin:zmin+128]
        img = torch.from_numpy(img)
        cnt_gt = torch.from_numpy(cnt_gt)
        sze_gt = torch.from_numpy(sze_gt)
        return img, cnt_gt, sze_gt

        # self.train_list, self.test_list, self.annotations = self.load_annotations()

    # def load_annotations(self):
    #     self.annos = pd.read_csv(self.annotation_file, index_col='name')
        # annotations = {}
        # file_list = []
        # train_list = []
        # test_list = []
        # for i in annos.index.values:
        #     # filename, x1, y1, x2, y2, gt_label = annos.iloc[i, 0:6]
        #     x, y, z, w, h, d, flag = annos.loc[i]
        #     filename = '0' + str(i)
        #     # if type(filename) == int:
        #     #     filename = '0' + str(filename)
        #     # elif filename[0] != '0':
        #     #     filename = '0' + filename
        #     img_path = osp.join(self.data_root_dir, filename, filename + '.npy')


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


def log(info):
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())), info)


# data = DataGenerator(cfg)
# train_data = myDataset(cfg, training=True)
# test_data = myDataset(cfg, training=False)
# train_data_loader = DataLoader(train_data, batch_size=2, shuffle=True)
# test_data_loader = DataLoader(test_data, batch_size=2, shuffle=False)
def train_crop(cv, fold):
    BATCH_SIZE = 4
    start_epoch = 0
    EPOCH = 150 + start_epoch
    dataset_train = myDataset(cfg, cv[fold], True)
    dataset_test = myDataset(cfg, cv[fold], False)
    dataloader_train = DataLoader(dataset_train, BATCH_SIZE, True)
    dataloader_test = DataLoader(dataset_test, BATCH_SIZE, False)

    device = torch.device('cuda:0')
    model = unet_CT_dsv_3D(n_classes=1, in_channels=1, is_dsv=False).to(device)
    loss1 = CenterLoss().to(device)
    loss2 = SizeLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.9)
    model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, 'centernet_crop', fold)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_model = True
    if load_model:
        model_weight = find_last(model_dir)
        if model_weight == '' or not os.path.exists(model_weight):
            print('no weights load')
            logger.info('no weights load')
        else:
            start_epoch = int(model_weight.split('_')[-2]) + 1
            model.load_state_dict(torch.load(model_weight))
            print('load model from ', model_weight)
            logger.info('load model from ' + model_weight)

    logger.info(fold)
    # steps = 2
    steps = len(dataloader_train)
    min_cnt_loss = 1000
    summaryWriter = SummaryWriter(logdir=model_dir)
    for epoch in range(start_epoch, EPOCH+1):
        print('---training---')
        model.train()
        train_cnt_loss = []
        train_sze_loss = []
        test_cnt_loss = []
        test_sze_loss = []
        # for i in range(steps):
        for i, (ims, cnt_gt, sze_gt) in enumerate(dataloader_train):
            ims = ims.to(device)
            cnt_gt = cnt_gt.to(device)
            sze_gt = sze_gt.to(device)
            cnt_pred, sze_pred = model(ims)
            optimizer.zero_grad()
            cnt_loss = loss1(cnt_pred, cnt_gt) * 0.5
            sze_loss = loss2(sze_pred, sze_gt) * 10
            loss = cnt_loss + sze_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(cnt_loss.detach().cpu().item(), sze_loss.detach().cpu().item())
            # t3 = time.time()
            # print(t2 - t1, t3 - t2)
            train_cnt_loss.append(cnt_loss.detach().cpu().item())
            train_sze_loss.append(sze_loss.detach().cpu().item())
            if i % 10 == 0:
                logger.info('epoch: {}/{}, setps: {}/{}, train center loss: {:.6f}, train size loss: {:.6f}, lr: {:.6f}'.format(epoch, EPOCH,
                                        i + 1, steps, train_cnt_loss[-1], train_sze_loss[-1], optimizer.param_groups[0]['lr']))

        model.eval()
        print('---eval---')
        with torch.no_grad():
            for i, (ims, cnt_gt, sze_gt) in enumerate(dataloader_test):
                ims = ims.to(device)
                cnt_gt = cnt_gt.to(device)
                sze_gt = sze_gt.to(device)
                cnt_pred, sze_pred = model(ims)
                cnt_loss = loss1(cnt_pred, cnt_gt) * 0.5
                sze_loss = loss2(sze_pred, sze_gt) * 10
                test_cnt_loss.append(cnt_loss.detach().cpu().item())
                test_sze_loss.append(sze_loss.detach().cpu().item())
                # print(cnt_loss.detach().cpu().item(), sze_loss.detach().cpu().item())
        mean_test_cnt_loss = np.mean(test_cnt_loss)
        mean_test_sze_loss = np.mean(test_sze_loss)
        logger.info('epoch: {}/{}, train center loss: {:.6f}, train size loss: {:.6f}, test center loss: {:.6f}, test size loss: {:.6f}'.format(
            epoch, EPOCH, np.mean(train_cnt_loss), np.mean(train_sze_loss), mean_test_cnt_loss, mean_test_sze_loss))
        summaryWriter.add_scalar('train center loss', np.mean(train_cnt_loss), epoch + 1)
        summaryWriter.add_scalar('train size loss', np.mean(train_sze_loss), epoch + 1)
        summaryWriter.add_scalar('test center loss', mean_test_cnt_loss, epoch + 1)
        summaryWriter.add_scalar('test size loss', mean_test_sze_loss, epoch + 1)
        if mean_test_cnt_loss < min_cnt_loss:
            model_name = os.path.join(model_dir, 'centernet_{}_{:.2f}.pth'.format(epoch, mean_test_cnt_loss))
            torch.save(model.state_dict(), model_name)
            print('save model in ', model_name)
            logger.info('save model in ' + model_name)
            min_cnt_loss = mean_test_cnt_loss
        if epoch % 10 == 0:
            model_name = os.path.join(model_dir, 'centernet_{}_{:.2f}.pth'.format(epoch, mean_test_cnt_loss))
            torch.save(model.state_dict(), model_name)
            print('save model in ', model_name)
            logger.info('save model in ' + model_name)


def train(cv, fold):
    datagenerator = DataGenerator(cfg, training=True, mode='detect', data_root=cfg.DATA_ROOT,
                                  annotation_file=cfg.anno_file, results_file=None, label_file=None, cross_validation=cv[fold])
    # test_datagenerator = DataGenerator(cfg, training=False)
    test_datagenerator = DataGenerator(cfg, training=False, mode='detect', data_root=cfg.DATA_ROOT,
                                       annotation_file=cfg.anno_file, results_file=None, label_file=None, cross_validation=cv[fold])

    device = torch.device('cuda:0')
    # model = CenterNet3d(outpooling=True).to(device)
    model = unet_CT_dsv_3D(n_classes=1, in_channels=1, is_dsv=False).to(device)
    # model = unet_CT_dsv_3D(n_classes=1, in_channels=1, is_dsv=False, is_merge=True).to(device)
    loss1 = CenterLoss().to(device)
    loss2 = SizeLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.9)
    model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, 'centernet_size_relu', fold)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_model = True
    if load_model:
        model_weight = find_last(model_dir)
        if model_weight == '' or not os.path.exists(model_weight):
            print('no weights load')
            logger.info('no weights load')
        else:
            model.load_state_dict(torch.load(model_weight))
            print('load model from ', model_weight)
            logger.info('load model from ' + model_weight)

    logger.info(fold)
    BATCH_SIZE = 4
    start_epoch = 1
    EPOCH = 100 + start_epoch
    steps = len(datagenerator.train_list) // BATCH_SIZE
    # steps = 2
    min_cnt_loss = 1000
    summaryWriter = SummaryWriter(logdir=model_dir)
    for epoch in range(start_epoch, EPOCH+1):
        print('---training---')
        model.train()
        train_cnt_loss = []
        train_sze_loss = []
        test_cnt_loss = []
        test_sze_loss = []
        # for i in range(steps):
        for i in range(steps):
            ims, cnt_gt, sze_gt = datagenerator.next_batch(BATCH_SIZE, channel_first=True)
            ims = torch.from_numpy(ims).to(device)
            cnt_gt = torch.from_numpy(cnt_gt).to(device)
            sze_gt = torch.from_numpy(sze_gt).to(device)
            cnt_pred, sze_pred = model(ims)
            # pred = model(ims)
            optimizer.zero_grad()
            cnt_loss = loss1(cnt_pred, cnt_gt) * 0.5
            sze_loss = loss2(sze_pred, sze_gt) * 0.1
            loss = cnt_loss + sze_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(cnt_loss.detach().cpu().item(), sze_loss.detach().cpu().item())
            # t3 = time.time()
            # print(t2 - t1, t3 - t2)
            train_cnt_loss.append(cnt_loss.detach().cpu().item())
            train_sze_loss.append(sze_loss.detach().cpu().item())
            if i % 10 == 0:
                logger.info('epoch: {}/{}, setps: {}/{}, train center loss: {:.6f}, train size loss: {:.6f}, lr: {:.6f}'.format(epoch, EPOCH,
                                        i + 1, steps, train_cnt_loss[-1], train_sze_loss[-1], optimizer.param_groups[0]['lr']))

        model.eval()
        print('---eval---')
        with torch.no_grad():
            for i in tqdm(range(len(test_datagenerator.train_list))):
                ims, cnt_gt, sze_gt = test_datagenerator.next_batch(1, channel_first=True)
                ims = torch.from_numpy(ims).to(device)
                cnt_gt = torch.from_numpy(cnt_gt).to(device)
                sze_gt = torch.from_numpy(sze_gt).to(device)
                cnt_pred, sze_pred = model(ims)
                # pred = model(ims)
                cnt_loss = loss1(cnt_pred, cnt_gt) * 0.5
                sze_loss = loss2(sze_pred, sze_gt) * 0.1
                test_cnt_loss.append(cnt_loss.detach().cpu().item())
                test_sze_loss.append(sze_loss.detach().cpu().item())
                # print(cnt_loss.detach().cpu().item(), sze_loss.detach().cpu().item())
        mean_test_cnt_loss = np.mean(test_cnt_loss)
        mean_test_sze_loss = np.mean(test_sze_loss)
        logger.info('epoch: {}/{}, train center loss: {:.6f}, train size loss: {:.6f}, test center loss: {:.6f}, test size loss: {:.6f}'.format(
            epoch, EPOCH, np.mean(train_cnt_loss), np.mean(train_sze_loss), mean_test_cnt_loss, mean_test_sze_loss))
        summaryWriter.add_scalar('train center loss', np.mean(train_cnt_loss), epoch + 1)
        summaryWriter.add_scalar('train size loss', np.mean(train_sze_loss), epoch + 1)
        summaryWriter.add_scalar('test center loss', mean_test_cnt_loss, epoch + 1)
        summaryWriter.add_scalar('test size loss', mean_test_sze_loss, epoch + 1)
        if mean_test_cnt_loss < min_cnt_loss:
            model_name = os.path.join(model_dir, 'centernet_{}_{:.2f}.pth'.format(epoch, mean_test_cnt_loss))
            torch.save(model.state_dict(), model_name)
            print('save model in ', model_name)
            logger.info('save model in ' + model_name)
            min_cnt_loss = mean_test_cnt_loss
        if epoch % 10 == 0:
            model_name = os.path.join(model_dir, 'centernet_{}_{:.2f}.pth'.format(epoch, mean_test_cnt_loss))
            torch.save(model.state_dict(), model_name)
            print('save model in ', model_name)
            logger.info('save model in ' + model_name)


def train_FPN(cv, fold):
    datagenerator = DataGenerator(cfg, training=True, mode='detect', data_root=cfg.DATA_ROOT,
                                  annotation_file=cfg.train_anno_file, results_file=None, label_file=None,
                                  cross_validation=cv[fold])
    # test_datagenerator = DataGenerator(cfg, training=False)
    test_datagenerator = DataGenerator(cfg, training=False, mode='detect', data_root=cfg.DATA_ROOT,
                                       annotation_file=cfg.train_anno_file, results_file=None, label_file=None,
                                       cross_validation=cv[fold])

    device = torch.device('cuda:0')
    # model = CenterNet3d(outpooling=True).to(device)
    model = unet_CT_dsv_3D_FPN(n_classes=1, in_channels=1, is_dsv=False).to(device)
    loss1 = CenterLoss().to(device)
    loss2 = SizeLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.9)
    model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, 'centernet_torch_sumloss3_FPN', fold)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_model = True
    if load_model:
        model_weight = find_last(model_dir)
        if model_weight == '' or not os.path.exists(model_weight):
            print('no weights load')
            logger.info('no weights load')
        else:
            model.load_state_dict(torch.load(model_weight))
            print('load model from ', model_weight)
            logger.info('load model from ' + model_weight)

    logger.info(fold)
    BATCH_SIZE = 2
    start_epoch = 0
    EPOCH = 100 + start_epoch
    steps = len(datagenerator.train_list) // BATCH_SIZE
    # steps = 2
    min_cnt_loss = 1000
    summaryWriter = SummaryWriter(logdir=model_dir)
    for epoch in range(start_epoch, EPOCH+1):
        print('---training---')
        model.train()
        train_cnt_loss = []
        train_sze_loss = []
        test_cnt_loss = []
        test_sze_loss = []
        # for i in range(steps):
        for i in range(steps):
            ims, cnt_gt, sze_gt = datagenerator.next_batch(BATCH_SIZE, channel_first=True)
            ims = torch.from_numpy(ims).to(device)
            cnt_gt = torch.from_numpy(cnt_gt).to(device)
            sze_gt = torch.from_numpy(sze_gt).to(device)
            preds = model(ims)
            cnt_losses = None
            sze_losses = None
            optimizer.zero_grad()
            for predid, (cnt_pred, sze_pred) in enumerate(preds):
                if cnt_losses is None:
                    cnt_losses = loss1(cnt_pred, cnt_gt) * 0.5
                    sze_losses = loss2(sze_pred, sze_gt) * 10
                else:
                    cnt_losses += loss1(cnt_pred, cnt_gt) * 0.5
                    sze_losses += loss2(sze_pred, sze_gt) * 10
            # cnt_losses = torch.Tensor(cnt_losses)
            # sze_losses = torch.Tensor(sze_losses)
            loss = cnt_losses + sze_losses
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(cnt_loss.detach().cpu().item(), sze_loss.detach().cpu().item())
            # t3 = time.time()
            # print(t2 - t1, t3 - t2)
            train_cnt_loss.append(cnt_losses.detach().cpu().item())
            train_sze_loss.append(sze_losses.detach().cpu().item())
            if i % 10 == 0:
                logger.info('epoch: {}/{}, setps: {}/{}, train center loss: {:.6f}, train size loss: {:.6f}, lr: {:.6f}'.format(epoch, EPOCH,
                                        i + 1, steps, train_cnt_loss[-1], train_sze_loss[-1], optimizer.param_groups[0]['lr']))

        model.eval()
        print('---eval---')
        with torch.no_grad():
            for i in tqdm(range(len(test_datagenerator.train_list))):
                ims, cnt_gt, sze_gt = test_datagenerator.next_batch(1, channel_first=True)
                ims = torch.from_numpy(ims).to(device)
                cnt_gt = torch.from_numpy(cnt_gt).to(device)
                sze_gt = torch.from_numpy(sze_gt).to(device)
                preds = model(ims)
                cnt_losses = None
                sze_losses = None
                optimizer.zero_grad()
                for cnt_pred, sze_pred in preds:
                    if cnt_losses is None:
                        cnt_losses = loss1(cnt_pred, cnt_gt) * 0.5
                        sze_losses = loss2(sze_pred, sze_gt) * 10
                    else:
                        cnt_losses += loss1(cnt_pred, cnt_gt) * 0.5
                        sze_losses += loss2(sze_pred, sze_gt) * 10
                test_cnt_loss.append(cnt_losses.detach().cpu().item())
                test_sze_loss.append(sze_losses.detach().cpu().item())
                # print(cnt_loss.detach().cpu().item(), sze_loss.detach().cpu().item())
        mean_test_cnt_loss = np.mean(test_cnt_loss)
        mean_test_sze_loss = np.mean(test_sze_loss)
        logger.info('epoch: {}/{}, train center loss: {:.6f}, train size loss: {:.6f}, test center loss: {:.6f}, test size loss: {:.6f}'.format(
            epoch, EPOCH, np.mean(train_cnt_loss), np.mean(train_sze_loss), mean_test_cnt_loss, mean_test_sze_loss))
        summaryWriter.add_scalar('train center loss', np.mean(train_cnt_loss), epoch + 1)
        summaryWriter.add_scalar('train size loss', np.mean(train_sze_loss), epoch + 1)
        summaryWriter.add_scalar('test center loss', mean_test_cnt_loss, epoch + 1)
        summaryWriter.add_scalar('test size loss', mean_test_sze_loss, epoch + 1)
        if mean_test_cnt_loss < min_cnt_loss:
            model_name = os.path.join(model_dir, 'centernet_{}_{:.2f}.pth'.format(epoch, mean_test_cnt_loss))
            torch.save(model.state_dict(), model_name)
            print('save model in ', model_name)
            logger.info('save model in ' + model_name)
            min_cnt_loss = mean_test_cnt_loss
        if epoch % 10 == 0:
            model_name = os.path.join(model_dir, 'centernet_{}_{:.2f}.pth'.format(epoch, mean_test_cnt_loss))
            torch.save(model.state_dict(), model_name)
            print('save model in ', model_name)
            logger.info('save model in ' + model_name)


def train_2d(cv, fold):
    datagenerator = DataGenerator(cfg, training=True, mode='detect', data_root=cfg.DATA_ROOT,
                                  annotation_file=cfg.anno_file, results_file=None, label_file=None,
                                  cross_validation=cv[fold])
    # test_datagenerator = DataGenerator(cfg, training=False)
    test_datagenerator = DataGenerator(cfg, training=False, mode='detect', data_root=cfg.DATA_ROOT,
                                       annotation_file=cfg.anno_file, results_file=None, label_file=None,
                                       cross_validation=cv[fold])

    device = torch.device('cuda:0')
    model = unet_CT_dsv_2D(n_classes=1, in_channels=3, is_dsv=False).to(device)
    loss1 = CenterLoss().to(device)
    loss2 = SizeLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.9)
    model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, 'centernet_2d', fold)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_model = True
    if load_model:
        model_weight = find_last(model_dir)
        if model_weight == '' or not os.path.exists(model_weight):
            print('no weights load')
            logger.info('no weights load')
        else:
            model.load_state_dict(torch.load(model_weight))
            print('load model from ', model_weight)
            logger.info('load model from ' + model_weight)

    logger.info(fold)
    BATCH_SIZE = 64
    NUM = 4
    start_epoch = 1
    EPOCH = 100 + start_epoch
    steps = len(datagenerator.train_list) // NUM
    # steps = 2
    min_cnt_loss = 10000
    summaryWriter = SummaryWriter(logdir=model_dir)
    for epoch in range(start_epoch, EPOCH + 1):
        print('---training---')
        model.train()
        train_cnt_loss = []
        train_sze_loss = []
        test_cnt_loss = []
        test_sze_loss = []
        # for i in range(steps):
        for i in range(steps):
            ims, cnt_gt, sze_gt = datagenerator.next_batch_2d(BATCH_SIZE, NUM)
            ims = torch.from_numpy(ims).to(device)
            cnt_gt = torch.from_numpy(cnt_gt).to(device)
            sze_gt = torch.from_numpy(sze_gt).to(device)
            cnt_pred, sze_pred = model(ims)
            # pred = model(ims)
            optimizer.zero_grad()
            cnt_loss = loss1(cnt_pred, cnt_gt) * 1
            sze_loss = loss2(sze_pred, sze_gt) * 0.1
            loss = cnt_loss + sze_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(cnt_loss.detach().cpu().item(), sze_loss.detach().cpu().item())
            # t3 = time.time()
            # print(t2 - t1, t3 - t2)
            train_cnt_loss.append(cnt_loss.detach().cpu().item())
            train_sze_loss.append(sze_loss.detach().cpu().item())
            if i % 10 == 0:
                logger.info(
                    'epoch: {}/{}, setps: {}/{}, train center loss: {:.6f}, train size loss: {:.6f}, lr: {:.6f}'.format(
                        epoch, EPOCH,
                        i + 1, steps, train_cnt_loss[-1], train_sze_loss[-1], optimizer.param_groups[0]['lr']))

        model.eval()
        print('---eval---')
        with torch.no_grad():
            for i in tqdm(range(len(test_datagenerator.train_list))):
                ims, cnt_gt, sze_gt = test_datagenerator.next_batch_2d(BATCH_SIZE, 1)
                ims = torch.from_numpy(ims).to(device)
                cnt_gt = torch.from_numpy(cnt_gt).to(device)
                sze_gt = torch.from_numpy(sze_gt).to(device)
                cnt_pred, sze_pred = model(ims)
                # pred = model(ims)
                cnt_loss = loss1(cnt_pred, cnt_gt) * 1
                sze_loss = loss2(sze_pred, sze_gt) * 0.1
                test_cnt_loss.append(cnt_loss.detach().cpu().item())
                test_sze_loss.append(sze_loss.detach().cpu().item())
                # print(cnt_loss.detach().cpu().item(), sze_loss.detach().cpu().item())
        mean_test_cnt_loss = np.mean(test_cnt_loss)
        mean_test_sze_loss = np.mean(test_sze_loss)
        logger.info(
            'epoch: {}/{}, train center loss: {:.6f}, train size loss: {:.6f}, test center loss: {:.6f}, test size loss: {:.6f}'.format(
                epoch, EPOCH, np.mean(train_cnt_loss), np.mean(train_sze_loss), mean_test_cnt_loss, mean_test_sze_loss))
        summaryWriter.add_scalar('train center loss', np.mean(train_cnt_loss), epoch + 1)
        summaryWriter.add_scalar('train size loss', np.mean(train_sze_loss), epoch + 1)
        summaryWriter.add_scalar('test center loss', mean_test_cnt_loss, epoch + 1)
        summaryWriter.add_scalar('test size loss', mean_test_sze_loss, epoch + 1)
        if mean_test_cnt_loss < min_cnt_loss:
            model_name = os.path.join(model_dir, 'centernet_{}_{:.2f}.pth'.format(epoch, mean_test_cnt_loss))
            torch.save(model.state_dict(), model_name)
            print('save model in ', model_name)
            logger.info('save model in ' + model_name)
            min_cnt_loss = mean_test_cnt_loss
        if epoch % 10 == 0:
            model_name = os.path.join(model_dir, 'centernet_{}_{:.2f}.pth'.format(epoch, mean_test_cnt_loss))
            torch.save(model.state_dict(), model_name)
            print('save model in ', model_name)
            logger.info('save model in ' + model_name)


if __name__ == '__main__':
    with open(cfg.cross_validation, 'r') as f:
        cv = json.load(f)
    # fold = 'fold0'
    for fold in cv.keys():
    # for fold in ['fold0', 'fold2', 'fold3', 'fold4']:
        print(fold)
        # train_crop(cv, fold)
        train_2d(cv, fold)