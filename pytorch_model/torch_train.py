import torch
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import os
import glob
import time
from torch.utils.data import Dataset, DataLoader

from config import cfg
from data_generator import DataGenerator
from pytorch_model.models import CenterNet3d, CenterLoss, SizeLoss, unet_CT_dsv_3D
from data_processor import generate_gaussian_mask_3d, hu2gray

class myDataset(Dataset):
    def __init__(self, cfg, training=True):
        self.annotation_file = cfg.anno_file
        self.data_root_dir = cfg.DATA_ROOT
        self.cfg = cfg
        print('loading annotations from ', self.annotation_file)
        anno = pd.read_csv(self.annotation_file, index_col='name')
        if training:
            self.anno = anno[anno['train'] == 1]
        else:
            self.anno = anno[anno['train'] == 0]

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        box = self.anno.iloc[idx]
        name = '0' + str(box.name)
        img = np.load(os.path.join(self.data_root_dir, name, name + '_192.npy'))
        img = hu2gray(img, WL=40, WW=350).astype(np.float32) / 255.0
        x, y, z, w, h, d = np.array(box[:6], dtype=np.int)
        cnt_gt = generate_gaussian_mask_3d(img.shape, x, y, z, w, h, d)
        sze_gt = np.zeros((3, img.shape[0], img.shape[1], img.shape[2]))
        sze_gt[:, x, y, z] = (w, h, d)
        img = np.expand_dims(img, axis=0)
        cnt_gt = np.expand_dims(cnt_gt, axis=0)
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

datagenerator = DataGenerator(cfg, training=True, mode='detect', data_root=cfg.DATA_ROOT,
                              annotation_file=cfg.train_anno_file, results_file=None, label_file=None)
# test_datagenerator = DataGenerator(cfg, training=False)
test_datagenerator = DataGenerator(cfg, training=False, mode='detect', data_root=cfg.TEST_DATA_ROOT,
                                   annotation_file=cfg.test_anno_file, results_file=None, label_file=None)


device = torch.device('cuda:0')
# model = CenterNet3d(outpooling=True).to(device)
model = unet_CT_dsv_3D(n_classes=1, in_channels=1, is_dsv=False).to(device)
loss1 = CenterLoss().to(device)
loss2 = SizeLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
model_dir = os.path.join(cfg.CHECKPOINTS_ROOT, 'centernet_torch_sumloss1')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

load_model = True
if load_model:
    model_weight = find_last(model_dir)
    if model_weight == '' or not os.path.exists(model_weight):
        print('no weights load')
    else:
        model.load_state_dict(torch.load(model_weight))
        print('load model from ', model_weight)

BATCH_SIZE = 4
EPOCH = 200
steps = len(datagenerator.train_list) // BATCH_SIZE
min_cnt_loss = 1000
summaryWriter = SummaryWriter(logdir='./pytorch_model/output_pooing')
for epoch in range(1, EPOCH+1):
    print('---training---')
    model.train()
    train_cnt_loss = []
    train_sze_loss = []
    test_cnt_loss = []
    test_sze_loss = []
    # for i in range(steps):
    for i in range(steps):
    # for i, (ims, cnt_gt, sze_gt) in enumerate(train_data_loader):
        # t1 = time.time()
        # ims, cnt_gt, sze_gt = data.next_batch(BATCH_SIZE, channel_first=True)
        # t2 = time.time()
        # print(ims.shape, cnt_gt.shape, sze_gt.shape)
        # ims = torch.from_numpy(ims).to(device)
        # cnt_gt = torch.from_numpy(cnt_gt).to(device)
        # sze_gt = torch.from_numpy(sze_gt).to(device)
        ims, cnt_gt, sze_gt = datagenerator.next_batch(BATCH_SIZE, channel_first=True)
        ims = torch.from_numpy(ims).to(device)
        cnt_gt = torch.from_numpy(cnt_gt).to(device)
        sze_gt = torch.from_numpy(sze_gt).to(device)
        cnt_pred, sze_pred = model(ims)
        optimizer.zero_grad()
        cnt_loss = loss1(cnt_pred, cnt_gt)
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
            log('epoch: {}/{}, setps: {}/{}, train center loss: {:.6f}, train size loss: {:.6f}, lr: {:.6f}'.format(epoch, EPOCH,
                                    i + 1, steps, np.mean(train_cnt_loss), np.mean(train_sze_loss), optimizer.param_groups[0]['lr']))

    model.eval()
    print('---eval---')
    with torch.no_grad():
        for i in range(len(test_datagenerator.train_list)):
        # for i, (ims, cnt_gt, sze_gt) in enumerate(test_data_loader):
            # t1 = time.time()
            # ims, cnt_gt, sze_gt = data.next_batch(BATCH_SIZE, channel_first=True)
            # t2 = time.time()
            # print(ims.shape, cnt_gt.shape, sze_gt.shape)
            # ims = torch.from_numpy(ims).to(device)
            # cnt_gt = torch.from_numpy(cnt_gt).to(device)
            # sze_gt = torch.from_numpy(sze_gt).to(device)
            ims, cnt_gt, sze_gt = test_datagenerator.next_batch(1, channel_first=True)
            ims = torch.from_numpy(ims).to(device)
            cnt_gt = torch.from_numpy(cnt_gt).to(device)
            sze_gt = torch.from_numpy(sze_gt).to(device)
        # for i in range(5):
        #     ims, cnt_gt, sze_gt = data.next_test_batch(BATCH_SIZE, channel_first=True)
        #     # print(ims.shape, cnt_gt.shape, sze_gt.shape)
        #     ims = torch.from_numpy(ims).to(device)
        #     cnt_gt = torch.from_numpy(cnt_gt).to(device)
        #     sze_gt = torch.from_numpy(sze_gt).to(device)
            cnt_pred, sze_pred = model(ims)
            cnt_loss = loss1(cnt_pred, cnt_gt)
            sze_loss = loss2(sze_pred, sze_gt)
            test_cnt_loss.append(cnt_loss.detach().cpu().item())
            test_sze_loss.append(sze_loss.detach().cpu().item())
            # print(cnt_loss.detach().cpu().item(), sze_loss.detach().cpu().item())
    mean_test_cnt_loss = np.mean(test_cnt_loss)
    mean_test_sze_loss = np.mean(test_sze_loss)
    log('epoch: {}/{}, train center loss: {:.6f}, train size loss: {:.6f}, test center loss: {:.6f}, test size loss: {:.6f}'.format(
        epoch, EPOCH, np.mean(train_cnt_loss), np.mean(train_sze_loss), mean_test_cnt_loss, mean_test_sze_loss))
    summaryWriter.add_scalar('train center loss', np.mean(train_cnt_loss), epoch + 1)
    summaryWriter.add_scalar('train size loss', np.mean(train_sze_loss), epoch + 1)
    summaryWriter.add_scalar('test center loss', mean_test_cnt_loss, epoch + 1)
    summaryWriter.add_scalar('test size loss', mean_test_sze_loss, epoch + 1)
    if mean_test_cnt_loss < min_cnt_loss:
        model_name = os.path.join(model_dir, 'centernet_{}_{:.2f}.pth'.format(epoch, mean_test_cnt_loss))
        torch.save(model.state_dict(), model_name)
        print('save model in ', model_name)
        min_cnt_loss = mean_test_cnt_loss