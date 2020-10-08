import torch

from pytorch_model.models import CenterNet3d, unet_CT_dsv_3D

model = unet_CT_dsv_3D(n_classes=1, in_channels=1, is_dsv=True).cuda()

x = torch.rand(size=(1, 1, 128, 96, 128))

cnt_pred, sze_pred = model(x.cuda())
print(cnt_pred.shape, sze_pred.shape)
