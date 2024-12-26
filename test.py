import argparse
import torch
import os
import numpy as np
from tools import data_utils, uiqm_utils
import skimage.color as sc
import cv2
from models import models
from tqdm import tqdm
# Testing settings

parser = argparse.ArgumentParser(description='Test')
parser.add_argument("--model", type=str, default='SRIDM',
                    help='the model to test')
parser.add_argument("--test_hr_folder", type=str, default='dataset/Test_Dataset/hr/',
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default='dataset/Test_Dataset/lr_2x/',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='results/')
parser.add_argument("--checkpoint", type=str, default='checkpoint/SRIDM/x2/epoch_5.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=2,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_hr_folder

filelist = data_utils.get_list(filepath, ext='.jpg')
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))
uiqm_list = np.zeros(len(filelist))

if opt.model == 'SRCNN':
    model = models.SRCNN(upscale=opt.upscale_factor)
elif opt.model == 'DSRCNN':
    model = models.DSRCNN(upscale=opt.upscale_factor)
elif opt.model == 'EDSR':
    model = models.EDSR(upscale=opt.upscale_factor)
elif opt.model == 'RCAN':
    model = models.RCAN(upscale=opt.upscale_factor)
elif opt.model == 'IMDN':
    model = models.IMDN(upscale=opt.upscale_factor)
elif opt.model == 'MSIDN':
   model = models.MSIDN(upscale=opt.upscale_factor)
elif opt.model =='IMDN_CCA':
    model = models.IMDN_CCA(upscale=opt.upscale_factor)
elif opt.model == 'MYNET_CA_SA':
    model = models.MYNET_CA_SA(upscale=opt.upscale_factor.scale)
elif opt.model == 'MYNET':
    model = models.MYNET(upscale=opt.upscale_factor)
elif opt.model == 'IDN':
    model = models.IDN(upscale=opt.upscale_factor)
elif opt.model == 'CARN':
    model = models.CARN(upscale=opt.upscale_factor)
elif opt.model == 'SRIDM':
    model = models.SRIDM(upscale=opt.upscale_factor)
elif opt.model == 'SRIDM_no_SA':
    model = models.SRIDM_no_SA(upscale=opt.upscale_factor)
elif opt.model == 'SRIDM_no_GFF':
    model = models.SRIDM_no_GFF(upscale=opt.upscale_factor)
elif opt.model == 'SRIDM_Base':
    model = models.SRIDM_Base(upscale=opt.upscale_factor)
elif opt.model == 'SRIDM_rate1':
    model = models.SRIDM_rate1(upscale=opt.upscale_factor)
elif opt.model == 'SRIDM_rate2':
    model = models.SRIDM_rate2(upscale=opt.upscale_factor)
elif opt.model == 'SRDRM':
  if opt.upscale_factor == 2:
    model = models.SRDRM_x2(upscale=opt.upscale_factor)
  elif opt.upscale_factor == 4:
    model = models.SRDRM_x4(upscale=opt.upscale_factor)
  else:
    model = models.SRDRM_x8(upscale=opt.upscale_factor)
else:
    print('input model name error')



model_dict = data_utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=True)

i = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
import pandas as pd
result = pd.DataFrame(columns=('img_name', 'psnr', 'ssim', 'uiqm', 'time'))
for imname in tqdm(filelist):
    im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    im_gt = data_utils.modcrop(im_gt, opt.upscale_factor)
    im_l = cv2.imread(opt.test_lr_folder + imname.split('/')[-1], cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB

    if len(im_gt.shape) < 3:# 如果test图像为灰度图；则将其转变成三通道的RGB图像
        im_gt = im_gt[..., np.newaxis]# 新增一个通道轴
        im_gt = np.concatenate([im_gt] * 3, 2)# 在通道轴上进行拼接
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
    if opt.model == 'SRCNN' or opt.model == 'DSRCNN':# Y通道进行测试
        h, w = im_l.shape[0:2]
        im_l = cv2.resize(im_l, (w * opt.upscale_factor, h * opt.upscale_factor),
                          interpolation=cv2.INTER_CUBIC)
        im_l_ycbcr = (sc.rgb2ycbcr(im_l))
        im_l = im_l_ycbcr[:, :, 0]
        im_l = np.expand_dims(im_l, 2)
    im_input = im_l / 255.0# 0-255=>0-1.0
    im_input = np.transpose(im_input, (2, 0, 1))# HxWxC=>CxHxW
    im_input = im_input[np.newaxis, ...]# 由CxHxW变为1xCxHxW,转换成tensor所需的shape
    im_input = torch.from_numpy(im_input).float()# 从numpy.ndarray中创建一个张量；返回的张量与ndarray共享同一内存

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    with torch.no_grad():# 被该语句包裹起来的部分不会被track梯度；执行的计算不会在反向传播中被记录
        start.record()
        out = model(im_input)
        end.record()
        torch.cuda.synchronize()
        time_list[i] = start.elapsed_time(end)  # milliseconds

    out = out.detach()[0]
    out_y = out.mul(255.0).cpu().numpy()
    out_img = data_utils.tensor2np(out)
  
    crop_size = opt.upscale_factor
    cropped_sr_img = data_utils.shave(out_img, crop_size)# 忽略图像边缘scale个像素的影响；即CxHxW=>Cx(H-2*scale)x(W-2*scale)
    cropped_gt_img = data_utils.shave(im_gt, crop_size)
    # 转到Y通道进行psnr;ssim的计算
    if opt.is_y is True:
        if  opt.model == 'SRCNN' or opt.model == 'DSRCNN':
            im_label = data_utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = data_utils.quantize(cropped_sr_img[..., 0])
        else:
            im_label = data_utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = data_utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
    # RGB全通道进行计算
    else:
        if opt.model == 'SRCNN' or opt.model == 'DSRCNN':
            im_label = cropped_gt_img
            im_pre = np.array([out_y[0, ...], im_l_ycbcr[..., 1], im_l_ycbcr[..., 2]])
            im_pre = data_utils.shave(im_pre, crop_size)
            im_pre = data_utils.quantize(sc.ycbcr2rgb(im_pre)*255.0)
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
    psnr_list[i] = data_utils.compute_psnr(im_pre, im_label)
    ssim_list[i] = data_utils.compute_ssim(im_pre, im_label)
    # pd.iloc[i]['psnr']=psnr_list[i]
    # pd.iloc[i]['img_name']=imname
    if opt.model == 'SRCNN' or opt.model == 'DSRCNN':
        out_img = np.array([out_y[0, ...], im_l_ycbcr[..., 1], im_l_ycbcr[..., 2]]).transpose([1, 2, 0])
        out_img = data_utils.quantize(sc.ycbcr2rgb(out_img)*255.0)
    uiqm_list[i] = uiqm_utils.getUIQM(out_img)
    #result = result.append(pd.DataFrame({'img_name':[imname.split('/')[-1]],'psnr':[psnr_list[i]], 'ssim':[ssim_list[i]], 'uiqm':[uiqm_list[i]], 'time':[time_list[i]]}))
    new_row = pd.DataFrame(
        {'img_name': [imname.split('/')[-1]], 'psnr': [psnr_list[i]], 'ssim': [ssim_list[i]], 'uiqm': [uiqm_list[i]],
         'time': [time_list[i]]})
    result = pd.concat([result, new_row], ignore_index=True)
    output_folder = opt.output_folder + '{}/x{}/'.format(opt.model, opt.upscale_factor)
    output_path = output_folder + imname.split('/')[-1]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cv2.imwrite(output_path, out_img[:, :, [2, 1, 0]])
    i += 1
# np.savetxt(output_path + 'result_psnr_per_image.txt', filelist, psnr_list)
result.to_csv(output_folder + 'result.csv')
print("Mean PSNR: {}, SSIM: {}, UIQM: {} TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(uiqm_list), np.mean(time_list)))

