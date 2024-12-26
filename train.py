import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import models
from data import USR248, val
from tools import data_utils,log_utils,loss_utils
import skimage.color as sc
import random
from collections import OrderedDict
import sys
from tensorboardX import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Training settings
parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--model", default="SRIDM",
                    help="the model to train")
parser.add_argument("--batch_size", type=int, default=8,
                    help="training batch size")
parser.add_argument("--testBatchSize",type=int, default=1,
                    help="testing batch size")
parser.add_argument("--nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=250,# 学习率衰减策略；每250个epoch后进行一次衰减
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,# 学习率衰减为原来的一半
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=False,# 默认为False;加上“--cuda”触发后为True
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,# 从指定的模型参数文件.pth开始继续进行训练；
                    help="path to checkpoint")
parser.add_argument("--start_epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=0,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="dataset/Train_Dataset/",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=1040,# 训练集图像数量
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=5,# 验证集图像数量
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=130)
parser.add_argument("--scale", type=int, default=2,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=80,# 子图像大小；用于训练的sub-image
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,# 用于训练的通道数；3通道即为RGB;单通道即为Y通道
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,# 预训练模型加载路径
                    help="path to pretrained models")
parser.add_argument("--save", type=str, default="checkpoint/",# 保存的模型路径
                    help="the path to save models")
parser.add_argument("--loss", type=str, default='L1')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.jpg')
parser.add_argument("--phase", type=str, default='train')

args = parser.parse_args()
# 写入log文件
log_path = "log/{}/".format(args.model)
if not os.path.exists(log_path):
    os.makedirs(log_path)
sys.stdout = log_utils.Logger(fileN=log_path + "x{}.txt".format(args.scale))

print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

trainset = USR248.usr248(args)
testset = val.DatasetFromFolderVal("dataset/Val_Dataset/hr/",
                                    "dataset/Val_Dataset/lr_{}x/".format(args.scale),
                                   args)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=True)# 一般将num_works设置为cpu的核心数
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print("===> Building models")
args.is_train = True
if args.model == 'SRCNN':
  model = models.SRCNN(upscale=args.scale)
elif args.model == 'DSRCNN':
  model = models.DSRCNN(upscale=args.scale)
elif args.model == 'EDSR':
  model = models.EDSR(upscale=args.scale)
elif args.model == 'RCAN':
  model = models.RCAN(upscale=args.scale)
elif args.model == 'IMDN':
  model = models.IMDN(upscale=args.scale)
elif args.model =='IMDN_CCA':
  model = models.IMDN_CCA(upscale=args.scale)
elif args.model == 'MYNET_CA_SA':
  model = models.MYNET_CA_SA(upscale=args.scale)
elif args.model == 'MYNET_CA_SA_CHUAN':
  model = models.MYNET_CA_SA_CHUAN(upscale=args.scale)
elif args.model == 'SRIDM':
  model = models.SRIDM(upscale=args.scale)
elif args.model == 'SRIDM_no_SA':
  model = models.SRIDM_no_SA(upscale=args.scale)
elif args.model == 'SRIDM_no_GFF':
  model = models.SRIDM_no_GFF(upscale=args.scale)
elif args.model == 'SRIDM_Base':
  model = models.SRIDM_Base(upscale=args.scale)
elif args.model == 'SRIDM_rate1':
  model = models.SRIDM_rate1(upscale=args.scale)
elif args.model == 'SRIDM_rate2':
  model = models.SRIDM_rate2(upscale=args.scale)
elif args.model == 'IDN':
  model = models.IDN(upscale=args.scale)
elif args.model == 'CARN':
  model = models.CARN(upscale=args.scale)
elif args.model == 'MSIDN':
  model = models.MSIDN(upscale=args.scale)
elif args.model == 'SWINSR':
  model = net(upscale=args.scale, in_chans=3, img_size=args.batch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
elif args.model == 'SRDRM':
  if args.scale == 2:
    model = models.SRDRM_x2(upscale=args.scale)
  elif args.scale == 4:
    model = models.SRDRM_x4(upscale=args.scale)
  else:
    model = models.SRDRM_x8(upscale=args.scale)
else:
    print('input model name error')

print("==> Setting loss")
if args.loss=='L1':
    loss = nn.L1Loss()
elif args.loss=='MSE':
    loss = nn.MSELoss()
elif args.loss=='SRDRM_loss':
    loss = loss_utils.SRDRM_gen_loss()
else:
    print('input loss name error')
print("===> Setting GPU")
if cuda:
    model = model.to(device)
    loss = loss.to(device)

if args.pretrained:

    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        new_state_dcit = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        model.load_state_dict(pretrained_dict, strict=True)

    else:
        print("===> no models found at '{}'".format(args.pretrained))

print("===> Setting Optimizer")

optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    data_utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    loss_per_epoch = 0
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        loss_ = loss(sr_tensor, hr_tensor)
        loss_sr = loss_

        loss_sr.backward()
        optimizer.step()
        if iteration % 26 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                             loss_.item()))
        loss_per_epoch = loss_.item()
    writer.add_scalar('loss', loss_per_epoch, global_step=epoch)
stop_training = False
def valid(epoch):
    model.eval()

    avg_psnr, avg_ssim = 0, 0
    psnr_per_epoch = 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor)

        sr_img = data_utils.tensor2np(pre.detach()[0])
        gt_img = data_utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = data_utils.shave(sr_img, crop_size)
        cropped_gt_img = data_utils.shave(gt_img, crop_size)
        if args.isY is True:
            if args.n_colors == 1:
                im_label = data_utils.quantize(cropped_gt_img[:, :, 0])
                im_pre = data_utils.quantize(cropped_sr_img[:, :, 0])
            else:
                im_label = data_utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
                im_pre = data_utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += data_utils.compute_psnr(im_pre, im_label)
        avg_ssim += data_utils.compute_ssim(im_pre, im_label)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))
    psnr_per_epoch = avg_psnr/len(testing_data_loader)
    writer.add_scalar('psnr', psnr_per_epoch, global_step=epoch)
    ssim_per_epoch = avg_ssim / len(testing_data_loader)
    if ssim_per_epoch >= 0.9:
        print("SSIM reached 0.9 or above, stopping training.")
        # 这里假设 stop_training 是一个全局变量，需要在调用 valid 函数的地方进行相应的定义和处理
        global stop_training
        stop_training = True

def save_checkpoint(epoch):
    model_folder = args.save + "{}/x{}/".format(args.model, args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


print("===> Training")
print_network(model)
writer = SummaryWriter('run/{}'.format(args.model))
for epoch in range(args.start_epoch, args.start_epoch + args.nEpochs):
    valid(epoch - 1)
    if stop_training:
        break
    train(epoch)
    # if epoch >= 900:
    if epoch >= 1:
      save_checkpoint(epoch)
writer.close()


