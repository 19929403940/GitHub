from models import models
from FLOPs.profile import profile
# 模型的输入图像尺寸
modellist = ['SRCNN', 'DSRCNN', 'EDSR', 'RCAN', 'CARN', 'IMDN', 'SRIDM']
scales = [2, 4, 8]
for scale in scales:
    width = 640 // scale
    height = 480 // scale
    for model in modellist:
        if(model=='SRCNN'):
            model_ = models.SRCNN(upscale=scale)
            channel = 1
        if (model == 'DSRCNN'):
            model_ = models.DSRCNN(upscale=scale)
            channel = 1
        if (model == 'EDSR'):
            model_ = models.EDSR(upscale=scale)
        if (model == 'RCAN'):
            model_ = models.RCAN(upscale=scale)
        if (model == 'CARN'):
            model_ = models.CARN(upscale=scale)
        if (model == 'IMDN'):
            model_ = models.IMDN_CCA(upscale=scale)
        if (model == 'SRIDM'):
            model_ = models.SRIDM(upscale=scale)
        if(model =='SRCNN' or model=='DSRCNN' ):
            flops, params = profile(model_, input_size=(1, 1, 480, 640))# 单通道
        else:
            flops, params = profile(model_, input_size=(1, 3, height, width))# rgb全通道

        print('放大倍数： {},model: {},输入尺寸: {} x {}, flops: {:.10f} GFLOPs, params: {}'.format(scale, model, height, width, flops/(1e9), params))
