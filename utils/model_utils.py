import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)

from models.discriminator import init_net
from models.discriminator import ScalarGAN, PatchGAN, PixelGAN
# from models.deeplabv3 import deeplab
from models.FCN import FCN8s, VGGNet
from models.deeplab import Res_Deeplab

seed = 1
os.environ['PYTHONHASHSEED']=str(seed)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed)
# 4. Set `pytorch` pseudo-random generator at a fixed value
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def load_models(mode, device, args):
    """

    :param mode: "SS" or "Discriminator"
    :param args:
    :return:
    """

    if mode == "segmentation":
        # model = deeplab.DeepLab(
        #     num_classes=args.nclass,
        #     backbone=args.backbone,
        #     output_stride=args.out_stride,
        #     sync_bn=args.sync_bn,
        #     freeze_bn=args.freeze_bn
        # )
        vgg_model = VGGNet(requires_grad=True)
        model = FCN8s(pretrained_net=vgg_model, n_class=args.nclass)
        # model = Res_Deeplab(num_classes=args.nclass)
        seed = 1
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model = init_net(model, device, init_type="kaiming")
        try:
            if args.checkpoint_seg:
                print('===============================')
                print("Loading pretrained SS model ...")
                print('===============================')
                model.load_state_dict(torch.load(args.checkpoint_seg))

        except Exception as e:
            print(e)
            sys.exit(0)

        model = model.to(device)

        return model
    elif mode == "discriminator":
        disc_scalar = ScalarGAN(
            num_classes=args.nclass,
        )
        seed = 1
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        disc_scalar = init_net(disc_scalar, device, init_type="xavier")
        disc_patch = PatchGAN(
            num_classes=args.nclass,
        )
        seed = 1
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        disc_patch = init_net(disc_patch, device, init_type="xavier")
        disc_pixel = PixelGAN(
            num_classes=args.nclass,
            input_size=args.image_size,
        )
        seed = 1
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        disc_pixel = init_net(disc_pixel, device, init_type="xavier")
    elif mode == "single_discriminator":
        disc_pixel = PixelGAN(
            num_classes=args.nclass,
            input_size=args.image_size,
        )
        seed = 1
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        disc_pixel = init_net(disc_pixel, device, init_type="xavier")
        return disc_pixel
    else:
        raise ValueError("Invalid mode {}!".format(mode))

    disc_scalar, disc_patch, disc_pixel = disc_scalar.to(device), disc_patch.to(device), disc_pixel.to(device)
    return disc_scalar, disc_patch, disc_pixel

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate_SS(optimizer, learning_rate, i_iter, max_steps, power):
    lr = lr_poly(learning_rate, i_iter, max_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, learning_rate, i_iter, max_steps, power):
    lr = lr_poly(learning_rate, i_iter, max_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10