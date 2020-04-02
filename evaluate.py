import os

import argparse
import numpy as np
import pickle
import itertools

from dataloaders.eye.eyeDataset import OpenEDSDataset_withoutLabels, OpenEDSDataset_withLabels, EverestDataset
from dataloaders.eye.photometric_transform import PhotometricTransform, photometric_transform_config

from utils.trainer import run_training_SDA, run_testing
from utils.image_pool import ImagePool
from utils.model_utils import load_models
from utils.utils import make_logger
from utils import dual_transforms

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

RES_DIR = "/home/yirus/Projects/Adversarial_Active_Learning/results"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial Active Learning")
    parser.add_argument('name', type=str,
                        help="Name of the model for storing and loading purposes.")
    parser.add_argument("--source_root", type=str,
                    default="/home/yirus/Datasets/Active_Learning/everest",
                      help="data directory of Source dataset",)
    parser.add_argument("--target_root", type=str,
                    default="/home/yirus/Datasets/Active_Learning/trinity",
                      help="data directory of Target dataset",)

    parser.add_argument("--nclass",type=int, default=4, help="#classes")
    parser.add_argument("--lr_seg", type=float, default=0.001, help="lr for SS")
    parser.add_argument("--lr_disc", type=float, default=0.001, help="lr for D")
    parser.add_argument("--workers", type=int, default=0, help="#workers for dataloader")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for Adam")

    parser.add_argument("--num_epochs", type=int, default=200, help="#epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="#data per batch")

    parser.add_argument('--image_size', type=list, default=[224, 224],
                        help='image_size scalar (currently support square images)')

    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='visulization with tensorboard')
    parser.add_argument('--checkpoint_seg', type=str, default=None,
                        help='Pretrained model of segmentation (.pth)')
    parser.add_argument('--checkpoint_disc', type=str, default=None,
                        help='Pretrained model of discriminator (.pth)')
    args = parser.parse_args()
    return args

def main(args):
    print('===================================\n', )
    print("Root directory: {}".format(args.name))
    args.exp_dir = os.path.join(os.path.join(RES_DIR, args.name), "evaluation_trinity_test")
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)
    print("EXP PATH: {}".format(args.exp_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    transforms_shape_target = dual_transforms.Compose(
        [
            dual_transforms.CenterCrop((400, 400)),
            dual_transforms.Scale(args.image_size[0]),
        ]
    )

    testdata = OpenEDSDataset_withLabels(
        root=os.path.join(args.target_root, "test"),
        image_size=args.image_size,
        data_to_train="",
        shape_transforms=transforms_shape_target,
        photo_transforms=None,
        train_bool=False,
    )
    test_loader = torch.utils.data.DataLoader(
        testdata,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    assert args.checkpoint_seg is not None, "Need trained .pth!"
    model_seg = load_models(
        mode="segmentation",
        device=device,
        args=args,
    )

    print("Evaluating ...................")
    miou_all, iou_all = run_testing(
        dataset=testdata,
        test_loader=test_loader,
        model=model_seg,
        args=args,
    )

    # print('Global Mean Accuracy: {:.3f}'.format(np.array(pm.GA).mean()))
    # print('Mean IOU: {:.3f}'.format(np.array(pm.IOU).mean()))
    # print('Mean Recall: {:.3f}'.format(np.array(pm.Recall).mean()))
    # print('Mean Precision: {:.3f}'.format(np.array(pm.Precision).mean()))
    # print('Mean F1: {:.3f}'.format(np.array(pm.F1).mean()))()

    print('Mean IOU: {:.3f}'.format(miou_all.mean()))
    print("Back: {:.4f}, Sclera: {:.4f}, Iris: {:.4f}, Pupil: {:.4f}".format(
        iou_all[:, 0].mean(),
        iou_all[:, 1].mean(),
        iou_all[:, 2].mean(),
        iou_all[:, 3].mean(),
    ))

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
