import os

import argparse
import numpy as np
import pickle
import itertools

from dataloaders.eye.eyeDataset import OpenEDSDataset_withLabels, EverestDataset
from utils.metrics import compute_mean_iou
from utils.model_utils import load_models
from utils.utils import make_logger, adjust_learning_rate
from utils import dual_transforms

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

RES_DIR = "/home/yirus/Projects/Adversarial_Active_Learning/results/"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial Active Learning")
    parser.add_argument('name', type=str,
                        help="Name of the model for storing and loading purposes.")
    parser.add_argument("--source_root", type=str,
                        default="/home/yirus/Datasets/Active_Learning/everest",
                        help="data directory of Source dataset", )
    parser.add_argument("--target_root", type=str,
                    default="/home/yirus/Datasets/Active_Learning/SS_Data_Cropped250x400",
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
    parser.add_argument("--epoch_to_eval", type=int, default=4, help="#epochs to evaluate")

    parser.add_argument('--image_size', type=list, default=[224, 224],
                        help='image_size scalar (currently support square images)')
    parser.add_argument('--pool_size', type=int, default=0,
                        help='buffer size for discriminator')
    parser.add_argument('--lambda_seg_source', type=float, default=1.0,
                        help='hyperparams for seg source')
    parser.add_argument('--lambda_seg_target', type=float, default=2.0,
                        help='hyperparams for seg target')
    parser.add_argument('--lambda_adv', type=float, default=0.001,
                        help='hyperparams for adv of target')

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
    args.exp_dir = os.path.join(RES_DIR, args.name)
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)
    print("EXP PATH: {}".format(args.exp_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    logger = make_logger(filename="TrainVal.log", args=args)
    if args.tensorboard:
        writer = SummaryWriter(args.exp_dir)
    else:
        writer = None

    print("===================================")
    print("====== Loading Training Data ======")
    print("===================================")
    transforms_target = dual_transforms.Compose(
        [
            dual_transforms.CenterCrop((400,400)),
            dual_transforms.Scale(args.image_size[0]),
        ]
    )
    train_target_data = EverestDataset(
        root=args.source_root,
        image_size=args.image_size,
        transforms=transforms_target,
        train_bool=False,
    )

    args.tot_source = train_target_data.__len__()
    args.total_iterations = args.num_epochs * args.tot_source // args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_target_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    model_seg = load_models(
        mode="segmentation",
        device=device,
        args=args,
    )
    optimizer_seg = optim.Adam(
        model_seg.parameters(),
        lr=args.lr_seg,
        betas=(args.beta1, 0.999),
    )
    optimizer_seg.zero_grad()

    seg_loss_target = torch.nn.CrossEntropyLoss().to(device)

    trainloader_iter = enumerate(train_loader)

    loss_seg_min = float("inf")
    miou_max = float("-inf")

    for i_iter in range(args.total_iterations):
        loss_seg_value = 0
        model_seg.train()
        optimizer_seg.zero_grad()

        adjust_learning_rate(
            optimizer=optimizer_seg,
            learning_rate=args.lr_seg,
            i_iter=i_iter,
            max_steps=args.total_iterations,
            power=0.9,
        )

        try:
            _, batch = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = enumerate(train_loader)
            _, batch = next(trainloader_iter)

        images, labels = batch
        images = Variable(images).to(args.device)
        labels = Variable(labels.long()).to(args.device)

        pred = model_seg(images)
        loss_seg = seg_loss_target(pred, labels)

        current_loss_seg = loss_seg.item()
        loss_seg_value += current_loss_seg

        loss_seg.backward()
        optimizer_seg.step()

        pred_img = pred.argmax(dim=1, keepdim=True)
        flat_pred = pred_img.detach().cpu().numpy().flatten()
        flat_gt = labels.detach().cpu().numpy().flatten()
        miou, _ = compute_mean_iou(flat_pred=flat_pred, flat_label=flat_gt)

        logger.info('iter = {0:8d}/{1:8d} '
                    'loss_seg = {2:.3f} '
                    'mIoU = {3:.3f} '.format(
            i_iter, args.total_iterations,
            loss_seg_value,
            miou,)
        )

        if args.tensorboard and (writer != None):
            writer.add_scalar('Train/Cross_Entropy',
                              current_loss_seg,
                              i_iter)
            writer.add_scalar('Train/mIoU',
                              miou,
                              i_iter)

        is_better_ss = current_loss_seg < loss_seg_min
        if is_better_ss:
            loss_seg_min = current_loss_seg
            torch.save(model_seg.state_dict(),
                os.path.join(args.exp_dir, "model_train_best.pth")
            )
        if miou > miou_max:
            miou_max = miou
            torch.save(model_seg.state_dict(),
                       os.path.join(args.exp_dir, "model_train_best_miou.pth")
                       )

    logger.info("==========================================")
    logger.info("Training DONE!")

    if args.tensorboard and (writer != None):
        writer.close()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
