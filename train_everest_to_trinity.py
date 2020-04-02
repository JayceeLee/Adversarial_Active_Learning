import os

import argparse
import numpy as np
import pickle
import itertools

from dataloaders.eye.eyeDataset import OpenEDSDataset_withoutLabels, OpenEDSDataset_withLabels, EverestDataset
from dataloaders.eye.photometric_transform import PhotometricTransform, photometric_transform_config

from utils.trainer import run_training, run_testing
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
    parser.add_argument("--epoch_to_eval", type=int, default=4, help="#epochs to evaluate")
    parser.add_argument("--epoch_to_eval_source", type=int, default=40, help="#epochs to evaluate")

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

    transforms_shape_source = dual_transforms.Compose(
        [
            dual_transforms.CenterCrop((400,400)),
            dual_transforms.Scale(args.image_size[0]),
        ]
    )

    transforms_shape_target = dual_transforms.Compose(
        [
            dual_transforms.CenterCrop((400,400)),
            dual_transforms.Scale(args.image_size[0]),
        ]
    )

    photo_transformer = PhotometricTransform(photometric_transform_config)

    source_data = EverestDataset(
        root=args.source_root,
        image_size=args.image_size,
        shape_transforms=transforms_shape_source,
        photo_transforms=photo_transformer,
        train_bool=False,
    )

    target_data = OpenEDSDataset_withoutLabels(
        root=os.path.join(args.target_root, "train"),
        image_size=args.image_size,
        shape_transforms=transforms_shape_target,
        photo_transforms=photo_transformer,
    )

    args.tot_source = source_data.__len__()
    args.total_iterations = args.num_epochs * source_data.__len__() // args.batch_size
    args.iters_to_eval = args.epoch_to_eval * source_data.__len__() // args.batch_size
    args.iter_source_to_eval = args.epoch_to_eval_source * source_data.__len__() // args.batch_size

    print("===================================")
    print("========= Loading Val Data ========")
    print("===================================")
    val_target_data = OpenEDSDataset_withLabels(
        root=os.path.join(args.target_root, "validation"),
        image_size=args.image_size,
        data_to_train="",
        shape_transforms=transforms_shape_target,
        photo_transforms=None,
        train_bool=False,
    )
    # class_weight_source = 1.0 / source_data.get_class_probability().to(device)

    source_loader = torch.utils.data.DataLoader(
        source_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    target_loader = torch.utils.data.DataLoader(
        target_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_target_data,
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
    disc_scalar, disc_patch, disc_pixel = load_models(
        mode="discriminator",
        device=device,
        args=args,
    )

    optimizer_seg = optim.Adam(
        model_seg.parameters(),
        lr=args.lr_seg,
        betas=(args.beta1, 0.999),
    )
    optimizer_seg.zero_grad()

    optimizer_disc = torch.optim.Adam(
        itertools.chain(
            disc_scalar.parameters(),
            disc_patch.parameters(),
            disc_pixel.parameters(),
        ),
        lr=args.lr_disc,
        betas=(args.beta1, 0.999),
    )
    optimizer_disc.zero_grad()

    seg_loss_source = torch.nn.CrossEntropyLoss().to(device)
    gan_loss = torch.nn.BCEWithLogitsLoss().to(device)

    history_true_mask = ImagePool(args.pool_size)
    history_fake_mask = ImagePool(args.pool_size)

    trainloader_iter = enumerate(source_loader)
    targetloader_iter = enumerate(target_loader)

    val_loss, val_miou = run_training(
        trainloader_source=source_loader,
        trainloader_target=target_loader,
        trainloader_iter=trainloader_iter,
        targetloader_iter=targetloader_iter,
        val_loader=val_loader,
        model_seg=model_seg,
        disc_scalar=disc_scalar,
        disc_patch=disc_patch,
        disc_pixel=disc_pixel,
        gan_loss=gan_loss,
        seg_loss_source=seg_loss_source,
        optimizer_seg=optimizer_seg,
        optimizer_disc=optimizer_disc,
        history_pool_true=history_true_mask,
        history_pool_fake=history_fake_mask,
        logger=logger,
        writer=writer,
        args=args,
    )

    with open("%s/train_performance.pkl" % args.exp_dir, "wb") as f:
        pickle.dump([val_loss, val_miou], f)

    logger.info("==========================================")
    logger.info("Evaluating on test data ...")

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

    pm = run_testing(
        dataset=testdata,
        test_loader=test_loader,
        model=model_seg,
        args=args,
    )

    logger.info('Global Mean Accuracy: {:.3f}'.format(np.array(pm.GA).mean()))
    logger.info('Mean IOU: {:.3f}'.format(np.array(pm.IOU).mean()))
    logger.info('Mean Recall: {:.3f}'.format(np.array(pm.Recall).mean()))
    logger.info('Mean Precision: {:.3f}'.format(np.array(pm.Precision).mean()))
    logger.info('Mean F1: {:.3f}'.format(np.array(pm.F1).mean()))

    IOU_ALL = np.array(pm.Iou_all)
    logger.info("Back: {:.4f}, Sclera: {:.4f}, Iris: {:.4f}, Pupil: {:.4f}".format(
        IOU_ALL[:, 0].mean(),
        IOU_ALL[:, 1].mean(),
        IOU_ALL[:, 2].mean(),
        IOU_ALL[:, 3].mean(),
    ))

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
