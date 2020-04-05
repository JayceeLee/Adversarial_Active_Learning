import os

import argparse
import pickle
import numpy as np
import ntpath
import matplotlib.pyplot as plt

from dataloaders.eye.eyeDataset import OpenEDSDataset_withLabels
from dataloaders.eye.photometric_transform import PhotometricTransform, photometric_transform_config
from utils.model_utils import load_models
from utils import dual_transforms

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F

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
    parser.add_argument("--batch_size", type=int, default=1, help="#data per batch")
    parser.add_argument('--image_size', type=list, default=[224, 224],
                        help='image_size scalar (currently support square images)')

    parser.add_argument('--save_test', action='store_true', default=False,
                        help='flag to save predictions of target')
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

    uncertainty_scores = np.load("uncertainty_scores.npy")

    assert args.checkpoint_seg is not None, "Need trained .pth!"
    model_seg = load_models(
        mode="segmentation",
        device=device,
        args=args,
    )
    # assert args.checkpoint_disc is not None, "Need trained .pth!"
    # model_disc = load_models(
    #     mode="single_discriminator",
    #     device=device,
    #     args=args,
    # )

    class FeatureExtractor(torch.nn.Module):
        def __init__(self, submodule, extracted_layers):
            super(FeatureExtractor, self).__init__()
            self.submodule = submodule
            self.extracted_layers = extracted_layers

        def forward(self, x):
            for name, module in self.submodule._modules.items():
                x = module(x)
                print(name)
                if name in self.extracted_layers:
                    return x['x5']

    exact_list = ["pretrained_net"]
    featExactor = FeatureExtractor(model_seg, exact_list)
    # a = torch.randn(1, 3, 224, 224)
    # a = Variable(a).to(args.device)
    # x = myexactor(a)
    # print(x)

    transforms_shape_target = dual_transforms.Compose(
        [
            dual_transforms.CenterCrop((400, 400)),
            dual_transforms.Scale(args.image_size[0]),
        ]
    )
    photo_transformer = PhotometricTransform(photometric_transform_config)

    traindata = OpenEDSDataset_withLabels(
        root=os.path.join(args.target_root, "train"),
        image_size=args.image_size,
        data_to_train="",
        shape_transforms=transforms_shape_target,
        photo_transforms=photo_transformer,
        train_bool=False,
    )
    train_loader = torch.utils.data.DataLoader(
        traindata,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    trainloader_iter = enumerate(train_loader)
    args.total_iterations = traindata.__len__() // args.batch_size

    model_seg.eval()
    feature_maps = np.empty((25088, traindata.__len__()))

    for i_iter in range(args.total_iterations):
        if i_iter % 1000 == 0:
            print("Processing {} ..........".format(i_iter))

        u_idx = np.where(uncertainty_scores[0]==i_iter)[0]

        _, batch = next(trainloader_iter)

        images, labels = batch
        images = Variable(images).to(args.device)
        # labels = Variable(labels.long()).to(args.device)

        feat = featExactor(images)
        feature_maps[:, u_idx] = feat.view(1,-1)[0].detach().cpu().numpy()

    print("Saving feature maps .......................")
    np.save("feature_maps_UDA.npy", feature_maps)

    A = np.matmul(feature_maps.transpose(), feature_maps)
    D = A.diagonal()
    distance_map = np.power(D, 0.5) * A * np.power(D, -0.5)

    print("Saving distance maps .......................")
    np.save("distance_maps_UDA.npy", distance_map)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
