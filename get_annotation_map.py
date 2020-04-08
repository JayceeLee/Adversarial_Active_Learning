import os

import argparse
import pickle
import numpy as np
import ntpath
import matplotlib.pyplot as plt

from dataloaders.eye.eyeDataset import JointDataset, OpenEDSDataset_withLabels
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

    assert args.checkpoint_seg is not None, "Need trained .pth!"
    model_seg = load_models(
        mode="segmentation",
        device=device,
        args=args,
    )
    assert args.checkpoint_disc is not None, "Need trained .pth!"
    model_disc = load_models(
        mode="single_discriminator",
        device=device,
        args=args,
    )

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
    model_disc.eval()

    confidence_map = []
    confidence_mean_score = []
    confidence_cnt = []
    imagename = []
    imageindex = []

    dst_folder = os.path.join(args.exp_dir, "visualization")
    if not os.path.isdir(dst_folder):
        os.mkdir(dst_folder)

    image_confidence = np.empty((traindata.__len__(), 224, 224, 2))
    # pred_all = np.empty((traindata.__len__(),4,224,224))
    # with open("dataloaders/eye/trinity_top_200.pkl", "rb") as f:
    #     top_200_lists = pickle.load(f)

    with open("dataloaders/eye/top_1_adv.pkl", "rb") as f:
        top_1_lists = pickle.load(f)

    with open("dataloaders/eye/top_2_adv.pkl", "rb") as f:
        top_2_lists = pickle.load(f)

    for i_iter in range(args.total_iterations):
        if i_iter % 1000 == 0:
            print("Processing {} ..........".format(i_iter))

        # if not (traindata.train_data_list[i_iter] in top_200_lists):
        #     continue

        # if traindata.train_data_list[i_iter] in top_1_lists:
        #     continue

        if traindata.train_data_list[i_iter] in top_2_lists:
            continue

        imageindex.append(i_iter)
        imagename.append(traindata.train_data_list[i_iter])
        _, batch = next(trainloader_iter)

        images, labels = batch
        images = Variable(images).to(args.device)
        labels = Variable(labels.long()).to(args.device)

        pred = model_seg(images)
        pred_softmax = F.softmax(pred, dim=1)
        D_out = model_disc(pred_softmax)
        D_out = torch.sigmoid(D_out)
        D_out = D_out[0,0,:,:].detach().cpu().numpy()

        pred = np.argmax(pred.detach().cpu().numpy(), axis=1)[0,:,:]
        # fig = plt.figure()
        # ax = fig.add_subplot(231)
        # ax.imshow(images[0,0,:,:].detach().cpu().numpy(), cmap="gray")
        # ax.set_xticks([])
        # ax.set_yticks([])
        #
        # ax = fig.add_subplot(232)
        # ax.imshow(labels[0, :, :].detach().cpu().numpy())
        # ax.set_xticks([])
        # ax.set_yticks([])
        #
        # ax = fig.add_subplot(233)
        # ax.imshow(pred, cmap="gray")
        # ax.set_xticks([])
        # ax.set_yticks([])
        #
        # ax = fig.add_subplot(234)
        # ax.imshow(D_out, cmap="gray")
        # ax.set_xticks([])
        # ax.set_yticks([])

        D_out_mean = D_out.mean()
        D_out_mean_map = (D_out > D_out_mean) * 1

        # labels = labels[0,:,:].detach().cpu().numpy()
        # semi_ignore_mask = (D_out < D_out_mean)
        # # pseudo_gt = labels.copy()
        # # pseudo_gt[semi_ignore_mask] = 4
        # # pseudo_gt = pseudo_gt.astype(np.uint8)
        # filename = traindata.train_data_list[i_iter].replace("/images/", "/masks/")
        # filename = filename.replace("/train/", "/train_pseudo/")
        # filename = filename.replace(".png", ".npy")
        # np.save(filename, semi_ignore_mask)
        # # print(D_out_mean_map.shape)

        # ax = fig.add_subplot(235)
        # ax.imshow(D_out_mean_map)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # plt.tight_layout()
        # filename = ntpath.basename(traindata.train_data_list[i_iter])
        # filename = os.path.join(dst_folder, filename)
        # plt.savefig(filename)

        # im_filename = traindata.train_data_list[i_iter].replace("/train/", "/train_pseudo/")
        # os.system("cp %s %s" % (traindata.train_data_list[i_iter], im_filename))

        confidence_mean_score.append(D_out_mean)
        confidence_cnt.append(D_out_mean_map.sum())

        ### generate confidence map ###
        # confidence_map.append(D_out_mean)
        # imagename.append(ntpath.basename(traindata.train_data_list[i_iter]))
        # image_confidence[i_iter,:,:,0] = images[0,0,:,:].detach().cpu().numpy()
        # image_confidence[i_iter,:,:,1] = D_out
        # pred_all[i_iter, ...] = pred_softmax[0,...].detach().cpu().numpy()
        ### generate confidence map ###

    # with open("%s/confidence_map_top1_adv.pkl" % (args.exp_dir), "wb") as f:
    #     pickle.dump([imageindex, imagename, confidence_mean_score, confidence_cnt], f)

    with open("%s/confidence_map_top2_adv.pkl" % (args.exp_dir), "wb") as f:
        pickle.dump([imageindex, imagename, confidence_mean_score, confidence_cnt], f)

    # confidence_map = np.array(confidence_map)
    # print(confidence_map.mean(), np.std(confidence_map))
    #
    # print("Saving ...............................")
    # filename = args.exp_dir+"/image_confidence_UDA.npy"
    # np.save(filename, image_confidence)
    # filename = args.exp_dir + "/predsoftmax_UDA.npy"
    # np.save(filename, pred_all)
    #
    # with open("%s/confidence_map_top1_adv.pkl" % (args.exp_dir), "wb") as f:
    #     pickle.dump([imagename, confidence_map], f)
    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # index = np.argsort(confidence_mean_score)[::-1]
    # x_axis = np.arange(len(confidence_mean_score))
    # ax.scatter(x_axis, np.array(confidence_mean_score)[index], s=2)
    # ax.set_title("Mean confidence score")
    # ax = fig.add_subplot(122)
    # index = np.argsort(confidence_cnt)[::-1]
    # ax.imshow(x_axis, np.array(confidence_cnt[index]), s=2)
    # ax.set_title("Confidence cnt")
    # plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
