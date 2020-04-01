import os
import sys

import numpy as np
import tqdm
import ntpath

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as vutils

# import matplotlib.pyplot as plt
# plt.switch_backend('agg')

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/.." % file_path)

from utils.utils import make_D_label, adjust_learning_rate
from utils.utils import rescale_image, convt_array_to_PIL, loss_calc
from utils.metrics import Performance_Metrics, compute_mean_iou, evaluate_segmentation

INPUT_CHANNELS = 3
NUM_CLASSES = 4

def validate(
    i_iter,
    val_loader,
    model,
    disc_scalar,
    disc_patch,
    disc_pixel,
    epoch,
    args,
    logger,
    writer,
    val_loss,
    val_iou,
):
    '''
    Pytorch model validation module
    :param val_loader: pytorch dataloader
    :param model: segmentation model
    :param criterion: loss function
    :param args: input arguments from __main__
    :param epoch: epoch counter used to display training progress
    :param val_loss: logs loss prior to invocation of train on batch
    '''
    logger.info("================== Validation ==================")

    seg_loss = torch.nn.CrossEntropyLoss().to(args.device)
    model.eval()
    disc_scalar.eval()
    disc_patch.eval()
    disc_pixel.eval()

    interp = torch.nn.Upsample(
        size=(args.image_size[1], args.image_size[0]),
        mode='bilinear',
        align_corners=False,
    )

    loss_f = 0.
    avg_iou = 0.

    random_batch_idx = np.random.randint(0, val_loader.__len__())
    for batch_idx, data in tqdm.tqdm(enumerate(val_loader), total=val_loader.__len__()):
        image, label = data
        image, label = Variable(image).float(), \
                      Variable(label).type(torch.LongTensor)

        image, label = image.to(args.device), label.to(args.device)

        with torch.set_grad_enabled(False):
            prediction = model(image)
            prediction_interp = interp(prediction)

        loss = seg_loss(prediction_interp, label)
        loss_f += loss.item()

        pred_img = prediction_interp.argmax(dim=1, keepdim=True)
        flat_pred = pred_img.detach().cpu().numpy().flatten()
        flat_gt = label.detach().cpu().numpy().flatten()
        miou, _ = compute_mean_iou(flat_pred=flat_pred, flat_label=flat_gt)
        avg_iou += miou

        if args.tensorboard and (writer != None) and (random_batch_idx == batch_idx):
            filename = os.path.join(args.exp_dir, "trainiter_{}.png".format(i_iter))
            pred_img = pred_img.float()
            gen_img = vutils.make_grid(pred_img, padding=2, normalize=True)
            writer.add_image('Val/Predictions', gen_img, i_iter+batch_idx)
            vutils.save_image(gen_img, filename)

    loss_f /= float(val_loader.__len__())
    avg_iou /= float(val_loader.__len__())
    logger.info("Epoch #{}\t Val Loss: {:.3f}\t Val mIoU: {:.3f}".format(
        epoch, loss_f, avg_iou))

    new_val_loss = loss_f
    new_miou = avg_iou

    if new_val_loss < val_loss:
        print(val_loss, '--->', new_val_loss)
        logger.info('saving checkpoint ....')
        torch.save(model.state_dict(), os.path.join(args.exp_dir, "model_val_best_loss.pth"))
        torch.save(disc_scalar.state_dict(), os.path.join(args.exp_dir, "Dscalar_best_loss.pth"))
        torch.save(disc_patch.state_dict(), os.path.join(args.exp_dir, "Dpatch_best_loss.pth"))
        torch.save(disc_pixel.state_dict(), os.path.join(args.exp_dir, "Dpixel_best_loss.pth"))
    if new_miou > val_iou:
        print(val_iou, '--->', new_miou)
        logger.info('saving checkpoint ....')
        torch.save(model.state_dict(), os.path.join(args.exp_dir, "model_val_best_miou.pth"))
        torch.save(disc_scalar.state_dict(), os.path.join(args.exp_dir, "Dscalar_best_miou.pth"))
        torch.save(disc_patch.state_dict(), os.path.join(args.exp_dir, "Dpatch_best_miou.pth"))
        torch.save(disc_pixel.state_dict(), os.path.join(args.exp_dir, "Dpixel_best_miou.pth"))

    return new_val_loss, new_miou


def run_testing(
    dataset,
    test_loader,
    model,
    args,
):
    '''
    Module to run testing on trained model
    :param dataset: dataset for test
    :param test_loader: dataloader for test
    :param model: pretrained pytorch model
    :param: args: input arguments
    '''
    model.eval()

    interp = torch.nn.Upsample(
        size=(args.image_size[1], args.image_size[0]),
        mode='bilinear',
        align_corners=False,
    )

    ## Compute Metrics:
    Global_Accuracy=[]; Class_Accuracy=[]; Precision=[]; Recall=[]; F1=[]; IOU=[];IOU_list=[]; iou_all=[]
    pm_orig = Performance_Metrics(
        Global_Accuracy,
        Class_Accuracy,
        Precision,
        Recall,
        F1,
        IOU,
        IOU_list,
        iou_all,
    )

    for batch_idx, data in tqdm.tqdm(enumerate(test_loader), total=test_loader.__len__()):
        image, label = data
        image, label = Variable(image).float(), \
                      Variable(label).type(torch.LongTensor)

        image, label = image.to(args.device), label.to(args.device)

        with torch.set_grad_enabled(False):
            prediction = model(image)
            prediction_interp = interp(prediction)

        label = label.detach().cpu().numpy()
        pred = np.argmax(prediction_interp.detach().cpu().numpy(), axis=1)

        for idx in np.arange(pred.shape[0]):
            ga, ca, prec, rec, f1, iou, iou_list = evaluate_segmentation(
                pred[idx, :],
                label[idx, :],
                NUM_CLASSES,
            )
            pm_orig.GA.append(ga)
            pm_orig.CA.append(ca)
            pm_orig.Precision.append(prec)
            pm_orig.Recall.append(rec)
            pm_orig.F1.append(f1)
            pm_orig.IOU.append(iou)

        pred = pred[0, :, :]
        pred = rescale_image(pred)
        pred = convt_array_to_PIL(pred)
        filename = ntpath.basename(dataset.img_list[batch_idx])
        filename = os.path.join(args.exp_dir, filename)
        pred.save(filename)

    return pm_orig

def run_training(
    trainloader_source,
    trainloader_target,
    trainloader_iter,
    targetloader_iter,
    val_loader,
    model_seg,
    disc_scalar,
    disc_patch,
    disc_pixel,
    gan_loss,
    seg_loss_source,
    optimizer_seg,
    optimizer_disc,
    history_pool_true,
    history_pool_fake,
    logger,
    writer,
    args,
):
    source_label = 0
    target_label = 1

    val_loss = []
    val_loss_f = float("inf")
    val_miou = []
    val_miou_f = float("-inf")
    loss_seg_min = float("inf")

    interp = torch.nn.Upsample(
        size=(args.image_size[1], args.image_size[0]),
        mode='bilinear',
        align_corners=False,
    )

    for i_iter in range(args.total_iterations):
        loss_seg_source_value = 0
        loss_adv_value = 0
        loss_D_value = 0
        loss_D_scalar_value = 0
        loss_D_patch_value = 0
        loss_D_pixel_value = 0

        model_seg.train()
        disc_scalar.train()
        disc_patch.train()
        disc_pixel.train()

        optimizer_seg.zero_grad()
        optimizer_disc.zero_grad()

        adjust_learning_rate(
            optimizer=optimizer_seg,
            learning_rate=args.lr_seg,
            i_iter=i_iter,
            max_steps=args.total_iterations,
            power=0.9,
        )

        adjust_learning_rate(
            optimizer=optimizer_disc,
            learning_rate=args.lr_disc,
            i_iter=i_iter,
            max_steps=args.total_iterations,
            power=0.9,
        )

        #################################################
        ######### (1) train segmentation network ########
        #################################################
        for param in disc_scalar.parameters():
            param.requires_grad = False
        for param in disc_patch.parameters():
            param.requires_grad = False
        for param in disc_pixel.parameters():
            param.requires_grad = False

        try:
            _, batch = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = enumerate(trainloader_source)
            _, batch = next(trainloader_iter)

        images, labels = batch
        images = Variable(images).to(args.device)
        labels = Variable(labels.long()).to(args.device)

        pred_source = model_seg(images)
        # pred_source_interp = interp(pred_source)
        pred_source_softmax = F.softmax(pred_source, dim=1)
        loss_seg_source = seg_loss_source(pred_source, labels)
        # loss_seg_source = loss_calc(pred_source_interp, labels, args.device)

        current_loss_seg = loss_seg_source.item()
        loss_seg_source_value += current_loss_seg

        try:
            _, batch = next(targetloader_iter)
        except StopIteration:
            targetloader_iter = enumerate(trainloader_target)
            _, batch = next(targetloader_iter)

        images = batch
        images = Variable(images).to(args.device)
        pred_target = model_seg(images)
        # pred_target_interp = interp(pred_target)
        pred_softmax_target = F.softmax(pred_target, dim=1)

        ### adversarial loss ###
        D_out_scalar = disc_scalar(pred_softmax_target)
        disc_label_scalar = make_D_label(
            input=D_out_scalar,
            value=source_label,
            device=args.device,
            random=False,
        )
        D_out_patch = disc_patch(pred_softmax_target)
        disc_label_patch = make_D_label(
            input=D_out_patch,
            value=source_label,
            device=args.device,
            random=False,
        )
        D_out_pixel = disc_pixel(pred_softmax_target)
        disc_label_pixel = make_D_label(
            input=D_out_pixel,
            value=source_label,
            device=args.device,
            random=False,
        )

        loss_adv_scalar = gan_loss(D_out_scalar, disc_label_scalar)
        loss_adv_patch = gan_loss(D_out_patch, disc_label_patch)
        loss_adv_pixel = gan_loss(D_out_pixel, disc_label_pixel)
        loss_adv_value += loss_adv_scalar.item()+loss_adv_patch.item()+loss_adv_pixel.item()

        loss = args.lambda_seg_source * loss_seg_source + \
               args.lambda_adv * loss_adv_pixel + \
               args.lambda_adv * loss_adv_patch + \
               args.lambda_adv * loss_adv_pixel

        loss = loss
        loss.backward()

        #################################################
        ############# (2) train discriminator ###########
        #################################################
        for param in disc_scalar.parameters():
            param.requires_grad = True
        for param in disc_patch.parameters():
            param.requires_grad = True
        for param in disc_pixel.parameters():
            param.requires_grad = True

        ## train with source ##
        pred_source_softmax = pred_source_softmax.detach()
        pool_source = history_pool_true.query(pred_source_softmax)
        D_out_scalar = disc_scalar(pool_source)
        disc_label_scalar = make_D_label(
            input=D_out_scalar,
            value=source_label,
            device=args.device,
            random=True,
        )
        D_out_patch = disc_patch(pool_source)
        disc_label_patch = make_D_label(
            input=D_out_patch,
            value=source_label,
            device=args.device,
            random=True,
        )
        D_out_pixel = disc_pixel(pool_source)
        disc_label_pixel = make_D_label(
            input=D_out_pixel,
            value=source_label,
            device=args.device,
            random=True,
        )

        loss_D_scalar = gan_loss(D_out_scalar, disc_label_scalar)
        loss_D_patch = gan_loss(D_out_patch, disc_label_patch)
        loss_D_pixel = gan_loss(D_out_pixel, disc_label_pixel)

        loss_D_real = (loss_D_scalar+loss_D_patch+loss_D_pixel) * 0.5
        loss_D_real.backward()
        loss_D_value += loss_D_real.item()
        loss_D_scalar_value += 0.5*loss_D_scalar.item()
        loss_D_patch_value += 0.5 * loss_D_patch.item()
        loss_D_pixel_value += 0.5*loss_D_pixel.item()

        ## train with target ##
        pred_softmax_target = pred_softmax_target.detach()
        pool_target = history_pool_fake.query(pred_softmax_target)
        D_out_scalar = disc_scalar(pool_target)
        disc_label_scalar = make_D_label(
            input=D_out_scalar,
            value=target_label,
            device=args.device,
            random=True,
        )
        D_out_patch = disc_patch(pool_target)
        disc_label_patch = make_D_label(
            input=D_out_patch,
            value=target_label,
            device=args.device,
            random=True,
        )
        D_out_pixel = disc_pixel(pool_target)
        disc_label_pixel = make_D_label(
            input=D_out_pixel,
            value=target_label,
            device=args.device,
            random=True,
        )

        loss_D_scalar = gan_loss(D_out_scalar, disc_label_scalar)
        loss_D_patch = gan_loss(D_out_patch, disc_label_patch)
        loss_D_pixel = gan_loss(D_out_pixel, disc_label_pixel)

        loss_D_fake = (loss_D_scalar + loss_D_patch + loss_D_pixel) * 0.5
        loss_D_fake.backward()
        loss_D_value += loss_D_fake.item()
        loss_D_scalar_value += 0.5*loss_D_scalar.item()
        loss_D_patch_value += 0.5*loss_D_patch.item()
        loss_D_pixel_value += 0.5*loss_D_pixel.item()

        optimizer_seg.step()
        optimizer_disc.step()

        logger.info('iter = {0:8d}/{1:8d} '
              'loss_seg_source = {2:.3f} '
              'loss_adv = {3:.3f} '
              'loss_D_all = {4:.3f} '
              'loss_D_scalar = {5:.3f} '
              'loss_D_patch = {6:.3f} '
              'loss_D_pixel = {7:.3f} '.format(
                i_iter, args.total_iterations,
                loss_seg_source_value,
                loss_adv_value,
                loss_D_value,
                loss_D_scalar_value,
                loss_D_patch_value,
                loss_D_pixel_value,
            )
        )

        if args.tensorboard and (writer != None):
            writer.add_scalar('Train/Cross_Entropy_Source',
                              loss_seg_source_value,
                              i_iter)
            writer.add_scalar('Train/Adversarial_loss',
                              loss_adv_value,
                              i_iter)
            writer.add_scalar('Train/Discriminator_all',
                              loss_D_value,
                              i_iter)
            writer.add_scalar('Train/Discriminator_scalar',
                              loss_D_scalar_value,
                              i_iter)
            writer.add_scalar('Train/Discriminator_patch',
                              loss_D_patch_value,
                              i_iter)
            writer.add_scalar('Train/Discriminator_pixel',
                              loss_D_pixel_value,
                              i_iter)

        current_epoch = i_iter * args.batch_size // args.tot_source
        if i_iter % args.iters_to_eval == 0:
            val_loss_f, val_miou_f = validate(
                i_iter=i_iter,
                val_loader=val_loader,
                model=model_seg,
                disc_scalar=disc_scalar,
                disc_patch=disc_patch,
                disc_pixel=disc_pixel,
                epoch=current_epoch,
                args=args,
                logger=logger,
                writer=writer,
                val_loss=val_loss_f,
                val_iou=val_miou_f,
            )

            val_loss.append(val_loss_f)
            val_loss_f = np.min(np.array(val_loss))
            val_miou.append(val_miou_f)
            val_miou_f = np.max(np.array(val_miou))

            if args.tensorboard and (writer != None):
                writer.add_scalar('Val/Cross_Entropy_Target',
                                  val_loss_f,
                                  i_iter)
                writer.add_scalar('Train/mIoU_Target',
                                  val_miou_f,
                                  i_iter)

            is_better_ss = current_loss_seg < loss_seg_min
            if is_better_ss:
                loss_seg_min = current_loss_seg
                torch.save(model_seg.state_dict(),
                    os.path.join(args.exp_dir, "model_train_best.pth")
                )
                torch.save(disc_scalar.state_dict(),
                    os.path.join(args.exp_dir, "Dscalar_train_best_loss.pth")
                )
                torch.save(disc_patch.state_dict(),
                    os.path.join(args.exp_dir, "Dpatch_train_best_loss.pth")
                )
                torch.save(disc_pixel.state_dict(),
                    os.path.join(args.exp_dir, "Dpixel_train_best_loss.pth")
                )

    current_epoch = int(args.total_iterations * args.batch_size / args.total_source)
    val_loss_f, val_miou_f = validate(
        i_iter=1000000000,
        val_loader=val_loader,
        model=model_seg,
        disc_scalar=disc_scalar,
        disc_patch=disc_patch,
        disc_pixel=disc_pixel,
        epoch=current_epoch,
        args=args,
        logger=logger,
        writer=writer,
        val_loss=val_loss_f,
        val_iou=val_miou_f,
    )
    val_loss.append(val_loss_f)
    val_miou.append(val_miou_f)

    logger.info("==========================================")
    logger.info("Training DONE!")

    if args.tensorboard and (writer != None):
        writer.close()

    return val_loss, val_miou

def validate_baseline(
    i_iter,
    val_loader,
    model,
    epoch,
    args,
    logger,
    writer,
    val_loss,
    val_iou,
):
    '''
    Pytorch model validation module
    :param val_loader: pytorch dataloader
    :param model: segmentation model
    :param criterion: loss function
    :param args: input arguments from __main__
    :param epoch: epoch counter used to display training progress
    :param val_loss: logs loss prior to invocation of train on batch
    '''
    logger.info("================== Validation ==================")

    seg_loss = torch.nn.CrossEntropyLoss().to(args.device)
    interp = torch.nn.Upsample(
        size=(args.image_size[1], args.image_size[0]),
        mode='bilinear',
        align_corners=False,
    )
    model.eval()

    loss_f = 0.
    avg_iou = 0.

    random_batch_idx = np.random.randint(0, val_loader.__len__())
    for batch_idx, data in tqdm.tqdm(enumerate(val_loader), total=val_loader.__len__()):
        image, label = data
        image, label = Variable(image).float(), \
                      Variable(label).type(torch.LongTensor)

        image, label = image.to(args.device), label.to(args.device)

        with torch.set_grad_enabled(False):
            prediction = model(image)
            prediction_interp = interp(prediction)

        loss = seg_loss(prediction_interp, label)
        loss_f += loss.item()

        pred_img = prediction_interp.argmax(dim=1, keepdim=True)
        flat_pred = pred_img.detach().cpu().numpy().flatten()
        flat_gt = label.detach().cpu().numpy().flatten()
        miou, _ = compute_mean_iou(flat_pred=flat_pred, flat_label=flat_gt)
        avg_iou += miou

        if args.tensorboard and (writer != None) and (random_batch_idx == batch_idx):
            filename = os.path.join(args.exp_dir, "trainiter_{}.png".format(i_iter))
            pred_img = pred_img.float()
            gen_img = vutils.make_grid(pred_img, padding=2, normalize=True)
            writer.add_image('Val/Predictions', gen_img, i_iter+batch_idx)
            vutils.save_image(gen_img, filename)

    loss_f /= float(val_loader.__len__())
    avg_iou /= float(val_loader.__len__())
    logger.info("Epoch #{}\t Val Loss: {:.3f}\t Val mIoU: {:.3f}".format(
        epoch, loss_f, avg_iou))

    new_val_loss = loss_f
    new_miou = avg_iou

    if new_val_loss < val_loss:
        print(val_loss, '--->', new_val_loss)
        logger.info('saving checkpoint ....')
        torch.save(model.state_dict(), os.path.join(args.exp_dir, "model_val_best_loss.pth"))
    if new_miou > val_iou:
        print(val_iou, '--->', new_miou)
        logger.info('saving checkpoint ....')
        torch.save(model.state_dict(), os.path.join(args.exp_dir, "model_val_best_miou.pth"))

    return new_val_loss, new_miou