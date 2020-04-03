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

from utils.utils import make_D_label, adjust_learning_rate, get_mean_threshold
from utils.utils import rescale_image, convt_array_to_PIL
from utils.metrics import Performance_Metrics, compute_mean_iou, evaluate_segmentation

INPUT_CHANNELS = 3
NUM_CLASSES = 4

def get_miou(
    dataloader,
    model,
    global_max_miou,
    args,
):
    model.eval()
    avg_iou = 0.

    for batch_idx, data in tqdm.tqdm(enumerate(dataloader), total=dataloader.__len__()):
        image, label = data
        image, label = Variable(image).float(), \
                      Variable(label).type(torch.LongTensor)

        image, label = image.to(args.device), label.to(args.device)

        with torch.set_grad_enabled(False):
            prediction = model(image)

        pred_img = prediction.argmax(dim=1, keepdim=True)
        flat_pred = pred_img.detach().cpu().numpy().flatten()
        flat_gt = label.detach().cpu().numpy().flatten()
        miou, _ = compute_mean_iou(flat_pred=flat_pred, flat_label=flat_gt)
        avg_iou += miou

    avg_iou /= float(dataloader.__len__())
    if avg_iou > global_max_miou:
        print("--------- TARGET TRAIN -----------")
        print(global_max_miou, '--->', avg_iou)

    return avg_iou

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
    data_type="target",
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

    # interp = torch.nn.Upsample(
    #     size=(args.image_size[1], args.image_size[0]),
    #     mode='bilinear',
    #     align_corners=False,
    # )

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
            # prediction_interp = interp(prediction)

        loss = seg_loss(prediction, label)
        loss_f += loss.item()

        pred_img = prediction.argmax(dim=1, keepdim=True)
        flat_pred = pred_img.detach().cpu().numpy().flatten()
        flat_gt = label.detach().cpu().numpy().flatten()
        miou, _ = compute_mean_iou(flat_pred=flat_pred, flat_label=flat_gt)
        avg_iou += miou

        if args.tensorboard and (writer != None) and \
                (random_batch_idx == batch_idx) and (data_type == "target"):
            filename = os.path.join(args.exp_dir, "Target_pred_trainiter_{}.png".format(i_iter))
            pred_img = pred_img.float()
            gen_img = vutils.make_grid(pred_img, padding=2, normalize=True)
            writer.add_image('Val/Target_Predictions', gen_img, i_iter+batch_idx)
            vutils.save_image(gen_img, filename)

            filename = os.path.join(args.exp_dir, "Target_img_trainiter_{}.png".format(i_iter))
            target_img = image.float()
            gen_target_img = vutils.make_grid(target_img, padding=2, normalize=True)
            vutils.save_image(gen_target_img, filename)

    loss_f /= float(val_loader.__len__())
    avg_iou /= float(val_loader.__len__())

    if data_type == "target":
        logger.info("Epoch #{}\t Target_Val Loss: {:.3f}\t Target_Val mIoU: {:.3f}".format(
            epoch, loss_f, avg_iou))
    elif data_type == "source":
        logger.info("Epoch #{}\t Source_Val Loss: {:.3f}\t Source_Val mIoU: {:.3f}".format(
            epoch, loss_f, avg_iou))

    new_val_loss = loss_f
    new_miou = avg_iou

    if new_val_loss < val_loss and (data_type == "target"):
        print("--------- TARGET VAL -----------")
        print(val_loss, '--->', new_val_loss)
        logger.info('saving checkpoint ....')
        torch.save(model.state_dict(), os.path.join(args.exp_dir, "model_val_best_loss.pth"))
        torch.save(disc_scalar.state_dict(), os.path.join(args.exp_dir, "Dscalar_val_best_loss.pth"))
        torch.save(disc_patch.state_dict(), os.path.join(args.exp_dir, "Dpatch_val_best_loss.pth"))
        torch.save(disc_pixel.state_dict(), os.path.join(args.exp_dir, "Dpixel_val_best_loss.pth"))
        # elif data_type == "source":
        #     torch.save(model.state_dict(), os.path.join(args.exp_dir, "source_model_best.pth"))
        #     torch.save(disc_scalar.state_dict(), os.path.join(args.exp_dir, "source_Dscalar_best.pth"))
        #     torch.save(disc_patch.state_dict(), os.path.join(args.exp_dir, "source_Dpatch_best.pth"))
        #     torch.save(disc_pixel.state_dict(), os.path.join(args.exp_dir, "source_Dpixel_best.pth"))

    if new_miou > val_iou and (data_type == "target"):
        print("--------- TARGET VAL -----------")
        print(val_iou, '--->', new_miou)
        logger.info('saving checkpoint ....')
        torch.save(model.state_dict(), os.path.join(args.exp_dir, "model_val_best_miou.pth"))
        torch.save(disc_scalar.state_dict(), os.path.join(args.exp_dir, "Dscalar_val_best_miou.pth"))
        torch.save(disc_patch.state_dict(), os.path.join(args.exp_dir, "Dpatch_val_best_miou.pth"))
        torch.save(disc_pixel.state_dict(), os.path.join(args.exp_dir, "Dpixel_val_best_miou.pth"))
        # elif data_type == "source":
        #     torch.save(model.state_dict(), os.path.join(args.exp_dir, "source_model_best_miou.pth"))
        #     torch.save(disc_scalar.state_dict(), os.path.join(args.exp_dir, "source_Dscalar_best_miou.pth"))
        #     torch.save(disc_patch.state_dict(), os.path.join(args.exp_dir, "source_Dpatch_best_miou.pth"))
        #     torch.save(disc_pixel.state_dict(), os.path.join(args.exp_dir, "source_Dpixel_best_miou.pth"))

    return new_val_loss, new_miou

def validate_singleD(
    i_iter,
    val_loader,
    model,
    model_disc,
    epoch,
    args,
    logger,
    writer,
    val_loss,
    val_iou,
    data_type="target",
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
    model_disc.eval()

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
            # prediction_interp = interp(prediction)

        loss = seg_loss(prediction, label)
        loss_f += loss.item()

        pred_img = prediction.argmax(dim=1, keepdim=True)
        flat_pred = pred_img.detach().cpu().numpy().flatten()
        flat_gt = label.detach().cpu().numpy().flatten()
        miou, _ = compute_mean_iou(flat_pred=flat_pred, flat_label=flat_gt)
        avg_iou += miou

        if args.tensorboard and (writer != None) and \
                (random_batch_idx == batch_idx) and (data_type == "target"):
            filename = os.path.join(args.exp_dir, "Target_pred_trainiter_{}.png".format(i_iter))
            pred_img = pred_img.float()
            gen_img = vutils.make_grid(pred_img, padding=2, normalize=True)
            writer.add_image('Val/Target_Predictions', gen_img, i_iter+batch_idx)
            vutils.save_image(gen_img, filename)

            filename = os.path.join(args.exp_dir, "Target_img_trainiter_{}.png".format(i_iter))
            target_img = image.float()
            gen_target_img = vutils.make_grid(target_img, padding=2, normalize=True)
            vutils.save_image(gen_target_img, filename)

    loss_f /= float(val_loader.__len__())
    avg_iou /= float(val_loader.__len__())

    if data_type == "target":
        logger.info("Epoch #{}\t Target_Val Loss: {:.3f}\t Target_Val mIoU: {:.3f}".format(
            epoch, loss_f, avg_iou))
    elif data_type == "source":
        logger.info("Epoch #{}\t Source_Val Loss: {:.3f}\t Source_Val mIoU: {:.3f}".format(
            epoch, loss_f, avg_iou))

    new_val_loss = loss_f
    new_miou = avg_iou

    if new_val_loss < val_loss and (data_type == "target"):
        print("--------- TARGET VAL -----------")
        print(val_loss, '--->', new_val_loss)
        logger.info('saving checkpoint ....')
        torch.save(model.state_dict(), os.path.join(args.exp_dir, "model_val_best_loss.pth"))
        torch.save(model_disc.state_dict(), os.path.join(args.exp_dir, "modelD_val_best_loss.pth"))

    if new_miou > val_iou and (data_type == "target"):
        print("--------- TARGET VAL -----------")
        print(val_iou, '--->', new_miou)
        logger.info('saving checkpoint ....')
        torch.save(model.state_dict(), os.path.join(args.exp_dir, "model_val_best_miou.pth"))
        torch.save(model_disc.state_dict(), os.path.join(args.exp_dir, "modelD_val_best_miou.pth"))

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

    miou_all = np.zeros(test_loader.__len__())
    iou_all = np.zeros((test_loader.__len__(), 4))

    for batch_idx, data in tqdm.tqdm(enumerate(test_loader), total=test_loader.__len__()):
        image, label = data
        image, label = Variable(image).float(), \
                      Variable(label).type(torch.LongTensor)

        image, label = image.to(args.device), label.to(args.device)

        with torch.set_grad_enabled(False):
            prediction = model(image)
            # prediction_interp = interp(prediction)

        label = label.detach().cpu().numpy()
        pred = np.argmax(prediction.detach().cpu().numpy(), axis=1)

        flat_pred = pred[0, :].flatten()
        flat_label = label[0, :].flatten()
        miou, iou_classes = compute_mean_iou(flat_pred=flat_pred, flat_label=flat_label)

        iou_all[batch_idx, :] = iou_classes[:]
        miou_all[batch_idx] = miou

        if args.save_test:
            pred = pred[0, :, :]
            pred = rescale_image(pred)
            pred = convt_array_to_PIL(pred)
            filename = ntpath.basename(dataset.train_data_list[batch_idx])
            filename = os.path.join(args.exp_dir, filename)
            pred.save(filename)

    return miou_all, iou_all

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
    source_loss_eval, source_miou_eval = float("inf"), float("-inf")
    source_loss, source_miou = [], []

    # interp = torch.nn.Upsample(
    #     size=(args.image_size[1], args.image_size[0]),
    #     mode='bilinear',
    #     align_corners=False,
    # )

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
        pred_source_softmax = F.softmax(pred_source, dim=1)
        loss_seg_source = seg_loss_source(pred_source, labels)

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
               args.lambda_adv * loss_adv_scalar + \
               args.lambda_adv * loss_adv_patch + \
               args.lambda_adv * loss_adv_pixel
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
        loss_D_scalar_value += 0.5 * loss_D_scalar.item()
        loss_D_patch_value += 0.5 * loss_D_patch.item()
        loss_D_pixel_value += 0.5 * loss_D_pixel.item()

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
                data_type="target",
            )

            val_loss.append(val_loss_f)
            val_loss_f = np.min(np.array(val_loss))
            val_miou.append(val_miou_f)
            val_miou_f = np.max(np.array(val_miou))

            if args.tensorboard and (writer != None):
                writer.add_scalar('Val/Target_Cross_Entropy',
                                  val_loss_f,
                                  i_iter)
                writer.add_scalar('Val/Target_mIoU',
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

        if i_iter % args.iter_source_to_eval == 0:
            source_loss_eval, source_miou_eval = validate(
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
                val_loss=source_loss_eval,
                val_iou=source_miou_eval,
                data_type="source",
            )

            if args.tensorboard and (writer != None):
                writer.add_scalar('Train/Source_Cross_Entropy',
                                  source_loss_eval,
                                  i_iter)
                writer.add_scalar('Train/Source_mIoU',
                                  source_miou_eval,
                                  i_iter)

            source_loss.append(source_loss_eval)
            source_loss_eval = np.min(np.array(source_loss))
            source_miou.append(source_miou_eval)
            source_miou_eval = np.max(np.array(source_miou))

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
        data_type="target",
    )
    val_loss.append(val_loss_f)
    val_miou.append(val_miou_f)

    logger.info("==========================================")
    logger.info("Training DONE!")

    if args.tensorboard and (writer != None):
        writer.close()

    return val_loss, val_miou

def run_training_single_D(
    trainloader_source,
    trainloader_target,
    trainloader_iter,
    targetloader_iter,
    val_loader,
    model_seg,
    model_disc,
    gan_loss,
    seg_loss_source,
    semi_loss_criterion,
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
    source_loss_eval, source_miou_eval = float("inf"), float("-inf")
    source_loss, source_miou = [], []

    for i_iter in range(args.total_iterations):
        loss_seg_source_value = 0
        loss_adv_value = 0
        loss_D_value = 0
        loss_semi_value = 0

        model_seg.train()
        model_disc.train()

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
        for param in model_disc.parameters():
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
        pred_source_softmax = F.softmax(pred_source, dim=1)
        loss_seg_source = seg_loss_source(pred_source, labels)

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
        pred_softmax_target = F.softmax(pred_target, dim=1)

        ### semi loss for DNet of target ###
        if args.iter_semi_start > 0 and i_iter >= args.iter_semi_start:
            D_out_semi = model_disc(pred_softmax_target)
            mean_domain_threshold = get_mean_threshold(D_outs=D_out_semi, device=args.device)
            semi_ignore_mask = (D_out_semi > mean_domain_threshold).squeeze(1)
            semi_gt = torch.argmax(pred_softmax_target.data.cpu(), dim=1)
            semi_gt[semi_ignore_mask] = 255

            semi_ratio = 1.0 - float(semi_ignore_mask.sum().item()) / float(np.prod(semi_ignore_mask.shape))
            if semi_ratio == 0.0:
                loss_semi = None
                loss_semi_value += 0
            else:
                semi_gt = torch.LongTensor(semi_gt).to(args.device)
                loss_semi = semi_loss_criterion(pred_target, semi_gt)
                loss_semi_value += loss_semi.item()
        else:
            loss_semi = None

        ### adversarial loss ###
        D_out_scalar = model_disc(pred_softmax_target)
        disc_label_scalar = make_D_label(
            input=D_out_scalar,
            value=source_label,
            device=args.device,
            random=False,
        )

        loss_adv_scalar = gan_loss(D_out_scalar, disc_label_scalar)
        loss_adv_value += loss_adv_scalar.item()

        if loss_semi is None:
            loss = args.lambda_seg_source * loss_seg_source + \
                   args.lambda_adv * loss_adv_scalar
        else:
            loss = args.lambda_seg_source * loss_seg_source + \
                   args.lambda_adv * loss_adv_scalar + \
                   args.lambda_semi * loss_semi

        # loss = args.lambda_seg_source * loss_seg_source + \
        #        args.lambda_adv * loss_adv_scalar
        # loss = loss
        loss.backward()

        #################################################
        ############# (2) train discriminator ###########
        #################################################
        for param in model_disc.parameters():
            param.requires_grad = True

        ## train with source ##
        pred_source_softmax = pred_source_softmax.detach()
        pool_source = history_pool_true.query(pred_source_softmax)
        D_out_scalar = model_disc(pool_source)
        disc_label_scalar = make_D_label(
            input=D_out_scalar,
            value=source_label,
            device=args.device,
            random=True,
        )
        loss_D_real = 0.5 * gan_loss(D_out_scalar, disc_label_scalar)
        loss_D_real.backward()
        loss_D_value += loss_D_real.item()

        ## train with target ##
        pred_softmax_target = pred_softmax_target.detach()
        pool_target = history_pool_fake.query(pred_softmax_target)
        D_out_scalar = model_disc(pool_target)
        disc_label_scalar = make_D_label(
            input=D_out_scalar,
            value=target_label,
            device=args.device,
            random=True,
        )
        loss_D_fake = 0.5 * gan_loss(D_out_scalar, disc_label_scalar)
        loss_D_fake.backward()
        loss_D_value += loss_D_fake.item()

        optimizer_seg.step()
        optimizer_disc.step()

        logger.info('iter = {0:8d}/{1:8d} '
              'loss_seg_source = {2:.3f} '
              'loss_adv = {3:.3f} '
              'loss_semi = {4:.4f} '
              'loss_D_all = {5:.3f} '.format(
                i_iter, args.total_iterations,
                loss_seg_source_value,
                loss_adv_value,
                loss_semi_value,
                loss_D_value,
            )
        )

        if args.tensorboard and (writer != None):
            writer.add_scalar('Train/Cross_Entropy_Source',
                              loss_seg_source_value,
                              i_iter)
            writer.add_scalar('Train/Adversarial_loss',
                              loss_adv_value,
                              i_iter)
            writer.add_scalar('Train/Semi_loss',
                              loss_semi_value,
                              i_iter)
            writer.add_scalar('Train/Discriminator_all',
                              loss_D_value,
                              i_iter)

        current_epoch = i_iter * args.batch_size // args.tot_source
        if i_iter % args.iters_to_eval == 0:
            val_loss_f, val_miou_f = validate_singleD(
                i_iter=i_iter,
                val_loader=val_loader,
                model=model_seg,
                model_disc=model_disc,
                epoch=current_epoch,
                args=args,
                logger=logger,
                writer=writer,
                val_loss=val_loss_f,
                val_iou=val_miou_f,
                data_type="target",
            )

            val_loss.append(val_loss_f)
            val_loss_f = np.min(np.array(val_loss))
            val_miou.append(val_miou_f)
            val_miou_f = np.max(np.array(val_miou))

            if args.tensorboard and (writer != None):
                writer.add_scalar('Val/Target_Cross_Entropy',
                                  val_loss_f,
                                  i_iter)
                writer.add_scalar('Val/Target_mIoU',
                                  val_miou_f,
                                  i_iter)

            is_better_ss = current_loss_seg < loss_seg_min
            if is_better_ss:
                loss_seg_min = current_loss_seg
                torch.save(model_seg.state_dict(),
                    os.path.join(args.exp_dir, "model_train_best.pth")
                )
                torch.save(model_disc.state_dict(),
                    os.path.join(args.exp_dir, "Dscalar_train_best_loss.pth")
                )

        if i_iter % args.iter_source_to_eval == 0:
            source_loss_eval, source_miou_eval = validate_singleD(
                i_iter=i_iter,
                val_loader=val_loader,
                model=model_seg,
                model_disc=model_disc,
                epoch=current_epoch,
                args=args,
                logger=logger,
                writer=writer,
                val_loss=source_loss_eval,
                val_iou=source_miou_eval,
                data_type="source",
            )

            if args.tensorboard and (writer != None):
                writer.add_scalar('Train/Source_Cross_Entropy',
                                  source_loss_eval,
                                  i_iter)
                writer.add_scalar('Train/Source_mIoU',
                                  source_miou_eval,
                                  i_iter)

            source_loss.append(source_loss_eval)
            source_loss_eval = np.min(np.array(source_loss))
            source_miou.append(source_miou_eval)
            source_miou_eval = np.max(np.array(source_miou))

    current_epoch = int(args.total_iterations * args.batch_size / args.total_source)
    val_loss_f, val_miou_f = validate_singleD(
        i_iter=1000000000,
        val_loader=val_loader,
        model=model_seg,
        model_disc=model_disc,
        epoch=current_epoch,
        args=args,
        logger=logger,
        writer=writer,
        val_loss=val_loss_f,
        val_iou=val_miou_f,
        data_type="target",
    )
    val_loss.append(val_loss_f)
    val_miou.append(val_miou_f)

    logger.info("==========================================")
    logger.info("Training DONE!")

    if args.tensorboard and (writer != None):
        writer.close()

    return val_loss, val_miou

def run_training_SDA(
    trainloader_source,
    trainloader_target,
    trainloader_iter,
    targetloader_iter,
    traineval_target_loader,
    val_loader,
    model_seg,
    disc_scalar,
    disc_patch,
    disc_pixel,
    gan_loss,
    seg_loss_source,
    seg_loss_target,
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

    val_loss, val_miou = [], []
    val_loss_f, val_miou_f = float("inf"), float("-inf")
    loss_seg_min = float("inf")
    source_loss_eval, source_miou_eval = float("inf"), float("-inf")
    source_loss, source_miou = [], []
    global_max_miou = float("-inf")
    target_train_miou = []

    for i_iter in range(args.total_iterations):
        loss_seg_source_value = 0
        loss_seg_target_value = 0
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
        pred_source_softmax = F.softmax(pred_source, dim=1)
        loss_seg_source = seg_loss_source(pred_source, labels)

        # current_loss_seg = loss_seg_source.item()
        loss_seg_source_value += loss_seg_source.item()

        try:
            _, batch = next(targetloader_iter)
        except StopIteration:
            targetloader_iter = enumerate(trainloader_target)
            _, batch = next(targetloader_iter)

        images, labels = batch
        images = Variable(images).to(args.device)
        labels = Variable(labels.long()).to(args.device)

        pred_target = model_seg(images)
        pred_softmax_target = F.softmax(pred_target, dim=1)
        loss_seg_target = seg_loss_target(pred_target, labels)

        current_loss_seg = loss_seg_target.item()
        loss_seg_target_value += current_loss_seg

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
               args.lambda_seg_target * loss_seg_target + \
               args.lambda_adv * loss_adv_scalar + \
               args.lambda_adv * loss_adv_patch + \
               args.lambda_adv * loss_adv_pixel
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
              'loss_seg_target = {3:.3f} '
              'loss_adv = {4:.3f} '
              'loss_D_all = {5:.3f} '
              'loss_D_scalar = {6:.3f} '
              'loss_D_patch = {7:.3f} '
              'loss_D_pixel = {8:.3f} '.format(
                i_iter, args.total_iterations,
                loss_seg_source_value,
                loss_seg_target_value,
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
                data_type="target",
            )

            val_loss.append(val_loss_f)
            val_loss_f = np.min(np.array(val_loss))
            val_miou.append(val_miou_f)
            val_miou_f = np.max(np.array(val_miou))

            if args.tensorboard and (writer != None):
                writer.add_scalar('Val/Target_Cross_Entropy',
                                  val_loss_f,
                                  i_iter)
                writer.add_scalar('Val/Target_mIoU',
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

            cur_target_train_miou = get_miou(
                dataloader=traineval_target_loader,
                model=model_seg,
                global_max_miou=global_max_miou,
                args=args,
            )
            if args.tensorboard and (writer != None):
                writer.add_scalar('Train/Target_mIoU',
                                  cur_target_train_miou,
                                  i_iter)

            target_train_miou.append(cur_target_train_miou)
            global_max_miou = np.max(np.array(target_train_miou))

        if i_iter % args.iter_source_to_eval == 0:
            source_loss_eval, source_miou_eval = validate(
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
                val_loss=source_loss_eval,
                val_iou=source_miou_eval,
                data_type="source",
            )

            if args.tensorboard and (writer != None):
                writer.add_scalar('Train/Source_Cross_Entropy',
                                  source_loss_eval,
                                  i_iter)
                writer.add_scalar('Train/Source_mIoU',
                                  source_miou_eval,
                                  i_iter)

            source_loss.append(source_loss_eval)
            source_loss_eval = np.min(np.array(source_loss))
            source_miou.append(source_miou_eval)
            source_miou_eval = np.max(np.array(source_miou))

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
        data_type="target",
    )
    val_loss.append(val_loss_f)
    val_miou.append(val_miou_f)

    logger.info("==========================================")
    logger.info("Training DONE!")

    if args.tensorboard and (writer != None):
        writer.close()

    return val_loss, val_miou, target_train_miou


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