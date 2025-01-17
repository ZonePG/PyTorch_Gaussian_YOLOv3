from __future__ import division

from utils.utils import *
from utils.seed import set_seed, setup_cudnn
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.kittiapi_evaluator import KITTIAPIEvaluator
from utils.mmkittiapi_evaluator import MMKITTIAPIEvaluator
from utils.parse_yolo_weights import parse_yolo_weights, parse_mmyolo_weights
from models.yolov3 import *
from models.mmyolov3 import MMYOLOv3
from dataset.cocodataset import *
from dataset.kittidataset import *
from dataset.mmkittidataset import *

import os
import argparse
import yaml
import random

import torch
from torch.autograd import Variable
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg',
                        help='config file. see readme')
    parser.add_argument('--weights_path', type=str,
                        default=None, help='darknet weights file')
    parser.add_argument('--n_cpu', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--checkpoint_interval', type=int,
                        default=1000, help='interval between saving checkpoints')
    parser.add_argument('--eval_interval', type=int,
                            default=4000, help='interval between evaluations')
    parser.add_argument('--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints',
                        help='directory where checkpoint files are saved')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument(
        '--tfboard_dir', help='tensorboard path for logging', type=str, default=None)
    parser.add_argument(
        '--multimodal', action='store_true', default=False, help='train with rgb and depth image')
    parser.add_argument('--data_dir', type=str,
                        required=True,
                        help='directory where rgb datasets are saved')
    return parser.parse_args()


def main():
    """
    YOLOv3 trainer. See README for details.
    """
    args = parse_args()
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    print("successfully loaded config file: ", cfg)

    momentum = cfg['TRAIN']['MOMENTUM']
    decay = cfg['TRAIN']['DECAY']
    burn_in = cfg['TRAIN']['BURN_IN']
    iter_size = cfg['TRAIN']['MAXITER']
    steps = eval(cfg['TRAIN']['STEPS'])
    batch_size = cfg['TRAIN']['BATCHSIZE']
    subdivision = cfg['TRAIN']['SUBDIVISION']
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    random_resize = cfg['AUGMENTATION']['RANDRESIZE']
    base_lr = cfg['TRAIN']['LR'] / batch_size / subdivision
    gradient_clip = cfg['TRAIN']['GRADIENT_CLIP']

    print('effective_batch_size = batch_size * iter_size = %d * %d' %
          (batch_size, subdivision))

    # Make trainer behavior deterministic
    set_seed(seed=0)
    setup_cudnn(deterministic=True)

    # Learning rate setup
    def burnin_schedule(i):
        if i < burn_in:
            factor = pow(i / burn_in, 4)
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    # Initiate model
    if not args.multimodal:
        model = YOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)
    else:
        model = MMYOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)
    print(model)

    if args.weights_path:
        print("loading darknet weights....", args.weights_path)
        if not args.multimodal:
            parse_yolo_weights(model, args.weights_path)
        else:
            parse_mmyolo_weights(model, args.weights_path)
    elif args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        state = torch.load(args.checkpoint)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    if cuda:
        print("using cuda") 
        model = model.cuda()

    if args.tfboard_dir:
        print("using tfboard")
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(args.tfboard_dir)

    model.train()

    imgsize = cfg['TRAIN']['IMGSIZE']
    # dataset = COCODataset(model_type=cfg['MODEL']['TYPE'],
    #               data_dir='/home/zonepg/datasets/coco/',
    #               img_size=imgsize,
    #               augmentation=cfg['AUGMENTATION'],
    #               debug=args.debug)
    if not args.multimodal:
        dataset = KITTIDataset(model_type=cfg['MODEL']['TYPE'],
                    data_dir=args.data_dir,
                    img_size=imgsize,
                    augmentation=cfg['AUGMENTATION'],
                    debug=args.debug)
    else:
        dataset = MMKITTIDataset(model_type=cfg['MODEL']['TYPE'],
                    data_dir0=args.data_dir,
                    data_dir1='/home/zonepg/datasets/kitti_depth/',
                    img_size=imgsize,
                    augmentation=cfg['AUGMENTATION'],
                    debug=args.debug)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
    dataiterator = iter(dataloader)

    if not args.multimodal:
        # evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
        #                 data_dir='/home/zonepg/datasets/coco/',
        #                 img_size=cfg['TEST']['IMGSIZE'],
        #                 confthre=cfg['TEST']['CONFTHRE'],
        #                 nmsthre=cfg['TEST']['NMSTHRE'])
        evaluator = KITTIAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                        data_dir=args.data_dir,
                        img_size=cfg['TEST']['IMGSIZE'],
                        confthre=cfg['TEST']['CONFTHRE'],
                        nmsthre=cfg['TEST']['NMSTHRE'])
    else:
        evaluator = MMKITTIAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                        data_dir0=args.data_dir,
                        data_dir1='/home/zonepg/datasets/kitti_depth/',
                        img_size=cfg['TEST']['IMGSIZE'],
                        confthre=cfg['TEST']['CONFTHRE'],
                        nmsthre=cfg['TEST']['NMSTHRE'])

    device = torch.device("cuda" if cuda else "cpu")

    # optimizer setup
    # set weight decay only on conv.weight
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'conv.weight' in key:
            params += [{'params':value, 'weight_decay':decay * batch_size * subdivision}]
        else:
            params += [{'params':value, 'weight_decay':0.0}]
    optimizer = optim.SGD(params, lr=base_lr, momentum=momentum,
                          dampening=0, weight_decay=decay * batch_size * subdivision)

    iter_state = 0

    if args.checkpoint:
        if 'optimizer_state_dict' in state.keys():
            optimizer.load_state_dict(state['optimizer_state_dict'])
            iter_state = state['iter'] + 1

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    # start training loop
    for iter_i in range(iter_state, iter_size + 1):

        # COCO evaluation
        # if iter_i % args.eval_interval == 0 and False:
        if iter_i % args.eval_interval == 0:
        # if True:
            print('evaluating...')
            ap = evaluator.evaluate(model)
            model.train()
            if args.tfboard_dir:
                # val/aP
                tblogger.add_scalar('val/aP50', ap['aP50'], iter_i)
                tblogger.add_scalar('val/aP75', ap['aP75'], iter_i)
                tblogger.add_scalar('val/aP5095', ap['aP5095'], iter_i)
                tblogger.add_scalar('val/aP5095_S', ap['aP5095_S'], iter_i)
                tblogger.add_scalar('val/aP5095_M', ap['aP5095_M'], iter_i)
                tblogger.add_scalar('val/aP5095_L', ap['aP5095_L'], iter_i)

        # subdivision loop
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            if not args.multimodal:
                try:
                    imgs, targets, _, ids = next(dataiterator)  # load a batch
                except StopIteration:
                    dataiterator = iter(dataloader)
                    imgs, targets, _, ids = next(dataiterator)  # load a batch
                imgs = imgs.to(device)
                targets = targets.to(device)
                loss = model(imgs, targets)
                loss.backward()
            else:
                try:
                    imgs0, imgs1, targets, _, ids = next(dataiterator)  # load a batch
                except StopIteration:
                    dataiterator = iter(dataloader)
                    imgs0, imgs1, targets, _, ids = next(dataiterator)  # load a batch
                imgs0 = imgs0.to(device)
                imgs1 = imgs1.to(device)
                targets = targets.to(device)
                loss = model(imgs0, imgs1, targets)
                loss.backward()

        if gradient_clip >= 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()
        scheduler.step()

        if iter_i % 10 == 0:
            # logging
            current_lr = scheduler.get_lr()[0] * batch_size * subdivision
            print('[Iter %d/%d] [lr %f] '
                  '[Losses: xy %f, wh %f, conf %f, cls %f, total %f, imgsize %d]'
                  % (iter_i, iter_size, current_lr,
                     model.loss_dict['xy'], model.loss_dict['wh'],
                     model.loss_dict['conf'], model.loss_dict['cls'], 
                     loss, imgsize),
                  flush=True)

            if args.tfboard_dir:
                # lr
                tblogger.add_scalar('lr', current_lr, iter_i)
                # train/loss
                tblogger.add_scalar('train/loss_xy', model.loss_dict['xy'], iter_i)
                tblogger.add_scalar('train/loss_wh', model.loss_dict['wh'], iter_i)
                tblogger.add_scalar('train/loss_conf', model.loss_dict['conf'], iter_i)
                tblogger.add_scalar('train/loss_cls', model.loss_dict['cls'], iter_i)
                tblogger.add_scalar('train/loss', loss, iter_i)

            # random resizing
            if random_resize:
                imgsize = (random.randint(0, 9) % 10 + 10) * 32
                dataset.img_shape = (imgsize, imgsize)
                dataset.img_size = imgsize
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
                dataiterator = iter(dataloader)

        # save checkpoint
        if args.checkpoint_dir and iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            torch.save({'iter': iter_i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        },
                        os.path.join(args.checkpoint_dir, "snapshot"+str(iter_i)+".ckpt"))

    if args.tfboard_dir:
        tblogger.close()


if __name__ == '__main__':
    main()
