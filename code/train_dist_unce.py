import argparse
import logging
import os
import random
import shutil
import sys
import time
from itertools import cycle
import tempfile

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

from dataloaders import utils
from dataloaders.dataset_semi import (BaseDataSets, RandomGenerator,RandomGenerator_Strong_Weak,
                                      TwoStreamBatchSampler)
from networks.discriminator import FCDiscriminator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume,test_single_volume_val,test_co_volume_val,test_single_volume_co_save

from utils.mixup import *

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ours_ws', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--early_stop', type=int, default=10000,
                    help='early_stop')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--check', type=int,
                    default=1000, help='maximum epoch number to train')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')

parser.add_argument('--lamda', type=float,
                    default=1, help='consistency')

parser.add_argument('--choice', type=str,
                    default='all', help='mix type')
parser.add_argument('--labeled_ratio', type=int,  default=8,
                    help='output channel of network')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model



def generate_cutmix_mask(shape, prop_range = 0.2, n_holes=1, random_aspect_ratio=True, within_bounds=True):
    if isinstance(prop_range, float):
        prop_range = (prop_range, prop_range)

    n_masks, _, h, w = list(shape)


    # mask = np.ones((h, w), np.float32)
    # valid = np.zeros((h ,w),np.float32)

    mask_props = np.random.uniform(prop_range[0], prop_range[1], size=(n_masks, n_holes))
    if random_aspect_ratio:
        y_props = np.exp(np.random.uniform(low=0.0, high=1.0, size=(n_masks, n_holes)) * np.log(mask_props))
        x_props = mask_props / y_props
    else:
        y_props = x_props = np.sqrt(mask_props)

    fac = np.sqrt(1.0 / n_holes)
    y_props *= fac
    x_props *= fac

    sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array((h, w))[None, None, :])

    if within_bounds:
        positions = np.round((np.array((h, w)) - sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(positions, positions + sizes, axis=2)
    else:
        centres = np.round(np.array((h, w)) * np.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

    masks = np.zeros(shape)
    for i, sample_rectangles in enumerate(rectangles):
        for y0, x0, y1, x1 in sample_rectangles:
            # print('len:', y0 - y1)
            # print('hig:', x0 - x1)

            masks[i,:, int(y0):int(y1), int(x0):int(x1)] = 1

    masks = torch.from_numpy(masks)

    return masks


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model



def generate_cpl_from_scribbles(
    scribble_batch,
    num_classes,
    decay_alpha = 0.1,
    threshold = 0.05,

    ):

    assert scribble_batch.dim() == 3 or (scribble_batch.dim() == 4 and scribble_batch.shape[1] == 1)
    
    if scribble_batch.dim() == 4:
        scribble_batch = scribble_batch.squeeze(1)
        
    batch_size, H, W = scribble_batch.shape
    device = scribble_batch.device
    
    final_cpl_batch = []

    for i in range(batch_size):
        scribble_image_np = scribble_batch[i].cpu().numpy().astype(np.uint8)
        cpl_maps_for_image = []

        for c in range(0, num_classes):
            binary_mask = (scribble_image_np == c)
            distance_map = distance_transform_edt(1 - binary_mask)
            cpl_map = np.exp(-decay_alpha * distance_map)
            cpl_map[cpl_map < threshold] = threshold           
            cpl_maps_for_image.append(cpl_map)


        foreground_cpls = np.stack(cpl_maps_for_image, axis=0)

        background_channel = np.zeros((1, H, W), dtype=np.float32)
        full_cpl_image = np.concatenate([background_channel, foreground_cpls], axis=0)

        final_cpl_batch.append(torch.from_numpy(full_cpl_image).float())

    return torch.stack(final_cpl_batch, dim=0).to(device)

def train(args, snapshot_path):
    print(args.model)
    args.model = 'unet_cct'
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    #model2 = create_model()
    
    model1_ema = create_model(ema=True)
    #model2_ema = create_model(ema=True)
    
    model1 = kaiming_normal_init_weight(model1)
    model1_ema = kaiming_normal_init_weight(model1_ema)
    #teacher = create_model(ema=True)

    db_train_labeled = BaseDataSets(base_dir=args.root_path, num=8, labeled_type="labeled",ratio = args.labeled_ratio, fold=args.fold, split="train", sup_type=args.sup_type, transform=transforms.Compose([
        RandomGenerator_Strong_Weak(args.patch_size)
    ]))
    db_train_unlabeled = BaseDataSets(base_dir=args.root_path, num=8, labeled_type="unlabeled", fold=args.fold, split="train", sup_type=args.sup_type, transform=transforms.Compose([
        RandomGenerator_Strong_Weak(args.patch_size)]))

    trainloader_labeled = DataLoader(db_train_labeled, batch_size=args.batch_size//2, shuffle=True,
                                     num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader_unlabeled = DataLoader(db_train_unlabeled, batch_size=args.batch_size//2, shuffle=True,
                                       num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    db_val = BaseDataSets(base_dir=args.root_path,
                          fold=args.fold, split="val", )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    
    
    

    model1.train()

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)


    ce_loss = CrossEntropyLoss(ignore_index=args.num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader_labeled)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader_labeled) + 1

    # randomly generate one aug for each iteration
    best_iter = 0
    best_performance1 = 0.0
    best_performance2 = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    visualize_iter = random.randint(0, len(trainloader_labeled)-1)

    for epoch_num in iterator:
        for i, data in enumerate(zip(cycle(trainloader_labeled), trainloader_unlabeled)):
            sampled_batch_labeled, sampled_batch_unlabeled = data[0], data[1]

            labeled_volume_batch_weak,_, label_batch = \
                sampled_batch_labeled['image_w'].cuda(), sampled_batch_labeled['image_s'].cuda(), sampled_batch_labeled['label'].cuda()
                
                
            _,unlabeled_volume_batch_weak = sampled_batch_unlabeled['image_s'].cuda(),\
                sampled_batch_unlabeled['image_w'].cuda()

            cpl_target = generate_cpl_from_scribbles(label_batch,num_classes)
        


            labeled_volume_batch_weak = labeled_volume_batch_weak
            #label_batch = torch.cat((label_batch,label_batch),0)
            
            output_sup_weak,output_sup_weak_ft = model1(labeled_volume_batch_weak,mutli = True)

            beta = random.random()



            probs_weak = F.softmax(output_sup_weak, dim=1)
            probs_ft =  F.softmax(output_sup_weak_ft, dim=1)

            probs_weak_combined = beta*probs_weak+(1-beta)*probs_ft


            confidence_threshold = 0.9

            max_probs_weak, pseudo_label_weak = torch.max(probs_weak_combined, dim=1)
            binary_mask_weak = max_probs_weak.ge(confidence_threshold).float()

            cpl_threshold = 0.5

            max_cpl_values, _ = torch.max(cpl_target[:, 1:, :, :], dim=1)
            cpl_binary_mask = max_cpl_values.ge(cpl_threshold)

            union_mask = binary_mask_weak.long() | cpl_binary_mask.long()
            union_mask = union_mask.long()

            masked_pseudo_label = torch.full_like(pseudo_label_weak, fill_value=args.num_classes)
            masked_pseudo_label = torch.where(union_mask.bool(), pseudo_label_weak, masked_pseudo_label)
            masked_pseudo_label = masked_pseudo_label.detach().long()

            loss_ce = 0.5*ce_loss(output_sup_weak, label_batch[:].long())+\
                       0.5 * ce_loss(output_sup_weak_ft, label_batch[:].long())+\
                       0.5 * ce_loss(output_sup_weak_ft, masked_pseudo_label[:].long())+\
                       0.5 * ce_loss(output_sup_weak, masked_pseudo_label[:].long())



            volume_combined_weak= unlabeled_volume_batch_weak

            with torch.no_grad():
                model1.eval()
                outputs1_unlabeled_weak, outputs1_unlabeled_weak_ft  = model1_ema(volume_combined_weak,mutli = True)
                weak_label1 = torch.softmax(outputs1_unlabeled_weak, dim=1).detach()
                weak_label1_ft = torch.softmax(outputs1_unlabeled_weak_ft, dim=1).detach()  
                model1.train()
                beta = random.random()

                weak_label1 = beta*weak_label1 +(1-beta)*weak_label1_ft 
               #weak_label2 = beta2 * custom_operation(weak_label2,strong_label2)+\
                #    (1-beta2) * custom_operation(strong_label2,weak_label2)
                
                
            shape = list(volume_combined_weak.shape)
            shape[1] = 1
            #print(shape)
            MixMask = generate_cutmix_mask(shape=shape).cuda().float()
            rand_index = torch.randperm(volume_combined_weak.size()[0]).cuda()
            
            volume_combined_weak = MixMask * volume_combined_weak + (1 - MixMask) * volume_combined_weak[rand_index]
            pseudo_lab1 = MixMask * weak_label1 + (1 - MixMask) * weak_label1[rand_index]


            pseudo_lab1 = torch.argmax(pseudo_lab1, dim=1)
            
            
            outputs1_unlabeled_weak, outputs1_unlabeled_weak_ft= model1(volume_combined_weak,mutli = True)
            #outputs1_unlabeled_weak = torch.softmax(outputs1_unlabeled_weak, dim=1)
            
            
            loss_unsup = 0.5*F.cross_entropy(outputs1_unlabeled_weak, pseudo_lab1)+\
                            0.5*F.cross_entropy(outputs1_unlabeled_weak_ft, pseudo_lab1)




            loss = loss_ce + args.lamda * loss_unsup
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            update_ema_variables(model1, model1_ema, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_


            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_unsup', loss_unsup, iter_num)
            
            if iter_num  % 500 ==0:
                import matplotlib.pyplot as plt
                from torchvision.utils import save_image

                vis_dir = os.path.join(snapshot_path,  f"jamit/{iter_num}")
                os.makedirs(vis_dir, exist_ok=True)

               
    
                #os.makedirs(f'jamit/{i}', exist_ok=True)

                torch.save(sampled_batch_labeled['image_w'].cpu(), os.path.join( vis_dir, 'image_w.pt'))
                torch.save(label_batch.cpu(), os.path.join( vis_dir, 'label.pt'))
                torch.save(cpl_target.cpu(), os.path.join( vis_dir, 'cpl_target.pt'))
                torch.save(binary_mask_weak.cpu() , os.path.join( vis_dir, 'binary_mask_weak.pt'))
                torch.save(cpl_binary_mask.cpu() , os.path.join( vis_dir, 'cpl_binary_mask.pt'))
                torch.save(max_cpl_values.cpu() , os.path.join( vis_dir, 'max_cpl_values.pt'))
                torch.save(probs_weak.cpu(), os.path.join( vis_dir, 'probs_weak.pt'))
                torch.save(probs_ft.cpu() , os.path.join(vis_dir, 'probs_ft.pt'))
                torch.save(probs_weak_combined.cpu() , os.path.join(vis_dir, 'probs_weak_final.pt'))
                torch.save(union_mask.cpu() , os.path.join(vis_dir, 'union_mask.pt'))
                torch.save(masked_pseudo_label.cpu() , os.path.join(vis_dir, 'masked_pseudo_label.pt'))

                #torch.save(outputs1_unlabeled_weak.cpu(), os.path.join(vis_dir, 'outputs1_unlabeled_weak.pt'))
                           
            
                print(f'dir saved to: {vis_dir}')
            
            if iter_num > 0 and iter_num % 100 == 0:
                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f, loss_unsup: %f' %
                    (iter_num, loss.item(), loss_ce.item(), loss_unsup.item()))

            if iter_num >= 0 and iter_num % args.check == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    #print(sampled_batch["label"].shape)
                    metric_i, prediction = test_single_volume_val(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    
                image = sampled_batch["image"][0, 0:1, :, :]
                #print(sampled_batch["image"][0].max())
                writer.add_image('train/val_Image_val', image/image.max(), iter_num)
                #print(sampled_batch["label"].shape)
                
                
                writer.add_image('train/val_Image_label', sampled_batch["label"][0,0:1 ,...].long()/4, iter_num)
                
                writer.add_image('train/val_model_Prediction1',
                                 prediction[0:1, ...]/4, iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                if performance > best_performance1:           
                    best_performance1 = performance
                    best_iter = iter_num
                    torch.save(model1.state_dict(), os.path.join(
                        snapshot_path, 'model_best.pth'))
                    logging.info('best model found, best_iter: %d' % best_iter)
                    
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)


                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f model 1' % (iter_num, performance, mean_hd95))
                model1.train()

            if iter_num >= max_iterations or iter_num- best_iter > args.early_stop: 
                
                break
        if iter_num >= max_iterations or iter_num- best_iter > args.early_stop:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    print(f'model: {args.model}')
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model_WSS/{}/{}/{}/{}".format(
        args.exp, args.labeled_ratio,args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    custom_tmp = os.path.join(snapshot_path, ".tmp")
    os.makedirs(custom_tmp, exist_ok=True)
    tempfile.tempdir = custom_tmp
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
