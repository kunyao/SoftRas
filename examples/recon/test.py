import argparse

import torch
import torch.nn.parallel
import datasets
from utils import AverageMeter, img_cvt
import soft_renderer as sr
import soft_renderer.functional as srf
import models
import time
import os
import imageio
import numpy as np

import sys
sys.path.append('./ChamferDistancePytorch/')
from chamfer3D import dist_chamfer_3D
import fscore

BATCH_SIZE = 100
IMAGE_SIZE = 64
CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')

PRINT_FREQ = 100
SAVE_FREQ = 100

MODEL_DIRECTORY = './data/models'
DATASET_DIRECTORY = './data/datasets'

SIGMA_VAL = 0.01
IMAGE_PATH = ''

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment-id', type=str)
parser.add_argument('-d', '--model-directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY)

parser.add_argument('-cls', '--class-ids', type=str, default=CLASS_IDS_ALL)
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-bs', '--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('-img', '--image-path', type=str, default=IMAGE_PATH)

parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)

parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)
parser.add_argument('-us', '--use-soft', action='store_true', default=False)  # add method switcher
parser.add_argument('-o', '--output-dir', type=str, default='data/results/test')
args = parser.parse_args()

# setup model & optimizer
model = models.Model('data/obj/sphere/sphere_642.obj', args=args)
model = model.cuda()

state_dicts = torch.load(args.model_directory)
model.load_state_dict(state_dicts['model'], strict=False)
model.eval()

dataset_val = datasets.ShapeNet(args.dataset_directory, args.class_ids.split(','), 'val', load_pc=False)

directory_output = args.output_dir
os.makedirs(directory_output, exist_ok=True)
directory_mesh = os.path.join(directory_output, args.experiment_id)
os.makedirs(directory_mesh, exist_ok=True)

def save_obj(filename, vertices, faces=[]):
    """
    Args:
        vertices: N*3 np.array
        faces: M*3 np.array
    """

    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v {:.3} {:.3} {:.3}\n'.format(*v))
        for f in faces:
            fp.write('f {} {} {}\n'.format(*(f + 1)))

def save_voxel(filename, voxel):
    '''
    Input data:
        Sampling range: [-0.5, 0.5]
        Sampling on the centroid
        Sampling resolution: res
        Start: -0.5 + 0.5 / res
        End: 0.5 - 0.5 / res
        Step: 1 / res
        Transfer: [0, res-1] --> [start, end]
    '''
    vertices = []
    res_x = voxel.shape[0]
    res_y = voxel.shape[1]
    res_z = voxel.shape[2]
    for i in range(res_x):
        for j in range(res_y):
            for k in range(res_z):
                if voxel[i, j, k] == 1:
                    X = j / res_y - 0.5 * (res_y - 1) / res_y
                    Y = i / res_x - 0.5 * (res_x - 1) / res_x
                    Z = -k / res_z + 0.5 * (res_z - 1) / res_z
                    vertices.append([X, Y, Z])

    save_obj(filename, vertices)

def test():
    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()

    iou_all = []
    d1_all = []
    chamfer_all = []
    f_all = []

    # chamLoss = dist_chamfer_3D.chamfer_3DDist()
    results_dir = os.path.join(directory_mesh, 'results.txt')
    f = open(results_dir, 'w')
    for class_id, class_name in dataset_val.class_ids_pair:

        directory_mesh_cls = os.path.join(directory_mesh, class_id)
        os.makedirs(directory_mesh_cls, exist_ok=True)
        iou = 0
        d1_sum = 0
        chamfer_sum = 0
        f_sum = 0

        # for i, (im, vx, pc) in enumerate(dataset_val.get_all_batches_for_evaluation(args.batch_size, class_id)):
        for i, (im, vx) in enumerate(dataset_val.get_all_batches_for_evaluation(args.batch_size, class_id)):
            images = torch.autograd.Variable(im).cuda()
            voxels = vx.numpy()

            batch_iou, vertices, faces, voxel_recon = model(images, voxels=voxels, task='test', return_voxel=True)
            iou += batch_iou.sum()
            # d1, d2, _, _ = chamLoss(vertices, pc.cuda())
            # f_score, _, _ = fscore.fscore(d1, d2)
            # d1_sum += d1.mean(1).sum().item()
            # chamfer_sum += d1.mean(1).sum().item() + d2.mean(1).sum().item()
            # f_sum += f_score.sum().item()


            batch_time.update(time.time() - end)
            end = time.time()

            # save demo images
            for k in range(vertices.size(0)):
                obj_id = (i * args.batch_size + k)
                if obj_id % args.save_freq == 0:
                    mesh_path = os.path.join(directory_mesh_cls, '{:06d}_mesh_iou{:.4f}.obj'.format(obj_id, batch_iou[k].item()))
                    voxel_path = os.path.join(directory_mesh_cls, '{:06d}_voxel_iou{:.4f}.obj'.format(obj_id, batch_iou[k].item()))
                    gt_path = os.path.join(directory_mesh_cls, '{:06d}_gt.obj'.format(obj_id))
                    input_path = os.path.join(directory_mesh_cls, '%06d.png' % obj_id)
                    srf.save_obj(mesh_path, vertices[k], faces[k])
                    imageio.imsave(input_path, img_cvt(images[k]))
                    save_voxel(voxel_path, voxel_recon[k])
                    save_voxel(gt_path, voxels[k])

            # print loss
            if i % args.print_freq == 0:
                print('Iter: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f}\t'
                      'IoU {2:.3f}\t'.format(i, ((dataset_val.num_data[class_id] * 24) // args.batch_size),
                                             batch_iou.mean(),
                                             batch_time=batch_time))

        iou_cls = iou / 24. / dataset_val.num_data[class_id] * 100
        # d1_cls =  d1_sum / 24. / dataset_val.num_data[class_id] * 100
        # chamfer_cls = chamfer_sum / 24. / dataset_val.num_data[class_id] * 100
        # f_cls = f_sum / 24. / dataset_val.num_data[class_id] * 100
        iou_all.append(iou_cls)
        # d1_all.append(d1_cls)
        # chamfer_all.append(chamfer_cls)
        # f_all.append(f_cls)

        print('=================================')
        print('Mean IoU: %.3f for class %s\n' % (iou_cls, class_name))
        f.write('=================================\n')
        f.write('Mean IoU: %.3f for class %s\n' % (iou_cls, class_name))
        f.flush()
        # print('Mean d1: %.3f for class %s\n' % (d1_cls, class_name))
        # print('Mean Chamfer: %.3f for class %s\n' % (chamfer_cls, class_name))
        # print('Mean F-score: %.3f for class %s\n' % (f_cls, class_name))

    print('=================================')
    print('Mean IoU: %.3f for all classes' % (sum(iou_all) / len(iou_all)))
    f.write('=================================\n')
    f.write('Mean IoU: %.3f for all classes\n' % (sum(iou_all) / len(iou_all)))
    f.close()


test()
