"""
Shape from silhouette
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse
import glob

import soft_renderer as sr
import soft_renderer.functional as srf
from recon import datasets

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')

dataset_directory = './data/datasets'
class_ids = '02828884'
# class_ids = (
    # '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    # '03691459,04090263,04256520,04379243,04401088,04530566')
dataset = datasets.ShapeNet(dataset_directory, class_ids.split(','), 'train', load_template=True, load_camera=True)

def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()

def iou_3d(faces, voxels):

    faces_norm = faces * 1. * (32. - 1) / 32. + 0.5
    voxels_predict = srf.voxelization(faces_norm, 32, False).cpu().numpy()
    voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
    iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))
    return iou, voxels_predict


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


def voxelize(res_x, res_y, res_z):
    '''
    Sampling range: [-alpha, alpha]
    Start: -alpha + alpha / res
    End: alpha - alpha / res
    Step: 1 / res
    Transfer: X or Y or Z: [0, res_x-1] --> [start, end]
    '''
    voxel_pos = np.zeros((res_x, res_y, res_z, 3))
    for i in range(res_x):
        for j in range(res_y):
            for k in range(res_z):
                X = i / res_x - 0.5 * (res_x - 1) / res_x
                Y = j / res_y - 0.5 * (res_y - 1) / res_y
                Z = k / res_z - 0.5 * (res_z - 1) / res_z
                voxel_pos[i, j, k] = np.array([Z, X, Y])
    return voxel_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str,
        default=os.path.join(data_dir, 'source.npy'))
    parser.add_argument('-c', '--camera-input', type=str,
        default=os.path.join(data_dir, 'camera.npy'))
    parser.add_argument('-o', '--output-dir', type=str,
        default=os.path.join(data_dir, 'results/output_deform3'))
    parser.add_argument('-b', '--batch-size', type=int,
        default=120)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # read training images and camera poses
    images_gt, dist_maps, voxel, camera_distances, elevations, viewpoints, template_v, template_f, camera_matrix = dataset.get_one_obj(6, load_template=True, load_camera=True)


    voxel = voxel.numpy()
    images_gt = images_gt.cuda()
    dist_maps = dist_maps.cuda()
    template_v = template_v.cuda()
    camera_matrix = camera_matrix.cuda()

    renderer = sr.SoftRenderer(camera_mode='projection', P=camera_matrix)

    images_size = 137
    image_space = True  # flag true only for images_sizes=137
    # Load mask array (24, 4, H, W) -> (24, H, W)
    mask = images_gt[:, 3]
    # Voxelize the space (idx, idy, idz) -> (x ,y ,z)
    samples = voxelize(32, 32, 32).reshape(-1, 3).astype('float32')  # must be float32 type
    # Project the voxel
    samples = torch.from_numpy(samples).repeat(24, 1, 1).cuda()  # (24, Nv, 3)
    p3d_m = sr.Mesh(samples, np.zeros((1, 1, 3)))  # fake mesh with ndim=3
    p25d_m = renderer.transform(p3d_m)
    p2d = p25d_m.vertices[:, :, :2]
    if not image_space:
        p2d = (p2d + 1.0) * images_size / 2
    p2d = p2d.long()

    X = p2d[:, :, 0]
    Y = p2d[:, :, 1]
    bx0 = (X < 0)
    bx1 = (X >= images_size)
    X[bx0] = 0
    Y[bx0] = 0
    X[bx1] = 0
    Y[bx1] = 0

    by0 = (Y < 0)
    by1 = (Y >= images_size)
    X[by0] = 0
    Y[by0] = 0
    X[by1] = 0
    Y[by1] = 0
    mask[:, 0, 0] = 5

    test_img = torch.zeros(24, images_size, images_size).cuda()
    for v in range(24):
        xs = p2d[v, : , 0]
        ys = p2d[v, : , 1]
        test_img[v, ys, xs] = 1.0
    torchvision.utils.save_image(test_img[:, None], os.path.join(args.output_dir, 'proj.png'))
    torchvision.utils.save_image(mask[:, None], os.path.join(args.output_dir, 'gt.png'))

    '''
    # Validate projection
    mesh = sr.Mesh(template_v.repeat(24, 1, 1).cuda(), template_f.repeat(24, 1, 1).cuda())
    mesh_proj = renderer.transform(mesh)
    # import ipdb
    # ipdb.set_trace()
    pixels = mesh_proj.vertices[:, :, :2]
    pixels[:, :, 1] = -pixels[:, :, 1]
    pixels = (pixels + 1.0) * images_size / 2
    pixels = pixels.long()
    test_img = torch.zeros(24, images_size, images_size).cuda()
    for v in range(24):
        xs = pixels[v, : , 0]
        ys = pixels[v, : , 1]
        test_img[v, ys, xs] = 1.0

    torchvision.utils.save_image(test_img[:, None], os.path.join(args.output_dir, 'proj.png'))
    torchvision.utils.save_image(mask[:, None], os.path.join(args.output_dir, 'gt.png'))
    '''
    # Carve the outside: each view insider = 1 outsider = 0, all views insider is True insider
    occ_2d = torch.zeros(24, 32*32*32).cuda()
    for v in range(24):
        occ_2d[v, mask[v, p2d[v, :, 1], p2d[v, : ,0]] != 0 ] = 1

    occ_3d = occ_2d.prod(axis=0).reshape(32, 32, 32)
    occ_3d = occ_3d.detach().cpu().numpy()

    save_voxel(os.path.join(args.output_dir, 'gt_vox.obj'), voxel[0])
    save_voxel(os.path.join(args.output_dir, 'sil_vox.obj'), occ_3d)

    # Get outmost surface (display)
    # Subtract iosurface with marchine cube (display)

if __name__ == '__main__':
    main()
