"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
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
# 02958343: car
# 02691156: airplane
# 02828884: bench
class_ids = '02958343'
set_select = 'train'
# class_ids = (
    # '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    # '03691459,04090263,04256520,04379243,04401088,04530566')
dataset = datasets.ShapeNet(dataset_directory, class_ids.split(','), set_select)

class Model(nn.Module):
    def __init__(self, template_path):
        super(Model, self).__init__()

        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(template_path)
        self.register_buffer('vertices', self.template_mesh.vertices * 0.5)
        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())

    def forward(self, batch_size):
        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))
        centroid = torch.tanh(self.center)
        vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid

        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()
        # laplacian_loss = torch.tensor(0)
        # flatten_loss = torch.tensor(0)

        return sr.Mesh(vertices.repeat(batch_size, 1, 1),
                       self.faces.repeat(batch_size, 1, 1)), laplacian_loss, flatten_loss


def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()


def iou_3d(faces, voxels):
    '''
    [-0.5, 0.5] --> [0.5 / res, 1 - 0.5 / res]
    '''
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

def diff_voxel(filename0, filename1, voxel0, voxel1):
    vertices0 = []
    vertices1 = []
    res_x = voxel0.shape[0]
    res_y = voxel0.shape[1]
    res_z = voxel0.shape[2]
    for i in range(res_x):
        for j in range(res_y):
            for k in range(res_z):
                X = j / res_y - 0.5 * (res_y - 1) / res_y
                Y = i / res_x - 0.5 * (res_x - 1) / res_x
                Z = -k / res_z + 0.5 * (res_z - 1) / res_z
                if voxel0[i, j, k] == 0 and voxel1[i, j, k] == 1:
                    vertices0.append([X, Y, Z])
                if voxel0[i, j, k] == 1 and voxel1[i, j, k] == 0:
                    vertices1.append([X, Y, Z])
    save_obj(filename0, vertices0)
    save_obj(filename1, vertices1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str,
        default=os.path.join(data_dir, 'source.npy'))
    parser.add_argument('-c', '--camera-input', type=str,
        default=os.path.join(data_dir, 'camera.npy'))
    parser.add_argument('-t', '--template-mesh', type=str,
        default=os.path.join(data_dir, 'obj/sphere/sphere_1352.obj'))
    parser.add_argument('-o', '--output-dir', type=str,
        default=os.path.join(data_dir, 'results/output_deform3'))
    parser.add_argument('-b', '--batch-size', type=int,
        default=120)
    parser.add_argument('-s', '--use_soft', action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = Model(args.template_mesh).cuda()
    renderer = sr.SoftRenderer(image_size=64, sigma_val=1e-4, aggr_func_rgb='hard',
                               camera_mode='look_at', viewing_angle=15)

    # read training images and camera poses
    images_gt, dist_maps, voxel, camera_distances, elevations, viewpoints = dataset.get_one_model(5)
    voxel = voxel.numpy()
    images_gt = images_gt.cuda()
    dist_maps = dist_maps.cuda()

    # Uncomment for hard boundary
    images_gt[images_gt >= 0.5] = 1.0
    images_gt[images_gt < 0.5] = 0.0
    #

    ''' Uncomment for viewpoints slection
    images_gt = images_gt[0:24,:]
    dist_maps = dist_maps[0:24,:]
    camera_distances = camera_distances[0:24]
    elevations = elevations[0:24]
    viewpoints = viewpoints[0:24]
    '''

    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))

    renderer.transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

    loop = tqdm.tqdm(list(range(0, 401)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'deform.gif'), mode='I')

    f2d = open(os.path.join(args.output_dir, '2d_iou.txt'), 'w')
    f3d = open(os.path.join(args.output_dir, '3d_iou.txt'), 'w')

    for i in loop:
        mesh, laplacian_loss, flatten_loss = model(24)
        iou3D, voxels_predict = iou_3d(mesh.face_vertices, voxel)
        images_pred = renderer.render_mesh(mesh, use_soft=args.use_soft)

        # optimize mesh with silhouette reprojection error and
        # geometry constraints
        loss = neg_iou_loss(images_pred[:, 3], images_gt[:, 3]) + \
               0.03 * laplacian_loss + \
               0.0003 * flatten_loss

        images_hard = images_pred.clone()
        # images_hard[images_hard < 0.99] = 0
        # images_hard[images_hard >= 0.99] = 1.0
        hard_iou = neg_iou_loss(images_hard[:, 3], images_gt[:, 3])

        loop.set_description('2D IoU: {}, 3D IoU: {}'.format(1 - hard_iou.item(), iou3D[0].item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            image = images_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
            writer.append_data((255*image).astype(np.uint8))
            imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png'%i), (255*image[..., -1]).astype(np.uint8))
            image_out = torch.cat((images_hard.detach()[:, 3][:, None, :, :], images_gt.detach()[:, 3][:, None, :, :].repeat(1, 2, 1, 1)), dim=1)
            torchvision.utils.save_image(image_out, os.path.join(args.output_dir, 'views_%05d.png'%i))
            model(1)[0].save_obj(os.path.join(args.output_dir, 'plane%05d.obj'%i), save_texture=False)
            f2d.write("{} {}\n".format(i, 1.0 - hard_iou.item()))
            f3d.write("{} {}\n".format(i, iou3D[0].item()))


    # save optimized mesh
    # model(1)[0].save_obj(os.path.join(args.output_dir, 'plane.obj'), save_texture=False)
    save_voxel(os.path.join(args.output_dir, 'target.obj'), voxel[0])
    save_voxel(os.path.join(args.output_dir, 'predict.obj'), voxels_predict[0])
    diff_voxel(os.path.join(args.output_dir, 'more.obj'), os.path.join(args.output_dir, 'less.obj'), voxel[0], voxels_predict[0])

    with imageio.get_writer(os.path.join(args.output_dir, 'views.gif'), mode='I') as writer:
        for filename in sorted(glob.glob(os.path.join(args.output_dir, 'views_*.png'))):
            writer.append_data(imageio.imread(filename))
    writer.close()


if __name__ == '__main__':
    main()
