import os

import soft_renderer.functional as srf
import torch
import numpy as np
import tqdm
from scipy import ndimage

class_ids_map = {
    '02691156': 'Airplane',
    '02828884': 'Bench',
    '02933112': 'Cabinet',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03211117': 'Display',
    '03636649': 'Lamp',
    '03691459': 'Loudspeaker',
    '04090263': 'Rifle',
    '04256520': 'Sofa',
    '04379243': 'Table',
    '04401088': 'Telephone',
    '04530566': 'Watercraft',
}

class ShapeNet(object):
    def __init__(self, directory=None, class_ids=None, set_name=None, load_template=False, load_camera=False):
        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732

        self.class_ids_map = class_ids_map

        images = []
        voxels = []
        vertices = []
        faces = []
        camera_matrix = []
        self.num_data = {}
        self.pos = {}
        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            images.append(list(np.load(
                os.path.join(directory, '%s_%s_images_137.npz' % (class_id, set_name))).items())[0][1])
                # os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))).items())[0][1])
            voxels.append(list(np.load(
                os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name))).items())[0][1])
            if load_template:
                vertices.append(np.load(
                    os.path.join(directory, '%s_%s_meshes_depth137_3000.npz' % (class_id, set_name)))['v'])
                faces.append(np.load(
                    os.path.join(directory, '%s_%s_meshes_depth137_3000.npz' % (class_id, set_name)))['f'])
            if load_camera:
                camera_matrix.append(np.load(
                    os.path.join(directory, '%s_%s_cam_137.npz' % (class_id, set_name)))['c'])
                    # os.path.join(directory, '%s_%s_cam.npz' % (class_id, set_name)))['c'])

            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]

        # concatenate subclasses
        W = np.array(images).shape[-1]
        H = np.array(images).shape[-2]
        images = np.concatenate(images, axis=0).reshape((-1, 4, H, W))
        images = np.ascontiguousarray(images)
        self.images = images
        self.voxels = np.ascontiguousarray(np.concatenate(voxels, axis=0))
        self.vertices = np.ascontiguousarray(np.concatenate(vertices, axis=0))
        self.faces = np.ascontiguousarray(np.concatenate(faces, axis=0))
        self.camera_matrix = np.ascontiguousarray(np.concatenate(camera_matrix, axis=0).reshape(-1, 3, 4))
        del images
        del voxels
        del vertices
        del faces
        del camera_matrix

    @property
    def class_ids_pair(self):
        class_names = [self.class_ids_map[i] for i in self.class_ids]
        return zip(self.class_ids, class_names)

    def get_random_batch(self, batch_size):
        '''
        images: obj_num*24 --> batch_size [a, b]
        vertices: obj_num --> batch_size
        cameras: obj*24 --> batch_size [a, b]
        '''
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        object_global_id = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        viewpoint_ids_b = torch.zeros(batch_size)
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])

            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_id_b = (object_id + self.pos[class_id]) * 24 + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            object_global_id[i] = object_id + self.pos[class_id]
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

        images_a = torch.from_numpy(self.images[data_ids_a].astype('float32') / 255.)
        images_b = torch.from_numpy(self.images[data_ids_b].astype('float32') / 255.)

        distances = torch.ones(batch_size).float() * self.distance
        elevations_a = torch.ones(batch_size).float() * self.elevation
        elevations_b = torch.ones(batch_size).float() * self.elevation
        viewpoints_a = srf.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 15)
        viewpoints_b = srf.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b * 15)

        vertices_batch = torch.from_numpy(self.vertices[object_global_id].astype('float32'))
        faces_batch = torch.from_numpy(self.faces[object_global_id].astype('float32'))

        return images_a, images_b, viewpoints_a, viewpoints_b, vertices_batch, faces_batch

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids

        distances = torch.ones(data_ids.size).float() * self.distance
        elevations = torch.ones(data_ids.size).float() * self.elevation
        viewpoints_all = srf.get_points_from_angles(distances, elevations, -torch.from_numpy(viewpoint_ids).float() * 15)

        for i in range((data_ids.size - 1) // batch_size + 1):
            images = torch.from_numpy(self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.)
            voxels = torch.from_numpy(self.voxels[data_ids[i * batch_size:(i + 1) * batch_size] // 24].astype('float32'))
            yield images, voxels

    def get_one_obj(self, obj_id, load_template=False, load_camera=False):
        '''
        Get multiview data of one spcific object in the dataset, including:
        2D target: images N*24 --> 24
        3D target: voxels N --> 24
        3D template: template_v, template_f N --> 1
        camera: camera_matrix N*24 --> 24
        '''
        num_views = 24

        data_ids = np.arange(num_views) + obj_id * 24
        distances = torch.ones(num_views).float() * self.distance
        elevations = torch.ones(num_views).float() * self.elevation
        viewpoints = -torch.from_numpy(np.arange(num_views)).float() * 15


        images = self.images[data_ids].astype('float32') / 255.
        images = torch.from_numpy(images)

        ''' Uncomment this for distance map
        masks = images[:, 3, :, :].copy()
        masks[masks > 0] = 1
        masks = 1 - masks
        dists = []
        for m in masks:
            d = ndimage.distance_transform_edt(m)
            dists.append(d)

        dists = np.array(dists).astype('float32')
        dists = torch.from_numpy(dists)
        '''

        voxels = torch.from_numpy(self.voxels[data_ids // 24].astype('float32'))
        dists = torch.zeros([])

        if load_template:
            template_v = torch.from_numpy(self.vertices[obj_id].astype('float32'))
            template_f = torch.from_numpy(self.faces[obj_id].astype('float32'))
        if load_camera:
            camera_matrix = torch.from_numpy(self.camera_matrix[data_ids].astype('float32'))

        return images, dists, voxels, distances, elevations, viewpoints, template_v, template_f, camera_matrix
