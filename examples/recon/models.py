import torch
import torch.nn as nn
import torch.nn.functional as F

import soft_renderer as sr
import soft_renderer.functional as srf
import math
import pointnet

class Encoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024, im_size=64):
        super(Encoder, self).__init__()
        dim_hidden = [dim1, dim1*2, dim1*4, dim2, dim2]

        self.conv1 = nn.Conv2d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=2)

        self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = nn.BatchNorm2d(dim_hidden[2])

        self.fc1 = nn.Linear(dim_hidden[2]*math.ceil(im_size/8)**2, dim_hidden[3])
        self.fc2 = nn.Linear(dim_hidden[3], dim_hidden[4])
        self.fc3 = nn.Linear(dim_hidden[4], dim_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        return x


class Decoder(nn.Module):
    def __init__(self, nv, nf, dim_in=512, centroid_scale=0.1, bias_scale=1.0, centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()
        # load .obj
        # self.template_mesh = sr.Mesh.from_obj(filename_obj)
        # vertices_base, faces = srf.load_obj(filename_obj)

        self.nv = nv
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 1.0

        dim = 1024
        dim_hidden = [dim, dim*2]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)

        self.fc3 = nn.Linear(dim_in, dim_hidden[0])
        self.fc4 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_tex = nn.Linear(dim_hidden[1], 64 * 64 *3)

        # ngf = 128
        # self.texture_recon = nn.Sequential(
            # # input is Z, going into a convolution
            # nn.ConvTranspose2d(dim_in, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # # state size. (ngf*8) x 4 x 4
            # nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(True),
            # # state size. (ngf*4) x 8 x 8
            # nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            # nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            # nn.Tanh()
            # # state size. (nc) x 64 x 64
        # )

    def forward(self, x, vertices_base):
        batch_size = x.shape[0]
        xx = F.relu(self.fc1(x), inplace=True)
        xx = F.relu(self.fc2(xx), inplace=True)

        # decoder follows NMR
        centroid = self.fc_centroid(xx) * self.centroid_scale

        bias = self.fc_bias(xx) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 1.0

        xt = F.relu(self.fc3(x), inplace=True)
        xt = F.relu(self.fc4(xt), inplace=True)
        texture_maps = self.fc_tex(xt).view(-1, 64, 64, 3)
        # texture_maps = self.texture_recon(x.view(-1, 512, 1, 1))
        return vertices, texture_maps


class Model(nn.Module):
    def __init__(self, filename_obj, args):
        super(Model, self).__init__()

        # self.template_mesh = sr.Mesh.from_obj(filename_obj)
        # self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0])#vertices_base)
        # self.register_buffer('faces', self.template_mesh.faces.cpu()[0])#faces)
        # self.register_buffer('uvs', self.template_mesh.uvs.cpu()[0])#uvs)
        self.nv = 1400
        self.nf = 3000

        self.encoder = Encoder(im_size=args.image_size)
        self.decoder = Decoder(self.nv, self.nf, 512+1024)
        self.pointfeat = pointnet.PointNetfeat(global_feat=True)
        self.renderer = sr.SoftRenderer(image_size=args.image_size, sigma_val=args.sigma_val,
                                        aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=15,
                                        dist_eps=1e-10)
        # self.laplacian_loss = sr.LaplacianLoss(self.vertices_base, self.faces)
        self.laplacian_loss = sr.LaplacianLossBatch(True)
        # self.flatten_loss = sr.FlattenLoss(self.faces)
        # self.flatten_loss = sr.FlattenLossBatch()

    def model_param(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def set_sigma(self, sigma):
        self.renderer.set_sigma(sigma)

    def reconstruct(self, images, vertices, faces):
        batch_size = images.shape[0]
        image_feat = self.encoder(images)
        # point_feat, _, _ = self.pointfeat(self.vertices_base[None,:].repeat(batch_size,1,1).permute(0,2,1))
        point_feat, _, _ = self.pointfeat(vertices.permute(0,2,1))
        feat = torch.cat((image_feat, point_feat), 1)
        # vertices, texture_maps = self.decoder(feat, self.vertices_base)
        vertices, texture_maps = self.decoder(feat, vertices)
        # faces = self.faces[None, :, :].repeat(batch_size, 1, 1)
        # uvs = self.uvs[None, :].repeat(batch_size, 1, 1, 1)
        uvs = torch.zeros((batch_size, 1))

        return vertices, faces, uvs, texture_maps

    def get_area(self, face_vertices, mode=0):

        v0 = face_vertices[:,:,0,:].clone()
        v1 = face_vertices[:,:,1,:].clone()
        v2 = face_vertices[:,:,2,:].clone()

        if mode==1:
            v0[:,:,2] = 0
            v1[:,:,2] = 0
            v2[:,:,2] = 0
        area = torch.norm(torch.cross((v0 - v1), (v0 - v2)), dim=2)

        return area

    def area_loss(self, vertices, faces):
        fv = srf.face_vertices(vertices, faces)
        area = self.get_area(fv)
        return area.sum(1)

    def len_loss(self, vertices, faces):
        fv = srf.face_vertices(vertices, faces)
        length = self.get_len(fv)
        return length.sum(1)

    def predict_multiview(self, image_a, image_b, viewpoint_a, viewpoint_b, vertices, faces):
        batch_size = image_a.size(0)
        # [Ia, Ib]
        images = torch.cat((image_a, image_b), dim=0)
        # [Va, Va, Vb, Vb], set viewpoints
        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
        self.renderer.transform.set_eyes(viewpoints)
        self.renderer.transform2.set_eyes(viewpoints)

        vertices = torch.cat((vertices, vertices), dim=0)
        faces = torch.cat((faces, faces), dim=0)
        vertices, faces, uvs, texture_maps = self.reconstruct(images, vertices, faces)
        laplacian_loss = self.laplacian_loss(vertices, faces)
        # laplacian_loss = torch.zeros([])
        # flatten_loss = self.flatten_loss(vertices, faces)
        flatten_loss = torch.zeros([])
        # area_loss = self.area_loss(vertices, faces)
        area_loss = torch.zeros([])

        # [Ma, Mb, Ma, Mb]
        vertices = torch.cat((vertices, vertices), dim=0)
        faces = torch.cat((faces, faces), dim=0)
        uvs = torch.cat((uvs, uvs), dim=0)
        # texture_maps = torch.cat((texture_maps, texture_maps), dim=0).permute(0, 2, 3, 1)
        texture_maps = torch.cat((texture_maps, texture_maps), dim=0)

        # [Raa, Rba, Rab, Rbb], cross render multiview images
        silhouettes = self.renderer(vertices, faces, uvs=uvs, texture_maps=texture_maps)
        return silhouettes.chunk(4, dim=0), laplacian_loss, flatten_loss, area_loss

    def evaluate_iou(self, images, voxels):
        vertices, faces = self.reconstruct(images)

        faces_ = srf.face_vertices(vertices, faces).data
        faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
        voxels_predict = srf.voxelization(faces_norm, 32, False).cpu().numpy()
        voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
        iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))
        return iou, vertices, faces

    def forward(self, images=None, viewpoints=None, vertices=None, faces=None, voxels=None, task='train'):
        if task == 'train':
            return self.predict_multiview(images[0], images[1], viewpoints[0], viewpoints[1], vertices, faces)
        elif task == 'test':
            return self.evaluate_iou(images, voxels)
