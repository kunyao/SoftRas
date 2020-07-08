
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

import soft_renderer as sr
from rasterize_op import FragmentRasterize


class Renderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100,
                 anti_aliasing=True, fill_back=True, eps=1e-6,
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0,
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
        super(Renderer, self).__init__()

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)

        # camera
        self.transform = sr.Transform(camera_mode,
                                      P, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale,
                                      eye, camera_direction)

        # rasterization
        self.rasterizer = sr.Rasterizer(image_size, background_color, near, far,
                                        anti_aliasing, fill_back, eps)

    def forward(self, mesh, mode=None):
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        return self.rasterizer(mesh, mode)


class SoftRenderer(nn.Module):
    def __init__(self, image_size=256, background_color=[0,0,0], near=1, far=100,
                 anti_aliasing=False, fill_back=True, eps=1e-3,
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='hard', aggr_func_alpha='prod',
                 texture_type='surface',
                 camera_mode='projection',
                 P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0,
                 eye=None, camera_direction=[0,0,1],
                 light_mode='surface',
                 light_intensity_ambient=0.5, light_color_ambient=[1,1,1],
                 light_intensity_directionals=0.5, light_color_directionals=[1,1,1],
                 light_directions=[0,1,0]):
        super(SoftRenderer, self).__init__()

        # light
        self.lighting = sr.Lighting(light_mode,
                                    light_intensity_ambient, light_color_ambient,
                                    light_intensity_directionals, light_color_directionals,
                                    light_directions)

        # camera
        self.transform = sr.Transform(camera_mode,
                                      P, dist_coeffs, orig_size,
                                      perspective, viewing_angle, viewing_scale,
                                      eye, camera_direction)

        self.transform2 = sr.Transform(camera_mode,
                                      P, dist_coeffs, orig_size,
                                      False, viewing_angle, viewing_scale,
                                      eye, camera_direction)

        # rasterization
        self.rasterizer = sr.SoftRasterizer(image_size, background_color, near, far,
                                            anti_aliasing, fill_back, eps,
                                            sigma_val, dist_func, dist_eps,
                                            gamma_val, aggr_func_rgb, aggr_func_alpha,
                                            texture_type)
        self.set_alpha()
        self.set_win_size()
        self.set_beta()

    def set_sigma(self, sigma):
        self.rasterizer.sigma_val = sigma

    def set_gamma(self, gamma):
        self.rasterizer.gamma_val = gamma

    def set_win_size(self, win_size=40):
        self.win_size = win_size

    def set_alpha(self, alpha=2.0):
        self.alpha = alpha

    def set_beta(self, beta=0):
        self.beta = beta

    def set_texture_mode(self, mode):
        assert mode in ['vertex', 'surface'], 'Mode only support surface and vertex'

        self.lighting.light_mode = mode
        self.rasterizer.texture_type = mode

    def get_area(self, mesh, mode=0):

        v0 = mesh.face_vertices[:,:,0,:].clone()
        v1 = mesh.face_vertices[:,:,1,:].clone()
        v2 = mesh.face_vertices[:,:,2,:].clone()

        if mode==1:
            v0[:,:,2] = 0
            v1[:,:,2] = 0
            v2[:,:,2] = 0
        area = torch.norm(torch.cross((v0 - v1), (v0 - v2)), dim=2)

        return area

    def render_mesh(self, mesh,  mode=None, use_soft=False):
        self.set_texture_mode(mesh.texture_type)
        mesh = self.lighting(mesh)
        mesh_copy = sr.Mesh(mesh.vertices, mesh.faces)

        area_3d = self.get_area(mesh_copy)
        mesh_copy = self.transform2(mesh_copy)
        area_2d = self.get_area(mesh_copy, 1)

        mesh = self.transform(mesh)

        if use_soft:  # use for comparison
            return self.rasterizer(mesh, mode)
        else:
            return FragmentRasterize.apply(mesh.face_vertices, area_2d / area_3d, torch.zeros(32, 64, 64).cuda(), 64, self.win_size, 50, 0.1, 25.0)[:, None, :, :].repeat(1, 4, 1, 1)

    def forward(self, vertices, faces, textures=None, mode=None, texture_type='surface', use_soft=False):
        mesh = sr.Mesh(vertices, faces, textures=textures, texture_type=texture_type)
        return self.render_mesh(mesh, mode, use_soft=use_soft)
