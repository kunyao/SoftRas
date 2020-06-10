import torch
import torch.nn as nn
import numpy as np


class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).sum(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x
        
class FlattenLoss(nn.Module):
    def __init__(self, faces, average=False):
        super(FlattenLoss, self).__init__()
        self.nf = faces.size(0)
        self.average = average
        
        faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []
        for v0, v1 in zip(v0s, v1s):
            count = 0
            for face in faces:
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        dims = tuple(range(cos.ndimension())[1:])
        loss = (cos + 1).pow(2).sum(dims)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss

class LaplacianLossBatch(nn.Module):
    def __init__(self, average=False):
        super(LaplacianLossBatch, self).__init__()
        self.average = average

    def forward(self, vertices, faces):
        self.nb = vertices.size(0)
        self.nv = vertices.size(1)
        faces = faces.long()

        laplacian = torch.zeros([self.nb, self.nv, self.nv]).cuda()

        loss = 0
        for i in range(self.nb):

            laplacian[i, faces[i, :, 0], faces[i, :, 1]] = -1
            laplacian[i, faces[i, :, 1], faces[i, :, 0]] = -1
            laplacian[i, faces[i, :, 1], faces[i, :, 2]] = -1
            laplacian[i, faces[i, :, 2], faces[i, :, 1]] = -1
            laplacian[i, faces[i, :, 2], faces[i, :, 0]] = -1
            laplacian[i, faces[i, :, 0], faces[i, :, 2]] = -1

            r, c = np.diag_indices(laplacian[i, :].shape[0])
            laplacian[i, :][r, c] = -laplacian[i, :].sum(1)

            num_neighbour = torch.diag(laplacian[i, :])
            num_neighbour[num_neighbour == 0] = 1
            laplacian[i, :] /= num_neighbour

            x = laplacian[i, :].clone()
            v_avg = torch.matmul(x, vertices[i, :])
            dims = tuple(range(v_avg.ndimension()))
            v_avg = v_avg.pow(2).sum(dims)
            loss += v_avg

        # if self.average:
            # return x.sum() / self.nb
        # else:
        return loss

class FlattenLossBatch(nn.Module):
    def __init__(self, average=False):
        super(FlattenLossBatch, self).__init__()
        self.average = average

    def forward(self, vertices, faces, eps=1e-6):
        self.nb = faces.size(0)
        self.nf = faces.size(1)

        loss = 0
        faces = faces.detach().cpu().numpy()
        for i in range(self.nb):
            vertices_id = list(set([tuple(v) for v in np.sort(np.concatenate((faces[i, :, 0:2], faces[i, :, 1:3]), axis=0))]))

            v0s = np.array([v[0] for v in vertices_id], 'int32')
            v1s = np.array([v[1] for v in vertices_id], 'int32')
            v2s = []
            v3s = []
            for v0, v1 in zip(v0s, v1s):
                count = 0
                # if v0 == v1:
                    # continue
                for face in faces:
                    if v0 in face and v1 in face:
                        v = np.copy(face)
                        v = v[v != v0]
                        v = v[v != v1]
                        if count == 0:
                            v2s.append(int(v[0]))
                            count += 1
                        elif count == 1:
                            v3s.append(int(v[0]))
                            count += 1
                        else:
                            continue
            v2s = np.array(v2s, 'int32')
            v3s = np.array(v3s, 'int32')

            # make v0s, v1s, v2s, v3s
            v0s = torch.from_numpy(v0s).long().cuda()
            v1s = torch.from_numpy(v1s).long().cuda()
            v2s = torch.from_numpy(v2s).long().cuda()
            v3s = torch.from_numpy(v3s).long().cuda()

            batch_size = vertices.size(0)

            v0s = vertices[i, v0s, :]
            v1s = vertices[i, v1s, :]
            v2s = vertices[i, v2s, :]
            v3s = vertices[i, v3s, :]

            a1 = v1s - v0s
            b1 = v2s - v0s
            a1l2 = a1.pow(2).sum(-1)
            b1l2 = b1.pow(2).sum(-1)
            a1l1 = (a1l2 + eps).sqrt()
            b1l1 = (b1l2 + eps).sqrt()
            ab1 = (a1 * b1).sum(-1)
            cos1 = ab1 / (a1l1 * b1l1 + eps)
            sin1 = (1 - cos1.pow(2) + eps).sqrt()
            c1 = a1 * (ab1 / (a1l2 + eps))[:, None]
            cb1 = b1 - c1
            cb1l1 = b1l1 * sin1

            a2 = v1s - v0s
            b2 = v3s - v0s
            a2l2 = a2.pow(2).sum(-1)
            b2l2 = b2.pow(2).sum(-1)
            a2l1 = (a2l2 + eps).sqrt()
            b2l1 = (b2l2 + eps).sqrt()
            ab2 = (a2 * b2).sum(-1)
            cos2 = ab2 / (a2l1 * b2l1 + eps)
            sin2 = (1 - cos2.pow(2) + eps).sqrt()
            c2 = a2 * (ab2 / (a2l2 + eps))[:, None]
            cb2 = b2 - c2
            cb2l1 = b2l1 * sin2

            cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

            dims = tuple(range(cos.ndimension())[1:])
            loss += (cos + 1).pow(2).sum(dims)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss
