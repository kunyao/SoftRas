import numpy as np
from tqdm import tqdm
import pymesh

in_file = './data/datasets/02828884_train_meshes.npz'
out_file = './data/datasets/02828884_train_meshes_clean2.npz'
mesh_dict = np.load(in_file)
vertices = mesh_dict['v']
faces = mesh_dict['f']

vertices_clean = np.zeros_like(vertices)
faces_clean = np.zeros_like(faces)

for i in tqdm(range(vertices.shape[0])):
    mesh = pymesh.form_mesh(vertices[i], faces[i])
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    mesh, _ = pymesh.remove_duplicated_faces(mesh)
    mesh, _ = pymesh.remove_isolated_vertices(mesh)
    mesh, _ = pymesh.remove_degenerated_triangles(mesh)
    mesh, _ = pymesh.collapse_short_edges(mesh, rel_threshold=0.3)
    meshes = pymesh.separate_mesh(mesh)
    if len(meshes) == 0:
        continue
    mesh = sorted([(m.elements.shape[0], m) for m in meshes], key=lambda x: x[0])[-1][1]

    vertices_clean[i, :mesh.vertices.shape[0], :mesh.vertices.shape[0]] = mesh.vertices
    faces_clean[i, :mesh.faces.shape[0], :mesh.faces.shape[0]] = mesh.faces

np.savez(out_file, v=vertices_clean, f=faces_clean)
