"""
Using open3d to visualize predicted mesh
"""
import numpy as np
import open3d as o3d
import pickle
from transforms3d.axangles import axangle2mat
from utils.util import OneEuroFilter, add_arm_vertices


def Worker(inputs_queue, output_queue, proc_id, window_size, ks, R=None):
    with open("./mano/MANO_RIGHT.pkl", 'rb') as f:
        righthand_model = pickle.load(f, encoding='latin1')
        righthand_vertices = np.array(righthand_model['v_template'], dtype=np.float)
        right_faces = righthand_model['f'].astype(np.int)
    with open("./mano/MANO_LEFT.pkl", 'rb') as f:
        lefthand_model = pickle.load(f, encoding='latin1')
        lefthand_vertices = np.array(lefthand_model['v_template'], dtype=np.float)
        left_faces = lefthand_model['f'].astype(np.int)
    # righthand_vertices, right_faces = add_arm_vertices(righthand_vertices, right_faces, 5)
    # lefthand_vertices, left_faces = add_arm_vertices(lefthand_vertices, left_faces, 5)
    faces = np.concatenate((right_faces, left_faces + len(righthand_vertices)), 0)
    vertices = np.concatenate((righthand_vertices, lefthand_vertices), 0)

    view_mat = axangle2mat([1, 0, 0], np.pi)
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, vertices.T).T)
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
        width=window_size + 1, height=window_size + 1,
        window_name="Hand Mesh")
    viewer.add_geometry(mesh)

    view_control = viewer.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = cam_params.extrinsic.copy()
    extrinsic[:3, 3] = 0
    cam_params.extrinsic = extrinsic
    cam_params.intrinsic.set_intrinsics(
        window_size + 1, window_size + 1, ks[0, 0], ks[1, 1],
        window_size // 2, window_size // 2)
    view_control.convert_from_pinhole_camera_parameters(cam_params)
    view_control.set_constant_z_far(1000)

    render_option = viewer.get_render_option()
    render_option.load_from_json('./mano/render_option.json')
    viewer.update_renderer()

    mesh_smoother = OneEuroFilter(4.0, 0.0)
    verts = vertices
    while True:
        if not inputs_queue.empty():
            print(inputs_queue.qsize())
            message = inputs_queue.get()
            if message == 'STOP':
                print("Quit 3D mesh displayer")
                break
            else:
                verts = message
                # verts = verts * 1000
                print(verts.shape)
                # L = 778
                # verts[:L, :] -= verts[0]
                # verts[L:, 0] = -verts[L:, 0]
                # verts[L:, :] -= verts[L] + 0.05
                verts[:, :] -= verts[0]
                verts = mesh_smoother.process(verts)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, verts.T).T)
                # mesh.vertices = o3d.utility.Vector3dVector(verts)
                mesh.paint_uniform_color([228/255, 178/255, 148/255])
                mesh.compute_triangle_normals()
                mesh.compute_vertex_normals()
                viewer.update_geometry(mesh)

                viewer.poll_events()
        else:
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, verts.T).T)
            # mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.paint_uniform_color([228/255, 178/255, 148/255])
            mesh.compute_triangle_normals()
            mesh.compute_vertex_normals()
            viewer.update_geometry(mesh)

            viewer.poll_events()
    output_queue.put('STOP')
