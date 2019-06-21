import copy
import numpy as np
from open3d import *

if __name__ == "__main__":

    # generate some neat n times 3 matrix using a variant of sync function
    x = np.linspace(0, 0.05, 100)
    y = np.linspace(0, 0.025, 50)

    mesh_x, mesh_y = np.meshgrid(x, y)

    xyz = np.zeros((np.size(mesh_x),3))
    xyz[:,0] = np.reshape(mesh_x,-1)
    xyz[:,1] = np.reshape(mesh_y,-1)
    xyz[:,2] = [0]


    y = np.linspace(0, 0.025, 50)
    z = np.linspace(0, 0.025, 50)

    mesh_y, mesh_z = np.meshgrid(y, z)

    xyz_ = np.zeros((np.size(mesh_y),3))
    xyz_[:,0] = [0]
    xyz_[:,1] = np.reshape(mesh_y,-1)
    xyz_[:,2] = np.reshape(mesh_z,-1)

    xyz = np.concatenate((xyz, xyz_))


    y = np.linspace(0, 0.025, 50)
    z = np.linspace(0, 0.025, 50)

    mesh_y, mesh_z = np.meshgrid(y, z)

    xyz_ = np.zeros((np.size(mesh_y),3))
    xyz_[:,0] = [0.05]
    xyz_[:,1] = np.reshape(mesh_y,-1)
    xyz_[:,2] = np.reshape(mesh_z,-1)

    xyz = np.concatenate((xyz, xyz_))

    x = np.linspace(0, 0.05, 50)
    z = np.linspace(0, 0.025, 50)

    mesh_x, mesh_z = np.meshgrid(x, z)

    xyz_ = np.zeros((np.size(mesh_x),3))
    xyz_[:,0] = np.reshape(mesh_x,-1)
    xyz_[:,1] = [0]
    xyz_[:,2] = np.reshape(mesh_z,-1)

    xyz = np.concatenate((xyz, xyz_))

    x = np.linspace(0, 0.05, 50)
    z = np.linspace(0, 0.025, 50)

    mesh_x, mesh_z = np.meshgrid(x, z)

    xyz_ = np.zeros((np.size(mesh_x),3))
    xyz_[:,0] = np.reshape(mesh_x,-1)
    xyz_[:,1] = [0.025]
    xyz_[:,2] = np.reshape(mesh_z,-1)

    xyz = np.concatenate((xyz, xyz_))

    x = np.linspace(0, 0.025, 50)    
    z = np.linspace(0.025, 0.05, 50)

    mesh_x, mesh_z = np.meshgrid(x, z)

    xyz_ = np.zeros((np.size(mesh_x),3))
    xyz_[:,0] = np.reshape(mesh_x,-1)
    xyz_[:,1] = [0.025]
    xyz_[:,2] = np.reshape(mesh_z,-1)

    xyz = np.concatenate((xyz, xyz_))

    x = np.linspace(0, 0.025, 50)    
    z = np.linspace(0.025, 0.05, 50)

    mesh_x, mesh_z = np.meshgrid(x, z)

    xyz_ = np.zeros((np.size(mesh_x),3))
    xyz_[:,0] = np.reshape(mesh_x,-1)
    xyz_[:,1] = [0]
    xyz_[:,2] = np.reshape(mesh_z,-1)

    xyz = np.concatenate((xyz, xyz_))

    y = np.linspace(0, 0.025, 50)    
    z = np.linspace(0.025, 0.05, 50)

    mesh_y, mesh_z = np.meshgrid(y, z)

    xyz_ = np.zeros((np.size(mesh_x),3))
    xyz_[:,0] = [0]
    xyz_[:,1] = np.reshape(mesh_y,-1)
    xyz_[:,2] = np.reshape(mesh_z,-1)

    xyz = np.concatenate((xyz, xyz_))

    y = np.linspace(0, 0.025, 50)    
    z = np.linspace(0.025, 0.05, 50)

    mesh_y, mesh_z = np.meshgrid(y, z)

    xyz_ = np.zeros((np.size(mesh_x),3))
    xyz_[:,0] = [0.025]
    xyz_[:,1] = np.reshape(mesh_y,-1)
    xyz_[:,2] = np.reshape(mesh_z,-1)

    xyz = np.concatenate((xyz, xyz_))

    x = np.linspace(0, 0.025, 50)    
    y = np.linspace(0, 0.025, 50)

    mesh_x, mesh_y = np.meshgrid(x, y)

    xyz_ = np.zeros((np.size(mesh_x),3))
    xyz_[:,0] = np.reshape(mesh_x,-1)
    xyz_[:,1] = np.reshape(mesh_y,-1)
    xyz_[:,2] = [0.05]

    xyz = np.concatenate((xyz, xyz_))

    x = np.linspace(0.025, 0.05, 50)    
    y = np.linspace(0, 0.025, 50)

    mesh_x, mesh_y = np.meshgrid(x, y)

    xyz_ = np.zeros((np.size(mesh_x),3))
    xyz_[:,0] = np.reshape(mesh_x,-1)
    xyz_[:,1] = np.reshape(mesh_y,-1)
    xyz_[:,2] = [0.025]

    xyz = np.concatenate((xyz, xyz_))


    # Pass xyz to Open3D.PointCloud and visualize
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    write_point_cloud("surf_sync.pcd", pcd)

    # Load saved point cloud and visualize it
    pcd_load = read_point_cloud("surf_sync.pcd")
    draw_geometries([pcd_load])

    # convert Open3D.PointCloud to numpy array
    # xyz_load = np.asarray(pcd_load.points)
    # print('xyz_load')
    # print(xyz_load)

    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    # img = Image((z_norm*255).astype(np.uint8))
    # write_image("sync.png", img)
    # draw_geometries([img])

    # Pass xyz to Open3D.PointCloud and visualize
    # pcd = PointCloud()
    # pcd.points = Vector3dVector(xyz)
    # write_point_cloud("sync.ply", pcd)
    
    # Let's make surface!
