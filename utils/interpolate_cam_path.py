import numpy as np
from scipy.spatial.transform import Rotation

# milano: img_id_list=[52, 144, 167, 722, 745, 749, 189], interpolate_times=[1, 1, 1, 1, 1, 1, 1]    [52, 722, 745]
# #st paul[1, 5]
# #notre_dame:img_id_list=[22,57, 61,2,3,90,91], interpolate_times=[72]*1000)
# [8,11,40,24,3519, 3695, 144,3570]         img_id_list=range(1000), interpolate_times=[3]*1000  [22,57, 61,2,3,90,2373,91]     [22,30,86,373,196,1529]
# 22,57, 61,2,3,90,91 [22,35,57,61,2,62,86,89,102,179]  img_id_list=[35,22,57,61,2,62,86,89,2920,3663,102,179], interpolate_times=[36]*3700
# portals - notre: [57,71,2,131] 22  [22,71,2,131]   [10]*3700



# notre - portals: [22,71] , notre - towers: [22, 102] notre - rose: [22, 1686] interpolate_times=[24,0]    # milano- portal:img_id_list= [40,558]
#       towers - [83,81]       blue mosque -minarets -  [383,103]
# st_pual - pediment/colonnade
# st paul - [41,168]
# 639
# blue mosque - windows:
def generate_camera_path(dataset, img_id_list= [40,558], interpolate_times=[24,1]):
    '''
    args:
        img_id_list: IDs of key-frame images
        interpolate_times: in each segment, how many frames should be interpolated
    '''
    N_keyframes = len(img_id_list)
    pose_list = []
    for index, id in enumerate(img_id_list):
        pose = dataset.poses_dict[id]
        # The camera pose is defined in 4x4 matrix.
        pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])))  # (1, 4, 4)

        # # st paul pediment - with 2*pi
        # if id == 331:
        #     z = pose[:3, 3][2]
        #     # z += 0.1 #0.05
        #     z += 0.3 #0.05
        #     y = pose[:3, 3][1]
        #     y -= 0.1
        #     x = pose[:3, 3][0]
        #     x -= 0.02
        #     pose[:3, 3][2] = z
        #     pose[:3, 3][1] = y
        #     pose[:3, 3][0] = x

        pose_list.append(pose)

    pose_interp_list = []
    for index in range(len(pose_list)):
        pose = pose_list[index]
        pose_next = pose_list[(index + 1) % N_keyframes]
        r = Rotation.from_matrix(pose[:3, : 3])
        r_n = Rotation.from_matrix(pose_next[:3, : 3])
        eulers = r.as_euler('xyz')
        eulers_n = r_n.as_euler('xyz')

        # if eulers[0] > 0 or eulers_n[0] > 0:
        #     continue

        # if eulers[0] < 0 or eulers_n[0] < 0:
        #     continue

        print(index)


        rx = np.linspace(eulers[0], eulers_n[0], num=interpolate_times[index], endpoint=False)
        # rx = np.linspace(eulers[0], eulers_n[0] + 2 * np.pi, num=interpolate_times[index], endpoint=False)

        ry = np.linspace(eulers[1], eulers_n[1], num=interpolate_times[index], endpoint=False)
        rz = np.linspace(eulers[2], eulers_n[2], num=interpolate_times[index], endpoint=False)
        eulers_list = np.concatenate((rx[:, np.newaxis], ry[:, np.newaxis], rz[:, np.newaxis]), axis=1)  # []
        r_list = Rotation.from_euler('xyz', eulers_list).as_matrix()

        positions = pose[:3, 3]
        positions_n = pose_next[:3, 3]
        px = np.linspace(positions[0], positions_n[0], num=interpolate_times[index], endpoint=False)
        py = np.linspace(positions[1], positions_n[1], num=interpolate_times[index], endpoint=False)
        pz = np.linspace(positions[2], positions_n[2], num=interpolate_times[index], endpoint=False)
        positions_list = np.concatenate((px[:, np.newaxis], py[:, np.newaxis], pz[:, np.newaxis]), axis=1)
        pose_l = np.concatenate((r_list, positions_list[:, :, np.newaxis]), axis=2)
        pose_interp_list.append(pose_l)

    pose_interp_list = np.concatenate(pose_interp_list, axis=0)
    return pose_interp_list