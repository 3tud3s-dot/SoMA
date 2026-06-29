import os
import numpy as np
import json
import pickle
from urdfpy import URDF
from urdfpy.utils import matrix_to_rpy, rpy_to_matrix

# 兼容 urdfpy 旧版本对 np.float 的引用
np.float = float
scale = 1.535 #1.362


def compute_eef_pose_from_qpos(
    robot_dir,
    qpos_array,
    link_name,
    scale=1.0,
    use_gripper=False,
    ref_link='body4',
    hand_bias=0.071,
    default_dir='x'
):
    """
    qpos_array: (T, 14) 关节角（左7+右7）
    link_name : 末端链接名，如 'left_link18' 或 'right_link28'
    scale     : 尺寸缩放，作用在整机几何（各关节原点平移）上
    use_gripper: True 则左/右各取前7维；False 则只取前6维（去掉夹爪）
    ref_link  : 参考坐标系，默认 'body4'
    hand_bias : 沿默认朝向的偏移距离（会随 scale 一起缩放）
    default_dir: 默认朝向轴（'x'/'y'/'z'）
    返回:
        eef_pos        : (T, 3) 末端位置，ref_link 系
        eef_rpy        : (T, 3) RPY 姿态，ref_link 系
        eef_pos_final  : (T, 3) 末端沿默认朝向偏移 hand_bias 后的位置，ref_link 系
    """
    urdf_path = os.path.join(robot_dir, 'lift2_collision.urdf')
    urdf_model = URDF.load(urdf_path)

    actuated = [j.name for j in urdf_model.actuated_joints]
    left_names_full = [n for n in actuated if 'left' in n]
    right_names_full = [n for n in actuated if 'right' in n]

    dof = 7 if use_gripper else 6
    left_names = left_names_full[:dof]
    right_names = right_names_full[:dof]
    left_slice = slice(0, dof)
    right_slice = slice(7, 7 + dof)

    link_obj = urdf_model.link_map[link_name]
    ref_obj = urdf_model.link_map[ref_link]
    use_names = right_names if 'right' in link_name.lower() else left_names
    use_slice = right_slice if 'right' in link_name.lower() else left_slice

    eef_pos, eef_rpy, eef_pos_final = [], [], []
    for joints in qpos_array:
        joint_vals = joints[use_slice]
        cfg = dict(zip(use_names, joint_vals))
        fk_dict = urdf_model.link_fk(cfg=cfg)

        # 对几何做统一缩放：仅缩放平移，不改旋转
        T_base_eef = fk_dict[link_obj].copy()
        T_base_ref = fk_dict[ref_obj].copy()
        T_base_eef[:3, 3] *= scale
        T_base_ref[:3, 3] *= scale

        # 计算 ref_link → eef
        T_ref_base = np.linalg.inv(T_base_ref)
        T_ref_eef = T_ref_base @ T_base_eef

        pos = T_ref_eef[:3, 3]
        rpy = matrix_to_rpy(T_ref_eef[:3, :3])

        # 偏移朝向：默认局部轴映射到 ref，再沿该轴偏 hand_bias（也随 scale 缩放）
        axis_map = {'x': np.array([1, 0, 0]), 'y': np.array([0, 1, 0]), 'z': np.array([0, 0, 1])}
        local_dir = axis_map.get(default_dir, axis_map['x'])
        R = rpy_to_matrix(rpy)
        dir_in_ref = R @ local_dir
        pos_final = pos + (hand_bias * scale) * dir_in_ref

        eef_pos.append(pos)
        eef_rpy.append(rpy)
        eef_pos_final.append(pos_final)

    return np.asarray(eef_pos), np.asarray(eef_rpy), np.asarray(eef_pos_final)

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n

def compute_cam_pose_in_robot_coordinate_system(camera_mount, scale):
    camera_bias_scale = camera_mount['camera_bias_scale']
    front = np.asarray(camera_mount["front_positions"], dtype=float) * scale
    back  = np.asarray(camera_mount["back_positions"],  dtype=float) * scale

    front_center = front.mean(axis=0)
    back_center  = back.mean(axis=0)
    dir_fb = _normalize(front_center - back_center)  # 从 back 指向 front

    # 面内基向量（选 front 面）
    edge1 = front[0] - front[1]
    edge2 = front[3] - front[0]  # 与 edge1 不共线
    n_tmp = np.cross(edge2, edge1)
    front_normal = _normalize(n_tmp)


    # 让法向与 dir_fb 同向
    if np.dot(front_normal, dir_fb) < 0:
        front_normal = -front_normal

    x_dir = _normalize(edge1)
    y_dir = _normalize(np.cross(front_normal, x_dir))
    x_dir = _normalize(np.cross(y_dir, front_normal))  # 再正交一次
    z_dir = front_normal  # 即 dir_fb

    dist = np.linalg.norm(front_center - back_center)
    cam_pos = front_center + dir_fb * (camera_bias_scale * dist)
    R = np.stack([x_dir, y_dir, z_dir], axis=1)  # 3x3

    return cam_pos, R

def compute_cam_pose_in_reconstruction_coordinate_system(c2w_matrix):
    """
    输入: c2w_matrix (4x4)，相机到重建世界坐标系的齐次矩阵
    输出:
        cam_pos_recon: (3,) 相机在重建坐标系下的位置
        cam_rot_recon: (3,3) 相机朝向旋转矩阵（列向量为相机轴在重建世界系的方向）
    """
    T = np.asarray(c2w_matrix, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    # 可选: 正交化，避免数值误差或含尺度
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:  # 确保右手系
        U[:, -1] *= -1
        R_ortho = U @ Vt

    cam_pos_recon = t
    cam_rot_recon = R_ortho
    return cam_pos_recon, cam_rot_recon

def transform_pose(pos_robot, T_robot2recon):
    """
    pos_robot: (n,3) 机器人坐标系下的点
    T_robot2recon: 4x4 齐次变换矩阵（列向量版）
    返回: (n,3) 重建坐标系下的点
    """
    pts = np.asarray(pos_robot, dtype=float)
    ones = np.ones((pts.shape[0], 1), dtype=float)
    homo = np.concatenate([pts, ones], axis=1)            # (n,4)
    homo_recon = (T_robot2recon @ homo.T).T               # (n,4)
    return homo_recon[:, :3]

def compute_transfer_matrix(cam_pos_robot, cam_rot_robot, cam_pos_recon, cam_rot_recon):
    # R: robot -> recon
    R_wr = cam_rot_recon @ cam_rot_robot.T
    t_wr = cam_pos_recon - R_wr @ cam_pos_robot
    T = np.eye(4)
    T[:3, :3] = R_wr
    T[:3, 3] = t_wr
    return T

from scipy.spatial.transform import Rotation as R

def decompose_rotmat(rot_mat, order='xyz'):
    """
    rot_mat: (3,3) 旋转矩阵（列向量为局部 x/y/z 轴在世界系的方向）
    order  : 欧拉角顺序，默认 'xyz'
    返回:
        axes: [x_axis, y_axis, z_axis]，每个为 shape (3,)
        euler_deg: shape (3,) 的欧拉角，单位度
    """
    rot_mat = np.asarray(rot_mat, dtype=float).reshape(3, 3)
    x_axis = rot_mat[:, 0]
    y_axis = rot_mat[:, 1]
    z_axis = rot_mat[:, 2]
    axes = [x_axis, y_axis, z_axis]

    euler_rad = R.from_matrix(rot_mat).as_euler(order, degrees=False)
    euler_deg = np.degrees(euler_rad)
    return axes, euler_deg

def track_robot(seq_dir, particle_num=30):
    robot_data_path = os.path.join(seq_dir, 'robots', 'robot_data.npz')
    robot_data = np.load(robot_data_path)
    pos_left = robot_data['pos_left']
    pos_right = robot_data['pos_right']
    contact_left = robot_data['contact_left']
    contact_right = robot_data['contact_right']

    pos_left_recon = None
    pos_right_recon = None
    if contact_left.sum() > 0:
        pos_left_recon = pos_left.reshape(-1,1, 3)
        pos_left_recon = pos_left_recon.repeat(particle_num, axis=1)
        pos_bias = np.random.randn(particle_num, 3) * 0.01
        pos_bias = pos_bias.reshape(1, particle_num, 3)
        pos_left_recon += pos_bias

    if contact_right.sum() > 0:
        pos_right_recon = pos_right.reshape(-1,1, 3)
        pos_right_recon = pos_right_recon.repeat(particle_num, axis=1)
        pos_bias = np.random.randn(particle_num, 3) * 0.01
        pos_bias = pos_bias.reshape(1, particle_num, 3)
        pos_right_recon += pos_bias

    if pos_left_recon is not None and pos_right_recon is not None:
        pos = np.concatenate([pos_left_recon, pos_right_recon], axis=1)
    elif pos_left_recon is not None:
        pos = pos_left_recon
    elif pos_right_recon is not None:
        pos = pos_right_recon
    else:
        assert False, 'No contact found'
    
    data = {
        'controller_points': pos,  # (T, N, 3)
    }
    with open(os.path.join(seq_dir, 'track_process_data.pkl'), 'wb') as f:
        pickle.dump(data, f)

def align_robot(robot_dir, seq_dir, cam_ids, dataset_name):
    # data paths
    urdf_path = os.path.join(robot_dir, 'lift2_collision.urdf')
    camera_mount_path = os.path.join(robot_dir, 'camera_mount.json')
    robot_data_path = os.path.join(seq_dir, 'robots', 'data.npz')
    camera_info_path = os.path.join(seq_dir, 'pi3', 'camera_meta.pkl')

    # load data
    urdf_model = URDF.load(urdf_path)
    camera_mount = json.load(open(camera_mount_path, 'r'))
    robot_data = np.load(robot_data_path)
    camera_info = pickle.load(open(camera_info_path, 'rb'))
    scale = camera_mount['scale_factor'][dataset_name]

    # 1. compute eef pose in robot coordinate system
    qpos = robot_data['/observations/qpos']
    # eef_world = robot_data['/observations/eef']
    aligned_eef = []

    hand_bias = camera_mount['hand_bias']
    # link_name = 'right_link21'
    # eef_pos, eef_rpy, eef_pos_robot_left = compute_eef_pose_from_qpos(robot_dir, qpos, link_name, scale, use_gripper=False)
    # print(eef_pos[:5])
    link_name = 'left_link17'
    eef_pos, eef_rpy, eef_pos_robot_left = compute_eef_pose_from_qpos(robot_dir, qpos, link_name, scale, use_gripper=False, hand_bias=hand_bias)
    # print('pos_left', eef_pos[:5], eef_pos_robot_left[:5])
    link_name = 'right_link27'
    eef_pos, eef_rpy, eef_pos_robot_right = compute_eef_pose_from_qpos(robot_dir, qpos, link_name, scale, use_gripper=False, hand_bias=hand_bias)
    # print('pos_right', eef_pos[:5], eef_pos_robot_right[:5])
    # np.savez(os.path.join(seq_dir, 'robots', 'eef_pos_robot.npz'), eef_pos_robot_left=eef_pos_robot_left, eef_pos_robot_right=eef_pos_robot_right)

    # 2. compute transform between robot coordinate system and reconstruction coordinate system
    # 2.1 compute cam pos/rot in robot coordinate system
    cam_pos_robot, cam_rot_robot = compute_cam_pose_in_robot_coordinate_system(camera_mount, scale)

    # 2.2 compute cam pos/rot in reconstruction coordinate system
    c2w_matrix = camera_info['c2ws'][cam_ids]
    cam_pos_recon, cam_rot_recon = compute_cam_pose_in_reconstruction_coordinate_system(c2w_matrix)
    # print('cam_pos_recon', cam_pos_recon)

    # 2.3 compute transfer matrix
    T_robot2recon = compute_transfer_matrix(cam_pos_robot, cam_rot_robot, cam_pos_recon, cam_rot_recon)
    np.savez(os.path.join(seq_dir, 'robots', 'T_robot2recon.npz'), T_robot2recon=T_robot2recon)
    # print('T_robot2recon', T_robot2recon.shape, T_robot2recon)
    
    # 3. compute eef pose in reconstruction coordinate system
    pos_left_recon = transform_pose(eef_pos_robot_left, T_robot2recon)
    pos_right_recon = transform_pose(eef_pos_robot_right, T_robot2recon)
    np.savez(os.path.join(seq_dir, 'robots', 'eef_pos_recon.npz'), pos_left_recon=pos_left_recon, pos_right_recon=pos_right_recon)
    # print('pos_left_recon', pos_left_recon.shape, pos_left_recon[20:25])
    # print('pos_right_recon', pos_right_recon.shape, pos_right_recon[20:25])

    # 4. infer the active robot hand from the scene name
    seq_name = seq_dir.split('/')[-1]
    if 'left' in seq_name:
        contact_left = np.ones((pos_left_recon.shape[0], 1))
        contact_right = np.zeros((pos_right_recon.shape[0], 1))
    elif 'right' in seq_name:
        contact_left = np.zeros((pos_left_recon.shape[0], 1))
        contact_right = np.ones((pos_right_recon.shape[0], 1))
    elif 'double' in seq_name:
        contact_left = np.ones((pos_left_recon.shape[0], 1))
        contact_right = np.ones((pos_right_recon.shape[0], 1))
    else:
        assert False, 'Unknown control hand: sequence name: ' + seq_name
    
    # 5. save data
    np.savez(os.path.join(seq_dir, 'robots', 'robot_data.npz'), pos_left=pos_left_recon, pos_right=pos_right_recon, contact_left=contact_left, contact_right=contact_right)
