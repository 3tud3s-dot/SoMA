import os
import json
import pickle
from pathlib import Path
import torch
import numpy as np
import PIL.Image as Image
import pickle
import open3d as o3d
import cv2


def pcd_align(source_pts, target_pts):
    """对齐两个点云（numpy数组），返回变换矩阵"""
    # 创建Open3D点云对象
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)
    
    # 计算法向量（ICP需要）
    source_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # ICP 对齐
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, 
        max_correspondence_distance=0.05,  # 根据您的数据尺度调整
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result.transformation

def downsample_pcd(source_path, target_path, downsample_type='random', downsample_rate=0.1, downsample_voxel_size=0.006):
    source_pcd = o3d.io.read_point_cloud(source_path)

    if downsample_type == 'random':
        total_points = len(source_pcd.points)
        keep_points = int(total_points * downsample_rate)
        indices = np.random.choice(total_points, keep_points, replace=False)
        source_pcd.points = o3d.utility.Vector3dVector(source_pcd.points[indices])
        source_pcd.colors = o3d.utility.Vector3dVector(source_pcd.colors[indices])
    elif downsample_type == 'voxel':
        source_pcd = source_pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    o3d.io.write_point_cloud(target_path, source_pcd)

# def downsample_points(pts, cols, voxel_size=0.01):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts)
#     pcd.colors = o3d.utility.Vector3dVector(cols)

#     pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
#     pts_down = np.asarray(pcd_down.points)
#     cols_down = np.asarray(pcd_down.colors)
#     return pts_down, cols_down

def run_pi3(base_dir, mounted_cam=0, valid_cam_ids=None, voxel_size=0.01, fixe_cam_pos=True, pi3_repo=None, pi3_model='yyfz233/Pi3'):
    if pi3_repo is not None:
        from .utils import add_python_path
        add_python_path(pi3_repo)
    from pi3.utils.basic import load_images_as_tensor, write_ply
    from pi3.models.pi3 import Pi3

    img_dir = os.path.join(base_dir, 'pi3', 'images')
    masked_img_dir = os.path.join(base_dir, 'pi3', 'masks')
    desk_masked_img_dir = os.path.join(base_dir, 'pi3', 'desk_masks')
    metadata_path = os.path.join(base_dir, 'metadata.json')
    out_cam_meta = os.path.join(base_dir, 'pi3', 'camera_meta.pkl')
    out_cam_pos = os.path.join(base_dir, 'calibrate.pkl')
    interp_path = os.path.join(base_dir, 'pi3', 'interp_poses.pkl')
    out_ply_masked = os.path.join(base_dir, 'pi3', 'observation_original.ply')
    out_ply_desk = os.path.join(base_dir, 'pi3', 'observation_desk.ply')
    out_ply_full = os.path.join(base_dir, 'pi3', 'observation_full.ply')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 1. 读图片（Pi3 会内部 resize 到统一 H,W）=====
    imgs_full   = load_images_as_tensor(str(img_dir), interval=1).to(device)  # (N,3,H_pi3,W_pi3)
    imgs_masked = load_images_as_tensor(str(masked_img_dir), interval=1).to(device)  # (N,3,H_pi3,W_pi3)
    flag_no_desk = False
    if os.path.exists(desk_masked_img_dir) and len(os.listdir(desk_masked_img_dir)) == 3:
        imgs_desk_masked = load_images_as_tensor(str(desk_masked_img_dir), interval=1).to(device)  # (N,3,H_pi3,W_pi3)
    else:
        imgs_desk_masked = imgs_masked
        flag_no_desk = True
        
    N, _, H_pi3, W_pi3 = imgs_full.shape
    print(f"Loaded {N} images from {img_dir}, resized to ({H_pi3}, {W_pi3}) for Pi3")
    # exit()
    # ===== 2. 跑 Pi3 一次，得到点云 + 相机位姿 =====
    print("Loading Pi3 model...")
    model = Pi3.from_pretrained(pi3_model).to(device).eval()

    print("Running Pi3 inference...")
    with torch.no_grad():
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                res = model(imgs_full[None])   # (1,N,3,H_pi3,W_pi3)
        else:
            res = model(imgs_full[None])
    
    points = res["points"][0]                                # (N,H_pi3,W_pi3,3)
    conf = res["conf"][0, ..., 0]                            # (N,H_pi3,W_pi3)
    local_points = res["local_points"][0]                    # (N,H_pi3,W_pi3,3)
    camera_poses = res["camera_poses"][0].cpu().numpy()      # (N,4,4)

    pts_world_list = []
    cols_list = []
    pts_world_desk_list = []
    cols_desk_list = []
    pts_world_full_list = []
    cols_full_list = []
    for i in range(camera_poses.shape[0]):
        # if valid_cam_ids is not None and i not in valid_cam_ids:
        #     continue

        T = camera_poses[i]
        masked_i = imgs_masked[i].cpu().numpy()
        masked_i = masked_i.transpose(1, 2, 0)
        pixel_masked = np.mean(masked_i, axis=2)
        mask = pixel_masked > 0
        mask = mask.reshape(-1)
        

        pts_local = local_points[i].reshape(-1, 3).cpu().numpy()
        pts_local_h = np.concatenate([pts_local, np.ones((pts_local.shape[0], 1))], axis=1)
        pts_world = (T @ pts_local_h.T).T[:, :3]
        cols = imgs_full[i].reshape(-1, 3).cpu().numpy()

        pts_world_list.append(pts_world[mask])
        cols_list.append(cols[mask])
        pts_world_full_list.append(pts_world)
        cols_full_list.append(cols)

    # 对齐相机位置和旋转
    for i in range(camera_poses.shape[0]):
        if i == mounted_cam:
            continue
        source_pcd = pts_world_list[i]
        target_pcd = pts_world_list[mounted_cam]
        transformation = pcd_align(source_pcd, target_pcd)
        if fixe_cam_pos:
            camera_poses[i] = transformation @ camera_poses[i]

    pts_world_list = []
    cols_list = []
    pts_world_full_list = []
    cols_full_list = []
    for i in range(camera_poses.shape[0]):
        if valid_cam_ids is not None and i not in valid_cam_ids:
            continue

        T = camera_poses[i]
        masked_i = imgs_masked[i].cpu().numpy()
        masked_i = masked_i.transpose(1, 2, 0)
        pixel_masked = np.mean(masked_i, axis=2)
        mask = pixel_masked > 0
        mask = mask.reshape(-1)

        desk_masked_i = imgs_desk_masked[i].cpu().numpy()
        desk_masked_i = desk_masked_i.transpose(1, 2, 0)
        pixel_desk_masked = np.mean(desk_masked_i, axis=2)
        desk_mask = pixel_desk_masked > 0
        desk_mask = desk_mask.reshape(-1)

        pts_local = local_points[i].reshape(-1, 3).cpu().numpy()
        pts_local_h = np.concatenate([pts_local, np.ones((pts_local.shape[0], 1))], axis=1)
        pts_world = (T @ pts_local_h.T).T[:, :3]
        cols = imgs_full[i].reshape(-1, 3).cpu().numpy()

        pts_world_list.append(pts_world[mask])
        cols_list.append(cols[mask])
        pts_world_desk_list.append(pts_world[desk_mask])
        cols_desk_list.append(cols[desk_mask])
        pts_world_full_list.append(pts_world)
        cols_full_list.append(cols)

    # 合并所有视角的点
    pts_masked = np.vstack(pts_world_list)
    cols_masked = np.vstack(cols_list)
    pts_desk = np.vstack(pts_world_desk_list)
    cols_desk = np.vstack(cols_desk_list)
    pts_full = np.vstack(pts_world_full_list)
    cols_full = np.vstack(cols_full_list)

    # pts_masked, cols_masked = downsample_points(pts_masked, cols_masked, voxel_size)

    write_ply(pts_masked, cols_masked, str(out_ply_masked))
    write_ply(pts_desk, cols_desk, str(out_ply_desk))
    write_ply(pts_full, cols_full, str(out_ply_full))

    # ===== 3. 内参 K_eff，写入 camera_meta.pkl =====
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    intrinsics_raw = np.array(meta["intrinsics"])   # (N_cam,3,3)，原始 K_raw
    K_eff_array = intrinsics_raw.astype(np.float32)
    W_raw, H_raw = meta["WH"]
    camera_meta = {
        "intrinsics": K_eff_array.tolist(),   # Pi3 分辨率下的等效 K
        "c2ws": camera_poses.tolist(),        # Pi3 估计的 cam-to-world
    }
    with open(out_cam_meta, "wb") as f:
        pickle.dump(camera_meta, f)
    print(f"Saved Pi3 camera meta (with K_eff) to: {out_cam_meta}")

    with open(out_cam_pos, "wb") as f:
        pickle.dump(camera_poses, f)
    print(f"Saved Pi3 camera poses to: {out_cam_pos}")

    # ===== 4 额外导出 interp_poses.pkl（对前三个视角做插值） =====
    num_key = min(3, camera_poses.shape[0])
    key_poses = camera_poses[:num_key]

    n_per_segment = 50
    interp_poses = []
    for seg_idx in range(num_key - 1):
        c2w0 = key_poses[seg_idx]
        c2w1 = key_poses[seg_idx + 1]
        R0, t0 = c2w0[:3, :3], c2w0[:3, 3]
        R1, t1 = c2w1[:3, :3], c2w1[:3, 3]

        # 线性插值平移，旋转做线性插值后用 SVD 正交化成合法旋转矩阵
        for alpha in np.linspace(0.0, 1.0, n_per_segment, endpoint=False):
            t = (1.0 - alpha) * t0 + alpha * t1

            M = (1.0 - alpha) * R0 + alpha * R1
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

            c2w_interp = np.eye(4, dtype=np.float32)
            c2w_interp[:3, :3] = R
            c2w_interp[:3, 3] = t
            interp_poses.append(c2w_interp)
    
    with open(interp_path, "wb") as f:
        pickle.dump(interp_poses, f)
    print(f"Saved interp poses (num={len(interp_poses)}) to: {interp_path}")
    return flag_no_desk


def gen_depth_map(source_dir, target_dir, pi3_repo=None, pi3_model='yyfz233/Pi3'):
    if pi3_repo is not None:
        from .utils import add_python_path
        add_python_path(pi3_repo)
    from pi3.utils.basic import load_images_as_tensor
    from pi3.models.pi3 import Pi3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgs_full = load_images_as_tensor(str(source_dir), interval=1).to(device)  # (N,3,H,W)
    N, _, H, W = imgs_full.shape
    im0 = Image.open(os.path.join(source_dir,  '0.jpg'))
    H0, W0 = im0.size

    model = Pi3.from_pretrained(pi3_model).to(device).eval()
    with torch.no_grad():
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                res = model(imgs_full[None])   # (1,N,3,H,W)
        else:
            res = model(imgs_full[None])
    
    local_points = res["local_points"][0]  # (N,H,W,3)
    depth_maps = local_points[..., 2].cpu().numpy()

    for i in range(N):
        depth_map = depth_maps[i]  # (H,W)
        depth_map = cv2.resize(depth_map, (W0, H0)) #可以检查一下长宽要不要互换
        output_path = os.path.join(target_dir, f"{i}.npy")
        np.save(output_path, depth_map)
