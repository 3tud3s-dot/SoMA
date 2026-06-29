import os
import numpy as np
from plyfile import PlyData

COLOR_DICT = {
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "red":   (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue":  (0.0, 0.0, 1.0),
}

def postprocess_gs(gs_model: str,
                   remain_color_list: list = ("green", "white"),
                   overwrite: bool = False,
                   dist_thresh: float = 0.25) -> str:
    """
    1) 对 remain_color_list 中每个目标颜色，在点云中找到与其最接近的颜色实例；
    2) 对所有点，若与任一“保留色”距离都大于 dist_thresh，则将其颜色改为找到的第一个保留色。
    颜色距离用 RGB L2 范数（0-1 空间）。
    """
    if not os.path.isfile(gs_model):
        raise FileNotFoundError(f"file not found: {gs_model}")

    ply = PlyData.read(gs_model)
    if not ply.elements:
        raise ValueError("PLY has no elements")
    vertex = ply.elements[0]
    required = {"f_dc_0", "f_dc_1", "f_dc_2", "opacity"}
    if not required.issubset(vertex.data.dtype.names):
        raise ValueError(f"Missing fields {required} in PLY")

    colors = np.stack(
        [vertex.data["f_dc_0"], vertex.data["f_dc_1"], vertex.data["f_dc_2"]],
        axis=1,
    )  # shape (N,3), float in 0-1

    # 统计唯一颜色及其频次
    quant = np.clip(np.round(colors * 255), 0, 255).astype(np.int16)
    uniq, counts = np.unique(quant, axis=0, return_counts=True)

    # 为每个目标颜色选择“频次优先、距离次之”的最匹配颜色
    keep_colors = []
    for name in remain_color_list:
        if name not in COLOR_DICT:
            raise ValueError(f"unknown color name: {name}")
        target = np.array(COLOR_DICT[name], dtype=np.float32) * 255.0  # 与 quant 对齐
        dists = np.linalg.norm(uniq - target[None, :], axis=1)
        # 先按频次降序，再按距离升序
        best_idx = np.lexsort((dists, -counts))[0]
        best_color = (uniq[best_idx].astype(np.float32) / 255.0)
        keep_colors.append(best_color)

    if not keep_colors:
        raise ValueError("no keep colors found")
    keep_colors = np.stack(keep_colors, axis=0)  # (K,3)

    # 计算每个点到最近保留色的距离
    dists = np.linalg.norm(colors[:, None, :] - keep_colors[None, :, :], axis=2)  # (N,K)
    min_d = dists.min(axis=1)  # (N,)

    # 远离所有保留色的点改为第一个保留色
    replace_color = keep_colors[1]
    mask = min_d > dist_thresh
    colors[mask] = replace_color

    # 写回
    vertex.data["f_dc_0"][:] = colors[:, 0]
    vertex.data["f_dc_1"][:] = colors[:, 1]
    vertex.data["f_dc_2"][:] = colors[:, 2]

    out_path = gs_model if overwrite else os.path.splitext(gs_model)[0] + "_post.ply"
    ply.write(out_path)
    return out_path
