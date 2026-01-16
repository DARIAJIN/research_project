import numpy as np
import nibabel as nib
import SimpleITK as sitk
from sklearn.decomposition import PCA
from pathlib import Path


def mask_to_world_points_1(mask, affine):
    ijk = np.array(np.where(mask > 0)).T
    ijk_h = np.c_[ijk, np.ones(len(ijk))]
    return (ijk_h @ affine.T)[:, :3]

def mask_to_world_points(mask, affine):
    ijk = np.array(np.where(mask > 0)).T
    if ijk.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    ijk_h = np.c_[ijk, np.ones(len(ijk))]
    xyz = (ijk_h @ affine.T)[:, :3]
    return xyz

def align_pca_strict_1(points):
    mean = points.mean(axis=0)
    pts = points - mean

    pca = PCA(3).fit(pts)
    R = pca.components_

    if R[0, 0] < 0:
        R[0] *= -1
    if R[1, 1] < 0:
        R[1] *= -1

    R[2] = np.cross(R[0], R[1])
    R[2] /= np.linalg.norm(R[2])

    return R, mean

def align_pca_strict(points):
    mean = points.mean(axis=0)
    pts_centered = points - mean

    pca = PCA(3).fit(pts_centered)
    basis = pca.components_  # PC1, PC2, PC3

    # PC1 朝 +X
    if basis[0, 0] < 0:
        basis[0] = -basis[0]
    # PC2 朝 +Y
    if basis[1, 1] < 0:
        basis[1] = -basis[1]

    # PC3 = PC1 × PC2（正交）
    pc1 = basis[0]
    pc2 = basis[1]
    pc3 = np.cross(pc1, pc2)
    pc3 = pc3 / np.linalg.norm(pc3)

    R = np.vstack([pc1, pc2, pc3])
    aligned = pts_centered @ R.T
    return aligned, R, mean


def build_RAS_transform_1(R, origin):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ origin
    return T

def build_RAS_transform(R_scap, origin, is_left):
    """
    输入：
      R_scap: 来自 PCA 的 scapula frame (3x3)
      origin: PCA 中心
    输出：
      T (4x4): world → RAS 的仿射
    """
    Xs = R_scap[0]
    Ys = R_scap[1]
    Zs = R_scap[2]

    # Z 指向 +Z
    if Zs[2] < 0:
        Zs = -Zs
        Ys = -Ys

    # X 指向 +X
    if Xs[0] < 0:
        Xs = -Xs
        Ys = -Ys

    Ys = np.cross(Zs, Xs)
    Ys = Ys / np.linalg.norm(Ys)

    R_ras = np.vstack([Xs, Ys, Zs])

    T = np.eye(4)
    T[:3, :3] = R_ras
    T[:3, 3]  = -R_ras @ origin
    return T

def resample_to_ras_1mm_1(src_path, T, out_path):
    img = sitk.ReadImage(str(src_path))
    spacing = (1.0, 1.0, 1.0)

    img.SetDirection(T[:3, :3].flatten())
    img.SetOrigin(T[:3, 3])

    size = [int(s * sp / 1.0) for s, sp in zip(img.GetSize(), img.GetSpacing())]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())

    out = resampler.Execute(img)
    sitk.WriteImage(out, str(out_path))

def resample_to_ras_1mm(nifti_path, T, out_path):
    img_nib = nib.load(str(nifti_path))
    affine  = img_nib.affine

    # 新 affine = T @ 原 affine
    new_affine = T @ affine

    sitk_img = sitk.ReadImage(str(nifti_path))
    sitk_img.SetDirection(new_affine[:3, :3].flatten())
    sitk_img.SetOrigin(new_affine[:3, 3])

    spacing = [1.0, 1.0, 1.0]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())

    old_size    = np.array(sitk_img.GetSize())
    old_spacing = np.array(sitk_img.GetSpacing())
    new_size    = np.ceil(old_size * (old_spacing / spacing)).astype(int)
    resampler.SetSize(new_size.tolist())

    out_img = resampler.Execute(sitk_img)
    sitk.WriteImage(out_img, str(out_path))

def preprocess_complete_patient(complete_dir: Path,
                                out_dir: Path,
                                struct: str = "scapula"):
    """
    完全等价于实验代码中的：
      - compute_scapula_frames (单病人版)
      - build_RAS_transform
      - resample_to_RAS_1mm

    输入：
      complete_dir/
        ├─ scapula_left.nii.gz
        └─ scapula_right.nii.gz

    输出：
      out_dir/
        ├─ scapula_left.nii.gz   (RAS + 1mm)
        └─ scapula_right.nii.gz  (RAS + 1mm)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Step 1: 对左右 scapula 分别计算 PCA frame
    # --------------------------------------------------
    scapula_frames = {}

    for side in ["left", "right"]:
        path = complete_dir / f"{struct}_{side}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(path)

        img = nib.load(str(path))
        mask = img.get_fdata() > 0

        pts = mask_to_world_points(mask, img.affine)
        if pts.shape[0] < 50:
            raise ValueError(f"Too few points in {path.name}")

        _, R_scap, origin = align_pca_strict(pts)
        scapula_frames[side] = (R_scap, origin)

    # --------------------------------------------------
    # Step 2: 用 PCA frame 构造 RAS transform + 重采样
    # --------------------------------------------------
    for side in ["left", "right"]:
        R_scap, origin = scapula_frames[side]

        T_ras = build_RAS_transform(
            R_scap=R_scap,
            origin=origin,
            is_left=(side == "left")
        )

        in_path  = complete_dir / f"{struct}_{side}.nii.gz"
        out_path = out_dir / f"{struct}_{side}.nii.gz"

        resample_to_ras_1mm(in_path, T_ras, out_path)


