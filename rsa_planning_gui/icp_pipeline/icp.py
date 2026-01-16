import numpy as np
import nibabel as nib
from scipy.spatial import KDTree
from scipy.ndimage import affine_transform


def icp_rigid(A, B, max_iter=300):
    T = np.eye(4)
    A_c = A.copy()

    for _ in range(max_iter):
        tree = KDTree(B)
        _, idx = tree.query(A_c)
        B_m = B[idx]

        mu_A, mu_B = A_c.mean(0), B_m.mean(0)
        W = (A_c - mu_A).T @ (B_m - mu_B)
        U, _, Vt = np.linalg.svd(W)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        t = mu_B - R @ mu_A
        A_c = (R @ A_c.T).T + t

    T[:3, :3] = R
    T[:3, 3] = t
    return T


def warp_mask_with_world_transform(mov_path, tgt_path, T_full, out_path):
    import nibabel as nib
    from scipy.ndimage import affine_transform
    import numpy as np

    nii_mov = nib.load(str(mov_path))
    nii_tgt = nib.load(str(tgt_path))

    vol_mov = nii_mov.get_fdata()
    A_mov = nii_mov.affine
    A_tgt = nii_tgt.affine

    # voxel_tgt -> world_tgt -> world_mov -> voxel_mov
    T_inv = np.linalg.inv(T_full)
    V = np.linalg.inv(A_mov) @ T_inv @ A_tgt

    M = V[:3, :3]
    offs = V[:3, 3]

    warped = affine_transform(
        vol_mov,
        matrix=M,
        offset=offs,
        output_shape=nii_tgt.shape,
        order=0
    )

    nib.save(nib.Nifti1Image(warped, A_tgt), str(out_path))
