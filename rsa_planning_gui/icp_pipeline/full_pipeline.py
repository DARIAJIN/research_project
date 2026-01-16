from pathlib import Path
import numpy as np
import nibabel as nib

from .preprocessing import preprocess_complete_patient,mask_to_world_points
from .icp import icp_rigid, warp_mask_with_world_transform



def run_full_pipeline(complete_dir: Path, work_dir: Path, struct="scapula"):
    print("[PIPELINE] Start full pipeline")

    # -------------------------------------------------
    # 1. Preprocessing: PCA + RAS + 1mm resample
    # -------------------------------------------------
    ras_dir = work_dir / "ras"
    ras_dir.mkdir(parents=True, exist_ok=True)

    print("[PIPELINE] Preprocessing (PCA + RAS + resample)")
    preprocess_complete_patient(complete_dir, ras_dir, struct)

    fixed_path  = ras_dir / f"{struct}_right.nii.gz"
    moving_path = ras_dir / f"{struct}_left.nii.gz"

    # -------------------------------------------------
    # 2. Load RAS images and extract WORLD point clouds
    # -------------------------------------------------
    print("[PIPELINE] Loading RAS data and building world point clouds")

    img_f = nib.load(str(fixed_path))
    img_m = nib.load(str(moving_path))

    mask_f = img_f.get_fdata() > 0
    mask_m = img_m.get_fdata() > 0

    pts_f = mask_to_world_points(mask_f, img_f.affine)
    pts_m = mask_to_world_points(mask_m, img_m.affine)

    # -------------------------------------------------
    # 3. Mirror LEFT → RIGHT (world space)
    # -------------------------------------------------
    print("[PIPELINE] Mirroring left → right (world space)")

    mirror = np.eye(4)
    mirror[0, 0] = -1.0

    pts_m_h = np.c_[pts_m, np.ones(len(pts_m))]
    pts_m_mir = (mirror @ pts_m_h.T).T[:, :3]

    # -------------------------------------------------
    # 4. ICP in WORLD space (RAS canonical frame)
    # -------------------------------------------------
    print("[PIPELINE] Running ICP (world space, micro-alignment)")

    T_icp = icp_rigid(pts_m_mir, pts_f)

    # Full transform: left → mirrored → ICP → right
    T_full = T_icp @ mirror

    # -------------------------------------------------
    # 5. Warp mask using WORLD transform (EXPERIMENT-CONSISTENT)
    # -------------------------------------------------
    print("[PIPELINE] Warping mask with world transform")

    out_dir = work_dir / "icp"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{struct}_after_icp.nii.gz"



    warp_mask_with_world_transform(
        mov_path=moving_path,
        tgt_path=fixed_path,
        T_full=T_full,
        out_path=out_path
    )

    print("[PIPELINE] Finished successfully")

    return fixed_path, moving_path, out_path
