import numpy as np
import matplotlib.pyplot as plt

def overlay_3view(fixed, warped, out_png, alpha=0.5):
    """
    fixed, warped: 3D numpy arrays (same shape)
    """
    assert fixed.shape == warped.shape

    z, y, x = fixed.shape
    slices = {
        "axial":    fixed[:, :, x // 2],
        "coronal":  fixed[:, y // 2, :],
        "sagittal": fixed[z // 2, :, :]
    }

    slices_w = {
        "axial":    warped[:, :, x // 2],
        "coronal":  warped[:, y // 2, :],
        "sagittal": warped[z // 2, :, :]
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, key in zip(axes, ["axial", "coronal", "sagittal"]):
        ax.imshow(slices[key], cmap="gray")
        ax.imshow(slices_w[key], cmap="jet", alpha=alpha)
        ax.set_title(key)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
