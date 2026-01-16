import torch
from dinov2.models.vision_transformer import vit_small, vit_base, vit_large


def load_dino_encoder(model_path, arch="vit_small"):
    """
    arch: "vit_small", "vit_base", "vit_large"
    model_path: .pth 权重路径
    """
    if arch == "vit_small":
        model = vit_small(patch_size=14)
    elif arch == "vit_base":
        model = vit_base(patch_size=14)
    elif arch == "vit_large":
        model = vit_large(patch_size=14)
    else:
        raise ValueError("Unknown architecture")

    state_dict = torch.load(model_path, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print("Loaded weights:", msg)

    model.eval().cuda()
    return model
