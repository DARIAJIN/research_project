import torch
import torch.nn as nn

class DINORegBackbone(nn.Module):
    """
    Backbone compatible with dinov2_vits14_reg4_pretrain.pth
    """
    def __init__(self, embed_dim=384):
        super().__init__()

        # 注册一个假的 backbone，用于接受 checkpoint 结构
        self.embed_dim = embed_dim

        # 必须包含 pos_embed、patch_embed 等字段
        # 用 ModuleDict/ParameterDict 接受任意 keys
        self.state_holder = nn.ParameterDict()

    def forward(self, x):
        raise NotImplementedError("Backbone only used for feature extraction.")


def load_dino_reg_encoder(model_path):
    print(f"[DINO-Reg] Loading reg4 encoder: {model_path}")

    model = DINORegBackbone()

    state_dict = torch.load(model_path, map_location="cpu")

    # 去掉 teacher/student 前缀
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        new_state[k] = v

    msg = model.load_state_dict(new_state, strict=False)
    print("Loaded encoder with msg:", msg)

    model.eval().cuda()
    return model
