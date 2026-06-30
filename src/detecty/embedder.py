"""DINOv3 embedder (timm). L2-normalized pooled features for nearest-prototype.

Default backbone: vit_large_patch16_dinov3.lvd1689m. On a 1 GB GPU this will NOT
fit — run on CPU (the default). Use a smaller backbone (vit_small/base_patch16_
dinov3.lvd1689m) for speed at a small accuracy cost.
"""
import numpy as np
import torch

DEFAULT_MODEL = "vit_large_patch16_dinov3.lvd1689m"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        import timm
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)
        self.input_size = cfg.get("input_size")
        self.name = model_name

    @torch.no_grad()
    def embed(self, pil) -> np.ndarray:
        x = self.transform(pil.convert("RGB")).unsqueeze(0).to(self.device)
        f = torch.nn.functional.normalize(self.model(x), dim=-1)
        return f[0].cpu().numpy()
