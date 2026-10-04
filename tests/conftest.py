import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

try:
    from transformers import DINOv3ViTImageProcessor
except ImportError:
    from transformers import DINOv3ViTImageProcessorFast as DINOv3ViTImageProcessor

from dinov3_embedder import DINOv3Embedder, EmbedderConfig


@pytest.fixture(scope="session")
def local_model(tmp_path_factory):
    """Actual Transformers DINOv3 implementation, random weights, no downloads."""
    torch.set_num_threads(1)
    torch.manual_seed(9)
    path = tmp_path_factory.mktemp("dinov3-model")
    config = DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        num_register_tokens=2,
    )
    DINOv3ViTModel(config).save_pretrained(path)
    DINOv3ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(path)
    return path


@pytest.fixture
def embedder(local_model):
    return DINOv3Embedder(
        EmbedderConfig(
            model_id=str(local_model),
            image_size=32,
            batch_size=2,
            local_files_only=True,
            device="cpu",
        )
    )
