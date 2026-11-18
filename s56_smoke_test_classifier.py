# ============================================================================
# Smoke test: load the classifier, predict one crop.
# Repository name : s56_smoke_test_classifier.py
# Original name   : test_imagenet_model.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
import os

# 使用 Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

import timm
import torch

def main() -> None:
    print("PyTorch版本：", torch.__version__)
    print("timm版本：", timm.__version__)
    print("正在下载或读取官方ImageNet权重，请稍候……")

    model = timm.create_model(
        model_name=(
            "hf_hub:"
            "timm/convnextv2_atto.fcmae_ft_in1k"
        ),
        pretrained=True,
        num_classes=8,
    )

    print("官方ImageNet预训练权重加载成功")
    print("输出类别数：", model.num_classes)


if __name__ == "__main__":
    main()