import os
import yaml
import warnings
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pytorchvideo.transforms
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from tqdm import tqdm

from model import (
    PersonalityClassifier,
    PersonalityRegressor,
)
from datamodule import MultiModalDataModule

warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    cfg_path = args.config
    task = task = args.task

    config = load_config(cfg_path)

    print("1. Load DataModule.\n")

    height, width = (112, 112)
    video_transform = transforms.Compose(
        [
            transforms.Resize((300, 400)),
            transforms.CenterCrop((height, width)),
            pytorchvideo.transforms.Normalize(
                (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)
            ),
        ]
    )

    audio_transform = None

    datamodule = MultiModalDataModule(
        config=config, video_transform=video_transform, audio_transform=audio_transform
    )
    datamodule.setup(stage="fit")

    print(f"Train Dataset Length: {len(datamodule.train_dataset)}")
    print(f"Validate Dataset Length: {len(datamodule.val_dataset)}\n")

    optimizer_class = optim.AdamW
    optimizer_params = {
        "lr": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
    }
    max_epochs = config.get("max_epochs")
    log_every_n_steps = config.get("log_every_n_steps")

    print("2. Set Parameters.")
    if task in ["reg", "regression"]:
        model = PersonalityRegressor(
            vision_config={"model_size": "base"},
            text_config={"pretrained_model": "klue/roberta-base"},
            audio_config={
                "pretrained_model": "MIT/ast-finetuned-audioset-10-10-0.4593"
            },
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
        )
    elif task in ["clf", "cls", "classification"]:
        model = PersonalityClassifier(
            vision_config={"model_size": "base"},
            text_config={"pretrained_model": "klue/roberta-base"},
            audio_config={
                "pretrained_model": "MIT/ast-finetuned-audioset-10-10-0.4593"
            },
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
        )
        model.load_state_dict(
            torch.load("/root/hskye/pai-rcg-test/models/regression_test_model4.ckpt")["state_dict"],
            strict=False,
        )

    print("3. Init Model.")
    trainer = pl.Trainer(
        logger=pl_loggers.CSVLogger(
            "lightning_logs", name="train", version=f"{task}_v2"
        ),
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        # detect_anomaly=True,
        precision=16,
    )

    print("4. Classification Model.")
    print("4.1 Fit Model.")
    trainer.fit(model, datamodule=datamodule)

    print("4.2 Test Model.")
    datamodule.setup(stage="test")
    print(f"Test Dataset Length: {len(datamodule.test_dataset)}")
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        "-t",
        default="regression",
        type=str,
        required=False,
        help="Select model task. 'classification' or 'regression'",
    )
    parser.add_argument(
        "--config",
        "-cfg",
        default="./configs/config.yaml",
        type=str,
        required=False,
        help="Config file path.",
    )
    args = parser.parse_args()

    main(args)
