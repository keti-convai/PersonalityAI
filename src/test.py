import os
import yaml
import argparse

import numpy as np
import random
import pandas as pd
import torch
import torch.optim as optim
import pytorch_lightning as pl
import pytorchvideo.transforms
from torchvision import transforms
from pytorch_lightning import loggers as pl_loggers

from model import PersonalityClassifier, PersonalityRegressor
from datamodule import PersonalityRecognizer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(
        seed
    ) 
    np.random.seed(seed) 
    random.seed(seed) 

   
    torch.backends.cudnn.deterministic = (
        True 
    )
    torch.backends.cudnn.benchmark = (
        False
    )


def setup_datamodule(config, stage='fit'):
    height, width = (112, 112)
    video_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224), 
        transforms.Resize((height, width)),
        pytorchvideo.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    audio_transform = None

    datamodule = ModalDataModule(config=config, video_transform=video_transform, audio_transform=audio_transform)
    datamodule.setup(stage=stage)

    if stage == "fit":
        print(f"Train Dataset Length: {len(datamodule.train_dataset)}")
        print(f"Validation Dataset Length: {len(datamodule.val_dataset)}")
    elif stage == "test":
        print(f"Test Dataset Length: {len(datamodule.test_dataset)}")
    elif stage == "predict":
        print(f"Predict Dataset Length: {len(datamodule.predict_dataset)}")
    elif stage is None:
        print(f"Train Dataset Length: {len(datamodule.train_dataset)}")
        print(f"Validation Dataset Length: {len(datamodule.val_dataset)}")
        print(f"Test Dataset Length: {len(datamodule.test_dataset)}")
    else:
        raise ValueError("Unsupported stage. Supported stages are 'fit', 'test', and 'predict'.")
    return datamodule

def load_model(checkpoint_path, task="regression", **kwargs):
    vision_config = kwargs.get('vision_config')
    audio_config = kwargs.get('audio_config')
    text_config = kwargs.get('text_config')
    optimizer_class = kwargs.get('optimizer_class')
    optimizer_params = kwargs.get('optimizer_params')

    if task in ['reg', 'regression']:
        model = PersonalityRegressor.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            vision_config=vision_config,
            audio_config=audio_config,
            text_config=text_config,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params
        )

    elif task in ['clf', 'cls', 'classification']:
        model = PersonalityClassifier(
                        vision_config={"model_size": "base"},
                        text_config={"pretrained_model": "klue/roberta-base"},
                        audio_config={"pretrained_model": "MIT/ast-finetuned-audioset-10-10-0.4593"},
                        optimizer_class=optimizer_class, 
                        optimizer_params=optimizer_params
        )
        model.load_state_dict(
                torch.load(checkpoint_path)['state_dict'], strict=False
        )
    else:
        raise ValueError(f"Invalid variable 'task': Please choose one of the valid names: 'classification' or 'regression'")
    
    return model

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    set_seed(42)
    cfg_path = args.config
    task = args.task

    config = load_config(cfg_path)
    
    vision_config={"model_size": "base"}
    text_config={"pretrained_model": "klue/roberta-base"}
    audio_config={"pretrained_model": "MIT/ast-finetuned-audioset-10-10-0.4593"}
    optimizer_class = optim.AdamW
    optimizer_params = {
        'lr': config.get('learning_rate'),
        'weight_decay': config.get('weight_decay')
    }
    

    print("\n1. Setup Datasets.")
    datamodule = setup_datamodule(config, stage='test')

    print("\n2. Load model.")
    if task in ['reg', 'regression']:
        checkpoint_path = config['checkpoint_path']['classification']
    elif task in ['clf', 'cls', 'classification']:
        checkpoint_path = config['checkpoint_path']['regression']
    
    model = load_model(
        checkpoint_path=checkpoint_path, 
        task=task,
        vision_config=vision_config,
        audio_config=audio_config,
        text_config=text_config,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params
    )


    print("\n3. Test model.")
    trainer = pl.Trainer(
        devices=1,
        max_epochs=1,
        logger= pl_loggers.CSVLogger("logs", name="test", version=task),
        precision=16,
        enable_progress_bar=True,
    )
    trainer.test(model=model, datamodule=datamodule, verbose=False)

    df = pd.read_csv(f'./logs/test/{task}/metrics.csv')
    result_dict = df.iloc[-1].to_dict()
    
    print("\nFinal Result")
    print("------------------------------------------------------------------------")
    
    if task in ["reg", "regression"]:
        report_mean_absolute_error(result_dict)
    elif task in ["clf", "cls", "classification"]:
        report_accuracy(result_dict)
        
    return print()

def report_accuracy(result_dict):
    def print_round_acc(value):
        return f"{round(value * 100, 4)} %"
    
    print("openness:", print_round_acc(result_dict['opn_acc_epoch']))
    print("conscientiousness:", print_round_acc(result_dict['con_acc_epoch']))
    print("extraversion:", print_round_acc(result_dict['ext_acc_epoch']))
    print("agreeableness:", print_round_acc(result_dict['agr_acc_epoch']))
    print("neuroticism:", print_round_acc(result_dict['neu_acc_epoch']))
    print("------------------------------------------------------------------------")
    print(f"Accuracy:", print_round_acc(result_dict['avg_acc_epoch']))
    
def report_mean_absolute_error(result_dict):
    def print_round_mae(value):
        return f"{value:.4f}"
    
    print("openness:", print_round_mae(result_dict['opn_mae_epoch']))
    print("conscientiousness:", print_round_mae(result_dict['con_mae_epoch']))
    print("extraversion:", print_round_mae(result_dict['ext_mae_epoch']))
    print("agreeableness:", print_round_mae(result_dict['agr_mae_epoch']))
    print("neuroticism:", print_round_mae(result_dict['neu_mae_epoch']))
    print("------------------------------------------------------------------------")
    print(f"One Minus MAE:", print_round_mae(result_dict['1-mae_epoch']))


def set_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(
        seed
    ) 
    np.random.seed(seed) 
    random.seed(seed) 

  
    torch.backends.cudnn.deterministic = (
        True 
    )
    torch.backends.cudnn.benchmark = (
        False 
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', default='regression', type=str, required=False, help="Select model task. 'classification' or 'regression'")
    parser.add_argument('--config', '-cfg', default='./configs/test_config.yaml', type=str, required=False, help="Config file path.")

    args = parser.parse_args()
    set_seed(42)
    main(args)
    
