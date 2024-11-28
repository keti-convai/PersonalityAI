from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl
from transformers import AutoModel, ASTModel
import pandas as pd
from einops import rearrange

from torchmetrics import classification as clf
from torchmetrics import regression as reg


class LitKetiCorpusMultimodalRegressor(pl.LightningModule):

    def __init__(
            self,
            vision_config,
            audio_config,
            text_config,
            optimizer_class,
            optimizer_params
        ):
        super().__init__()
        self.model = KetiCorpusMultimodalRecognizer(
            vision_config=vision_config,
            audio_config=audio_config,
            text_config=text_config
        )
        self.output = nn.Sequential(
            nn.Linear(128, 5),
            # nn.Sigmoid()
            nn.Softmax()
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        self.mean_absolute_error = reg.MeanAbsoluteError()
        self.test_step_outputs = defaultdict(list)
    
    def forward(self, video_input, audio_input, text_input):
        x = self.model(video_input, audio_input, text_input)
        x = self.output(x)
        return x
    
    def training_step(self, batch, batch_idx):
        video_clip_id, X_video, X_audio, X_text, y_true = batch
        y_pred = self(X_video, X_audio, X_text)
        loss = self.loss_fn(y_pred, y_true)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video_clip_id, X_video, X_audio, X_text, y_true = batch
        y_pred = self(X_video, X_audio, X_text)
        loss = self.loss_fn(y_pred, y_true)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        metrics = self.evaluate(y_pred, y_true)
        return loss
    
    def test_step(self, batch, batch_idx):
        video_clip_id, X_video, X_audio, X_text, y_true = batch
        y_pred = self(X_video, X_audio, X_text)
        loss = self.loss_fn(y_pred, y_true)
        
        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        metrics = self.evaluate(y_pred, y_true)
        
        batch_size = y_pred.shape[0]
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        for i in range(batch_size):
            self.test_step_outputs['video_clip_id'].append(video_clip_id[i])
            self.test_step_outputs['predictions'].append(y_pred[i])
            self.test_step_outputs['targets'].append(y_true[i])
        
        ## 결과 출력
        print("\n# ----------------------------------------------------- #")
        for i in range(2):
            print("  Video Clip ID:", video_clip_id[i])
            print("  preds:", y_pred[i])
            print("  targets:", y_true[i])
            print()
        print("  loss:", loss)
        print("# ------------------------------------------------------ #\n")  
        return loss
    
    def on_test_epoch_end(self):
        df = pd.DataFrame(self.test_step_outputs)
        df.to_csv('./logs/test/regression/output.csv', index=False)
        
        print("파일이 저장되었습니다: '/logs/test/regression/output.csv'")
    
    def evaluate(self, y_pred, y_true):
        mae = []
        for i in range(y_true.shape[-1]):
            mae.append(1-self.mean_absolute_error(y_pred[:, i], y_true[:, i]))
        
        mae.append(1-self.mean_absolute_error(y_pred, y_true))

        metrics = {
            'opn_mae': mae[0],
            'con_mae': mae[1],
            'ext_mae': mae[2],
            'agr_mae': mae[3],
            'neu_mae': mae[4],
            '1-mae': mae[5],
        }

        for metric, score in metrics.items():
            self.log(metric, score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return metrics
    
    def configure_optimizers(self):
        if self.optimizer_params:
            optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        optimizer = self.optimizer_class(self.parameters())
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'val_loss',  # 학습률 감소를 위해 모니터링할 메트릭 설정
            'interval': 'epoch',    # 에포크마다 체크
            'frequency': 1          # 매 에포크마다 확인
        }
        return [optimizer], [scheduler]


class LitKetiCorpusMultimodalClassifier(pl.LightningModule):

    def __init__(
            self,
            vision_config,
            audio_config,
            text_config,
            optimizer_class,
            optimizer_params
        ):
        super().__init__()
        self.save_hyperparameters()
        self.model = KetiCorpusMultimodalRecognizer(
            vision_config=vision_config,
            audio_config=audio_config,
            text_config=text_config
        )
        self.output = nn.Sequential(
            nn.Linear(128, 5),
            nn.Sigmoid()
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        
        self.test_step_outputs = defaultdict(list)
        self.multilable_accuracy = clf.MultilabelAccuracy(num_labels=5, threshold=0.5, average=None)
    
    def forward(self, video_input, audio_input, text_input):
        x = self.model(video_input, audio_input, text_input)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        X_video, X_audio, X_text, y_true = batch
        
        y_pred = self(X_video, X_audio, X_text)
        y_true = (y_true >= 0).to(y_pred.device).to(y_pred.dtype)
        # y_true = torch.cat((y_true[:, :2], y_true[:, 3:]), dim=1)

        loss = self.loss_fn(y_pred, y_true)

        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_video, X_audio, X_text, y_true = batch
        
        y_pred = self(X_video, X_audio, X_text)
        y_true = (y_true >= 0).to(y_pred.device).to(y_pred.dtype)
        # y_true = torch.cat((y_true[:, :2], y_true[:, 3:]), dim=1)

        loss = self.loss_fn(y_pred, y_true)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        metrics = self.evaluate(y_pred, y_true)
        return loss
    
    def test_step(self, batch, batch_idx):
        video_clip_id, X_video, X_audio, X_text, y_true = batch

        y_pred = self(X_video, X_audio, X_text)
        # threshold = torch.tensor([0.5686, 0.6016, 0.5421, 0.5623, 0.4315]).to(y_pred.device)
        threshold = torch.tensor([0.62, 0.60, 0.68, 0.72, 0.5]).to(y_pred.device)
        # threshold = torch.tensor([0.6875, 0.7143, 0.6504, 0.6586, 0.5190]).to(y_pred.device)
        
        y_pred = (y_pred >= threshold).to(y_pred.dtype)
        y_true = (y_true >= threshold).to(y_pred.dtype)

        loss = self.loss_fn(y_pred, y_true)
        
        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        metrics = self.evaluate(y_pred, y_true)
        
        batch_size = y_pred.shape[0]
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        for i in range(batch_size):
            self.test_step_outputs['video_clip_id'].append(video_clip_id[i])
            self.test_step_outputs['predictions'].append(y_pred[i])
            self.test_step_outputs['targets'].append(y_true[i])
        
        ## 결과 출력 ##
        print("\n# -------------------------------------------------- #")
        for i in range(2):
            print("  Video Clip ID:", video_clip_id[i])
            print("  preds:", y_pred[i])
            print("  targets:", y_true[i])
            print()
        print("  loss:", loss)
        print("# --------------------------------------------------- #\n")
        return loss
    
    def on_test_epoch_end(self):
        df = pd.DataFrame(self.test_step_outputs)
        df.to_csv('./logs/test/classification/output.csv', index=False)
        
        print("파일이 저장되었습니다: '/logs/test/classification/output.csv'")
    
    def evaluate(self, y_pred, y_true):
        metrics = self.evaluate_multilable(y_pred, y_true)
        return metrics
    
    def evaluate_multilable(self, y_pred, y_true):
        mla = self.multilable_accuracy(y_pred, y_true).cpu().numpy().tolist()
        avg_acc = sum(mla) / len(mla)
        metrics = {
            'opn_acc': mla[0],
            'con_acc': mla[1],
            'ext_acc': mla[2],
            'agr_acc': mla[3],
            'neu_acc': mla[4],
            'avg_acc': avg_acc
        }

        for metric, score in metrics.items():
            self.log(metric, score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return metrics
    
    def configure_optimizers(self):
        if self.optimizer_params:
            optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        optimizer = self.optimizer_class(self.parameters())
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'val_loss',  # 학습률 감소를 위해 모니터링할 메트릭 설정
            'interval': 'epoch',    # 에포크마다 체크
            'frequency': 1          # 매 에포크마다 확인
        }
        return [optimizer], [scheduler]


class KetiCorpusMultimodalRecognizer(nn.Module):

    def __init__(
            self,
            vision_config,
            audio_config,
            text_config,
        ):
        super().__init__()
        self.convnext = ConvNextWithLSTM(**vision_config)
        self.ast = AudioSpectrogramTransformer(**audio_config)
        self.roberta = RoBerta(**text_config)
        self.fc_layers = nn.Sequential(
            nn.Linear(3 * 768, 1024),  
            nn.ReLU(),           
            nn.Dropout(0.3),   

            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, video_input, audio_input, text_input):
        video_out = self.convnext(video_input)
        audio_out = self.ast(audio_input)
        text_out = self.roberta(text_input)

        x = torch.cat([video_out, audio_out, text_out], dim=1)
        x = self.fc_layers(x)
        return x


class TimeDistributed(nn.Module):
    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        batch_size, time_steps = x.size(0), x.size(1)
        x = x.reshape(batch_size * time_steps, *x.size()[2:])
        x = self.layer(x)
        x = x.reshape(batch_size, time_steps, *x.size()[1:])
        return x

class ConvNextWithLSTM(nn.Module):
    def __init__(self, model_size="base"):
        super(ConvNextWithLSTM, self).__init__()
        
        # ConvNext 모델 설정
        match model_size:
            case "tiny":
                self.convnext = models.convnext_tiny(pretrained=True)
            case "small":
                self.convnext = models.convnext_small(pretrained=True)
            case "base":
                self.convnext = models.convnext_base(pretrained=True)
            case "large":
                self.convnext = models.convnext_large(pretrained=True)
            case _:
                self.convnext = models.convnext_base(pretrained=True)
        
        self.convnext.classifier = nn.Identity()
        self.time_distributed_convnext = TimeDistributed(self.convnext)
        self.lstm = nn.LSTM(input_size=1024, hidden_size=768, num_layers=1, batch_first=True)

    def forward(self, x):
        batch_size, _, time_steps, _, _ = x.shape
        x = rearrange(x, "B C T H W -> B T C H W")
        x = self.time_distributed_convnext(x)  # (Batch, Time, Feature_Dim)
        x = x.squeeze(-1, -2)
        
        lstm_out, _ = self.lstm(x)  # (Batch, Time, Hidden_Size)
        output = lstm_out[:, -1, :]  # (Batch, Hidden_Size)
        return output

class RoBerta(nn.Module):
    
    def __init__(
            self, 
            pretrained_model: str = 'xlm-roberta-base',
            dropout: Optional[float] = None
        ) -> None:
        super().__init__()
        self.pretrained = AutoModel.from_pretrained(pretrained_model)     
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Transcription forward function
        Args:
            - x (torch.Tensor): Input tokens tensors tokenized by `pretrained_model`. (B, N)

        Returns:
            - torch.Tensor: `pretrained_model` embedding tensor. (B, N, C)
        
        """
        x = self.pretrained(x).pooler_output
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class AudioSpectrogramTransformer(nn.Module):

    def __init__(
            self,
            pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593", 
            dropout: Optional[float] = None
        ) -> None:
        super().__init__()
        self.pretrained = ASTModel.from_pretrained(pretrained_model)
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`AudioBranch` forward.

        Args:
            x (torch.Tensor): A tensor of input speech tensor.

        Returns:
            out (torch.Tensor): Outputs of `AudioBranch` (B, N, C).

        """
        x = self.pretrained(x).pooler_output
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    

if __name__ == "__main__":
    import yaml
    def load_config(path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    config = load_config('./config.yaml')
    optimizer_class = optim.AdamW
    optimizer_params = {
        'lr': config.get('learning_rate'),
        'weight_decay': config.get('weight_decay')
    }
    big5 = 'opn'
    clf_model = LitKetiCorpusMultimodalClassifier(
                vision_config={"model_size": "base"},
                text_config={"pretrained_model": "klue/roberta-base"},
                audio_config={"pretrained_model": "MIT/ast-finetuned-audioset-10-10-0.4593"},
                optimizer_class=optimizer_class, 
                optimizer_params=optimizer_params,
                big5=big5
            )
    
    video_tensor = torch.rand(2, 3, 2, 112, 112)
    audio_tensor = torch.rand(2, 1024, 128)
    text_tensor = torch.randint(size=(2, 10), low=1, high=300)
    
    out = clf_model(video_tensor, audio_tensor, text_tensor)
    print(out, out.shape)
