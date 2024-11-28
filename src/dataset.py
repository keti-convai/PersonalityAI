import os
import glob
import warnings
import subprocess
from typing import Optional, List, Dict

import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torchaudio.transforms import Resample
from transformers import AutoTokenizer, AutoProcessor

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import noisereduce as nr
from PIL import Image

warnings.filterwarnings('ignore')


PAD, SEP, UNK, BOS, EOS = "<pad>", "</s>", "<unk>", "<s>", "</s>"



class KetiCorpusDataset(Dataset):

    OCEAN_AVG = {
        'openness': 0.668582, 
        'conscientiousness': 0.668582,
        'extraversion': 0.642101,
        'agreeableness': 0.662362,
        'neuroticism': 0.531536
    }
    
    OCEAN_STD = {
        'openness': 0.097586, 
        'conscientiousness': 0.104025, 
        'extraversion': 0.116998, 
        'agreeableness': 0.082447, 
        'neuroticism': 0.119315
    }
    
    FACET_COEF = {
        'openness': [0.25, 0.22, 0.18, 0.21, 0.09, 0.05],
        'conscientiousness': [0.17, 0.17, 0.17, 0.21, 0.09, 0.19],
        'extraversion': [0.16, 0.14, 0.17, 0.18, 0.19, 0.16],
        'agreeableness': [0.16, 0.17, 0.16, 0.16, 0.21, 0.14],
        'neuroticism': [0.2, 0.16, 0.16, 0.12, 0.2, 0.16]
    }

    def _annotate(
            self, 
            video_clip_id: str, 
            annotation: dict, 
            random_facet: bool = False,
            z_score: bool = False,
            **kwargs
        ) -> List[float]:
        if not isinstance(annotation[video_clip_id], dict):
            return [annotation[video_clip_id]]
        
        if random_facet:
            n_facets = kwargs.get('n_facets', 5)
            replace = kwargs.get('replace', False)
            facets = annotation[video_clip_id]['facet']
            labels = self._annotate_random_facets(facets, n_facets, replace)
        else:
            labels = annotation[video_clip_id]['score']
        
        labels = [self._normalize(score, self.OCEAN_AVG[ocean], self.OCEAN_STD[ocean]) if z_score == True else score
                    for ocean, score in labels.items()]
        return labels
    
    def _annotate_random_facets(self, facets, n_facets=5, replace=False) -> Dict[str, float]:
        labels = {}
        for ocean, fcts in facets.items():
            label = np.mean(
                np.random.choice(list(fcts.values()), n_facets, p=self.FACET_COEF[ocean], replace=replace)
            )
            labels[ocean] = label
        return labels
    
    def _normalize(self, x, mu, std) -> float:
        return (x - mu) / std
    
    def get_video_id(self, file_path) -> str:
        file_name = os.path.basename(file_path)
        video_id = os.path.splitext(file_name)[0]
        return video_id
    
    def get_recording_id(self, file_path) -> str:
        file_name = os.path.basename(file_path)
        recording_id = '_'.join(os.path.splitext(file_name)[0].split('_')[:4])
        return recording_id
    

# ------------------------------------------------ MultiModal Dataset ------------------------------------------------ #


class KetiCorpusMultiModalDataset(KetiCorpusDataset):

    def __init__(
            self, 
            data_path: str,
            video_dir: str,
            audio_dir: str, 
            annotation_path: Optional[str] = None,
            min_speech_length: float = 5.9,
            max_speech_length: float = 10.7,
            annotate_random_facet: bool = True,
            n_facets: int = 5,
            z_score: bool = False,
            max_facet_score: int = 20,
            video_transform = None,
            audio_processor: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
            audio_transform = None,
            tokenizer: str = "klue/roberta-base",
            max_len: int = 512,
            n_sample_frame: int = 4,
            start_time: float = 7.5,
            end_time: float = 23.5,
            return_video_clip_id: bool = False,
        ):
        self.data = pd.read_csv(data_path)
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.annotation = pd.read_pickle(annotation_path) if annotation_path else None
        self.max_len = max_len
        self.return_video_clip_id = return_video_clip_id

        self.video_dataset = KetiCorpusVideoDataset(
            data_path=data_path,
            video_dir=video_dir,
            annotation_path=annotation_path,
            n_sample_frame=n_sample_frame,
            start_time=start_time,
            end_time=end_time,
            min_speech_length=min_speech_length,
            max_speech_length=max_speech_length,
            annotate_random_facet=annotate_random_facet,
            n_facets=n_facets,
            z_score=z_score,
            max_facet_score=max_facet_score,
            transform=video_transform,
            return_video_clip_id=True
        )

        self.audio_dataset = KetiCorpusAudioDataset(
            data_path=data_path,
            audio_dir=audio_dir,
            annotation_path=annotation_path,
            audio_processor=audio_processor,
            audio_transform=audio_transform,
            min_speech_length=min_speech_length,
            max_speech_length=max_speech_length,
            annotate_random_facet=False,
            n_facets=n_facets,
            z_score=z_score,
            max_facet_score=max_facet_score,
            return_video_clip_id=True
        )

        video_ids = [data[0] for data in self.video_dataset.dataset]
        audio_ids = [data[0] for data in self.audio_dataset.dataset]
        video_clip_ids = [video_clip_id for video_clip_id in video_ids if video_clip_id in audio_ids]
        
        self.text_dataset = KetiCorpusTranscriptionDataset(
            data_path=data_path,
            annotation_path=None,
            tokenizer=tokenizer,
            max_len=max_len,
            min_speech_length=min_speech_length,
            max_speech_length=max_speech_length,
            annotate_random_facet=False,
            n_facets=n_facets,
            z_score=z_score,
            max_facet_score=max_facet_score,
            return_video_clip_id=True,
            video_clip_ids=video_clip_ids
        )
    
    def __getitem__(self, idx):
        video_id, frames, labels = self.video_dataset[idx]
        audio_id, audio, _ = self.audio_dataset[idx]
        text_id, token_ids, _ = self.text_dataset[idx]
        labels = torch.tensor(labels)

        if video_id == audio_id == text_id:
            if self.return_video_clip_id:
                return  video_id, frames, audio, token_ids, labels
            else:
                return  frames, audio, token_ids, labels
        else: 
            raise ValueError(f"[{idx}]: The data id for each modality does not match. ('video id: {video_id}', audio id: '{audio_id}', text id: '{text_id}')")
    
    def __len__(self):
        if len(self.video_dataset) == len(self.audio_dataset) == len(self.text_dataset):
            return len(self.video_dataset)
        else:
            raise ValueError(f"The length of the dataset for each modality does not match.(video: {len(self.video_dataset)} | audio: {len(self.audio_dataset)} | text: {len(self.text_dataset)}) Please check the dataset.")

    def collate_fn(self, batch):
        if self.return_video_clip_id:
            video_clip_id, frames, audio, token_ids, labels = list(zip(*batch))
            frames = torch.stack(frames)
            audio = torch.stack(audio).squeeze(1)
            text = pad_sequences(token_ids, max_length=self.max_len, batch_first=True)
            labels = torch.stack(labels)
            return video_clip_id, frames, audio, text, labels
        
        frames, audio, token_ids, labels = list(zip(*batch))
        frames = torch.stack(frames)
        audio = torch.stack(audio).squeeze(1)
        text = pad_sequences(token_ids, max_length=self.max_len, batch_first=True)
        labels = torch.stack(labels)
        return frames, audio, text, labels
        
# ------------------------------------------------ Video Dataset ------------------------------------------------ #

class KetiCorpusVideoDataset(KetiCorpusDataset):

    def __init__(
            self, 
            data_path: str,
            video_dir: str, 
            annotation_path: Optional[str] = None,
            n_sample_frame: int = 4,
            start_time: float = 7.5,
            end_time: float = 23.5,
            min_speech_length: float = 5.9,
            max_speech_length: float = 10.7,
            annotate_random_facet: bool = True,
            n_facets: int = 5,
            z_score: bool = False,
            max_facet_score: int = 20,
            transform: transforms = None,
            return_video_clip_id: bool = False,
        ):
        data = pd.read_csv(data_path)
        self.video_dir = video_dir
        self.annotation = pd.read_pickle(annotation_path) if annotation_path else None
        self.n_sample_frame = n_sample_frame
        self.start_time = start_time
        self.end_time = end_time
        self.min_speech_length = min_speech_length
        self.max_speech_length = max_speech_length
        self.annotate_random_facet = annotate_random_facet
        self.n_facets = n_facets
        self.z_score = z_score
        self.max_facet_score = max_facet_score
        self.transform = transform
        self.return_video_clip_id = return_video_clip_id

        self.dataset = []
        for _, (video_clip_id, _, st, ed, opn, con, ext, agr, neu) in data.iterrows():
            if ed - st < min_speech_length or ed - st > max_speech_length:
                continue
            if self.annotation and video_clip_id not in list(self.annotation.keys()):
                continue

            frame_dir = os.path.join(self.video_dir, video_clip_id.split('_', 1)[0], f"{video_clip_id}.MP4") 
            if os.path.exists(frame_dir):
                self.dataset.append([video_clip_id, frame_dir, [opn, con, ext, agr, neu]])
            
        self.dataset.sort(key=lambda x: x[0])
    
    def _load_frames(self, frame_dir):
        frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg') or f.endswith('.png')])
        frame_files = frame_files[int(self.start_time * 29.97) : int(self.end_time * 29.97) + 1]
        if len(frame_files) == 0:
            raise ValueError(f"No frame files found in {frame_dir}")

        total_frames = len(frame_files)
        if total_frames >= self.n_sample_frame:
            indices = np.linspace(0, total_frames - 1, self.n_sample_frame, dtype=int)
        else:
            indices = list(range(total_frames)) + [total_frames - 1] * (self.n_sample_frame - total_frames)
        
        selected_frames = [frame_files[i] for i in indices]
        frames = [F.to_tensor(Image.open(f)) for f in selected_frames]  # (C, H, W) 형태로 변환
        frames = torch.stack(frames, dim=1)  # (C, n_sample_frame, H, W)

        if self.transform:
            frames = self.transform(frames)
        
        return frames

    def __getitem__(self, idx):
        video_clip_id, frame_dir, labels = self.dataset[idx]
        frames = self._load_frames(frame_dir)

        if self.annotation:
            labels = self._annotate(
                video_clip_id=video_clip_id, 
                annotation=self.annotation, 
                random_facet=self.annotate_random_facet,
                z_score=self.z_score
            )
        
        labels = torch.tensor(labels)
        if self.return_video_clip_id:
            return video_clip_id, frames, labels
        return frames, labels

    def __len__(self):
        return len(self.dataset)
    

# ------------------------------------------------ Audio Dataset ------------------------------------------------ #


class KetiCorpusAudioDataset(KetiCorpusDataset):

    def __init__(
            self,
            data_path: str,
            audio_dir: str, 
            annotation_path: Optional[str] = None,
            audio_processor: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
            min_speech_length: float = 5.9,
            max_speech_length: float = 10.7,
            annotate_random_facet: bool = True,
            n_facets: int = 5,
            z_score: bool = False,
            max_facet_score: int = 20,
            audio_transform = None,
            return_video_clip_id: bool = False,
        ):
        data = pd.read_csv(data_path)
        self.audio_dir = audio_dir
        self.annotation = pd.read_pickle(annotation_path) if annotation_path else None
        self.audio_processor = AutoProcessor.from_pretrained(audio_processor)

        self.annotate_random_facet = annotate_random_facet
        self.n_facets = n_facets
        self.z_score = z_score
        self.max_facet_score = max_facet_score
        self.audio_transform = audio_transform
        self.return_video_clip_id = return_video_clip_id

        self.dataset = []
        for _, (video_clip_id, _, st, ed, opn, con, ext, agr, neu) in data.iterrows():
            if ed - st < min_speech_length or ed - st > max_speech_length:
                continue
            if self.annotation and video_clip_id not in list(self.annotation.keys()):
                continue

            audio_file_path = os.path.join(self.audio_dir, video_clip_id.split('_', 1)[0], f"{video_clip_id}.wav") 
            if os.path.exists(audio_file_path):
                self.dataset.append([video_clip_id, audio_file_path, [opn, con, ext, agr, neu]])
            
        self.dataset.sort(key=lambda x: x[0])
    
    def _load_audio(self, file_path):
        y, sr = torchaudio.load(file_path)
        y = self._preprocess_audio(y, sr)

        resampler = Resample(orig_freq=sr, new_freq=16000)
        y = resampler(torch.tensor(y, dtype=torch.float32))
        audio_tensor = self.audio_processor(y, sampling_rate=16000, return_tensors='pt')['input_values']
        return audio_tensor
    
    def _preprocess_audio(self, audio, sample_rate):
        audio = audio.reshape(-1,)
        normalized_audio = normalize_audio_lufs(y=audio, sr=sample_rate)
        preprocessed_audio = np.clip(normalized_audio, -1.0, 1.0)
        return preprocessed_audio
    
    def __getitem__(self, idx):
        video_clip_id, audio_file_path, labels = self.dataset[idx]
        audio = self._load_audio(audio_file_path)

        if self.annotation:
            labels = self._annotate(
                video_clip_id=video_clip_id, 
                annotation=self.annotation, 
                random_facet=self.annotate_random_facet,
                z_score=self.z_score
            )
        
        labels = torch.tensor(labels)
        if self.return_video_clip_id:
            return video_clip_id, audio, labels
        return audio, labels
    
    def __len__(self):
        return len(self.dataset)


# ------------------------------------------------ Transcription Dataset ------------------------------------------------ #


class KetiCorpusTranscriptionDataset(KetiCorpusDataset):

    def __init__(
            self, 
            data_path: str, 
            annotation_path: Optional[str] = None,
            tokenizer: str = "klue/roberta-base",
            max_len: int = 512,
            min_speech_length: float = 5.9,
            max_speech_length: float = 10.7,
            annotate_random_facet: bool = True,
            n_facets: int = 5,
            z_score: bool = False,
            max_facet_score: int = 20,
            return_video_clip_id: bool = False,
            video_clip_ids: Optional[List[str]] = None,
        ):
        self.data = pd.read_csv(data_path)
        self.annotation = pd.read_pickle(annotation_path) if annotation_path else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, never_split=(PAD, SEP, UNK, BOS, EOS), clean_up_tokenization_spaces=True
        )
        self.max_len = max_len
        self.min_speech_length = min_speech_length
        self.max_speech_length = max_speech_length
        self.annotate_random_facet = annotate_random_facet
        self.n_facets = n_facets
        self.z_score = z_score
        self.max_facet_score = max_facet_score
        self.return_video_clip_id = return_video_clip_id

        if video_clip_ids:
            self.data = self.data.loc[self.data['video_clip_id'].isin(video_clip_ids)]

        self.data = self.data.loc[
            (self.data['end'] - self.data['start'] >= min_speech_length) & 
            (self.data['end'] - self.data['start'] <= max_speech_length)
        ].sort_values('video_clip_id', ascending=True).reset_index(drop=True)
    
    def _preprocess_text(self, text):
        text = text.strip()
        # text = re.sub(r'[^a-zA-Z0-9가-힣\s?!]', '', text)
        return text

    def _tokenize_by_pretrained_model(self, tokenizer, transcription: str):
        tokens = tokenizer.tokenize(BOS + transcription + EOS)
        converted_ids = tokenizer.convert_tokens_to_ids(tokens)
        return converted_ids
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        video_clip_id = data['video_clip_id']
        text = self._preprocess_text(data['transcription'])
        token_ids = self._tokenize_by_pretrained_model(self.tokenizer, text)
        
        if self.annotation:
            labels = self._annotate(
                video_clip_id=video_clip_id, 
                annotation=self.annotation, 
                random_facet=self.annotate_random_facet,
                z_score=self.z_score
            )
        else:
            labels = [
                data['openness'], 
                data['conscientiousness'], 
                data['extraversion'], 
                data['agreeableness'], 
                data['neuroticism']
            ]

        labels = torch.tensor(labels)
        if self.return_video_clip_id:
            return video_clip_id, token_ids, labels
        return token_ids, labels
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        if self. return_video_clip_id:
            video_clip_id, token_ids, labels  = list(zip(*batch))
            tokens_tensor = pad_sequences(token_ids, max_length=self.max_len, batch_first=True)
            labels = torch.tensor(labels)
            return video_clip_id, tokens_tensor, labels
        
        token_ids, labels  = list(zip(*batch))
        tokens_tensor = pad_sequences(token_ids, max_length=self.max_len, batch_first=True)
        labels = torch.tensor(labels)
        return tokens_tensor, labels

# -------------------------------------------------------------------------------------------------------------------- #

def pad_sequences(sequences, max_length, batch_first=True):
    sequences = [torch.tensor(seq[:max_length]) for seq in sequences]
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
    return padded_sequences

def normalize_audio_lufs(y, sr, target_lufs=-23):
    y = np.nan_to_num(y, nan=0., posinf=0., neginf=0.)

    y_noise_reduced = nr.reduce_noise(y=y, sr=sr)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y_noise_reduced)
    y_normalized = pyln.normalize.loudness(y_noise_reduced, loudness, target_lufs)
    y_normalized = np.nan_to_num(y_normalized, nan=0., posinf=0., neginf=0.)
    return y_normalized

def find_file_paths(directory, extensions=['*.MP4', '*.avi', '*.mov', '*.mkv']):
    if not isinstance(extensions, (list, tuple)):
        extensions = [extensions]
    file_paths = []
    for ext in extensions:
        if not ext.startswith('*'):
            ext = f"*{ext}"
        pattern = os.path.join(directory, '**', ext).replace('\\', '/')
        file_paths.extend([file_path.replace('\\', '/') for file_path in glob.glob(pattern, recursive=True)])
    
    return sorted(file_paths)

def check_qt_chapter_error(video_path):
    """
    주어진 비디오 파일에서 "Referenced QT chapter track not found" 오류가 있는지 확인합니다.

    Parameters:
    video_path (str): 비디오 파일 경로.

    Returns:
    bool: 오류가 있을 경우 True, 없을 경우 False.
    """
    # ffmpeg로 파일 정보 확인
    result = subprocess.run(["ffmpeg", "-i", video_path], stderr=subprocess.PIPE, text=True)
    # 오류 메시지 확인
    if "Referenced QT chapter track not found" in result.stderr:
        print(f"Referenced QT chapter track not found: {video_path}")
        return True
    return False


if __name__ == "__main__":
    import random
    from torch.utils.data import DataLoader


    data_path = './data/train.csv'
    # video_dir = 'D:/PAI데이터/samples/train/frames'
    # audio_dir = 'D:/PAI데이터/samples/train/wav'
    # data_path = 'D:/PAI데이터/samples/train/transcription_train.pkl'
    video_dir = '/Volumes/p31/PAI데이터/samples/train/frames'
    audio_dir = '/Volumes/p31/PAI데이터/samples/train/wav'
    annotation_path = './data/annotation.pkl'

    print("KetiCorpusDataset Test Code\n")

    print("1. Dataset Test Code")
    print("1-1: Video Dataset")
    video_dataset = KetiCorpusVideoDataset(
        data_path=data_path, video_dir=video_dir, annotation_path=annotation_path, return_video_clip_id=True
    )

    index = random.randint(0, len(video_dataset) - 1)
    video_clip_id, frames, labels = video_dataset[index]
    print(f"video dataset length: {len(video_dataset)}")
    print(f"[{index}]: {video_clip_id}")
    print(f"Video Frame shape: {frames.shape}")
    print(f"Video Dataset Labels: {labels}\n")

    print("1-2: Audio Dataset")
    audio_dataset = KetiCorpusAudioDataset(
        data_path=data_path, audio_dir=audio_dir, annotation_path=annotation_path, return_video_clip_id=True
    )

    print(f"audio dataset length: {len(audio_dataset)}")
    index = random.randint(0, len(audio_dataset) - 1)
    video_clip_id, audio, labels = audio_dataset[index]
    print(f"[{index}]: {video_clip_id}")
    print(f"Audio shape: {audio.shape}")
    print(f"Audio Dataset Labels: {labels}\n")

    print("1-3: Transcription Dataset")
    text_dataset = KetiCorpusTranscriptionDataset(
        data_path=data_path, annotation_path=annotation_path, return_video_clip_id=True
    )

    print(f"text dataset length: {len(text_dataset)}")
    index = random.randint(0, len(text_dataset) - 1)
    video_clip_id, text, labels = text_dataset[index]
    print(f"[{index}]: {video_clip_id}")
    print(f"Text length: {len(text)}")
    print(f"Text Dataset Labels: {labels}\n")


    print("1-4: MultiModal Dataset")
    keti_dataset = KetiCorpusMultiModalDataset(
        data_path=data_path, 
        video_dir=video_dir,
        audio_dir=audio_dir,
        annotation_path=annotation_path, 
        return_video_clip_id=True
    )

    print(f"multimodal dataset length: {len(keti_dataset)}")
    # index = random.randint(0, len(keti_dataset) - 1)
    index = 4
    video_clip_id, frames, audio, text, labels = keti_dataset[index]
    print(f"[{index}]: {video_clip_id}")
    print(f"Video Frame shape: {frames.shape}")
    print(f"Audio shape: {audio.shape}")
    print(f"Text length: {len(text)}")
    print(f"MultiModal Dataset Labels: {labels}\n")

    print("2. DataLoader Test Code\n")
    print("2-1 Video Dataloader")
    video_dataloader = DataLoader(video_dataset, batch_size=2, shuffle=True)
    for idx, batch in enumerate(video_dataloader):
        video_clip_id, video_tensor, labels = batch
        print(f"[{idx}]: {video_clip_id} video shape: {video_tensor.shape} labels: {labels}")

    print("2-2 Audio Dataloader")
    audio_dataloader = DataLoader(audio_dataset, batch_size=2, shuffle=True)
    for idx, batch in enumerate(audio_dataloader):
        video_clip_id, audio_tensor, labels = batch
        print(f"[{idx}]: {video_clip_id} audio shape: {audio_tensor.shape} labels: {labels}")

    print("2-3 Text Dataloader")
    # text_dataloader = DataLoader(text_dataset, batch_size=2, shuffle=True, collate_fn=text_dataset.collate_fn)
    # for idx, batch in enumerate(text_dataloader):
    #     video_clip_id, text_tensor, labels = batch
    #     print(f"[{idx}]: {video_clip_id} text shape: {text_tensor.shape} labels: {labels}")

    print("2-4 MultiModal Dataloader")
    keti_dataloader = DataLoader(keti_dataset, batch_size=2, shuffle=True, collate_fn=keti_dataset.collate_fn)
    for idx, batch in enumerate(keti_dataloader):
        video_clip_id, video_tensor, audio_tensor, text_tensor, labels = batch
        print(f"[{idx}]: {video_clip_id}")
        print(f"video shape: {video_tensor.shape} audio shape: {audio_tensor.shape} text shape: {text_tensor.shape} labels: {labels}")