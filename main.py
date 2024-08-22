# %%
from PIL import Image
import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.optim as optim
import vision_transformer as vits
from torch import nn
import utils
from torch.optim import AdamW
from transformers import ViTModel, ViTConfig
import numpy as np
import cv2
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm
from transformers import ViTImageProcessor
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import Trainer, TrainingArguments
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments
import wandb
from sklearn.metrics import confusion_matrix, classification_report

os.environ["WANDB_WATCH"] = "all"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128

class UnNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
 
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class CustomDataset(Dataset):
    def __init__(self, frames_dir, transform=None):
        self.frames_dir = frames_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(
            mean=torch.tensor([0.4850, 0.4560, 0.4060]),
            std=torch.tensor([0.2290, 0.2240, 0.2250]))])


        self.unnormalize = UnNormalize(
            mean=torch.tensor([0.4850, 0.4560, 0.4060]),
            std=torch.tensor([0.2290, 0.2240, 0.2250]),
        )

        self.data = []

        for folder_name in os.listdir(self.frames_dir):
            folder_path = os.path.join(self.frames_dir, folder_name)
            if os.path.isdir(folder_path):
                label = 1 if folder_name == "polyp" else 0
                for frame_file in os.listdir(folder_path):
                    frame_path = os.path.join(folder_path, frame_file)
                    if frame_file.endswith(".jpg"):
                        self.data.append((frame_path, label))

    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        frame_path, label = self.data[idx]
        image = Image.open(frame_path)
        image = self.transform(image)
        sample = {"pixel_values": image, "labels": label}
        return sample

frames_path = "test_colon_polyp"
custom_dataset = CustomDataset(frames_path)

train_data, test_data = train_test_split(custom_dataset.data, test_size=0.2, random_state=42)

train_dataset = CustomDataset(frames_path, transform=None)
train_dataset.data = train_data

test_dataset = CustomDataset(frames_path, transform=None)
test_dataset.data = test_data

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

def get_encoder(path, device):
    state_dict = torch.load(path, map_location="cpu")
    state_dict = state_dict["teacher"]

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    encoder = vits.__dict__["vit_base"](patch_size=8, num_classes=0)
    print(encoder.load_state_dict(state_dict, strict=False))

    encoder.eval()
    encoder.to(device)

    return encoder

encoder = get_encoder(encoder_path, device)

class Model(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.dropout1 = nn.Dropout(0.5)

        embedding_dim = 768
        self.classifier = nn.Linear(embedding_dim, num_classes)
        weight = torch.tensor([0.5788, 3.6725])
        self.loss = nn.CrossEntropyLoss(weight=weight)

        for param in self.encoder.parameters(): #freezing the weights
            param.requires_grad = False

    def forward(self, pixel_values, labels):
        embeddings = self.encoder(pixel_values)
        embeddings = self.dropout1(embeddings)
        logits = self.classifier(embeddings)
        loss_result = self.loss(logits.view(-1, self.num_classes), labels.view(-1))
        return {'output': logits, 'loss': loss_result}

model = Model(encoder).to(device)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    tn, fp, fn, tp = confusion_matrix(p.label_ids, preds).ravel()
    f1 = f1_score(p.label_ids, preds, average="weighted")
    f1_bin = f1_score(p.label_ids, preds, average="binary")
    precision = precision_score(p.label_ids, preds, average="weighted")
    recall = recall_score(p.label_ids, preds, average="weighted")
    f1_normal = f1_score(p.label_ids, preds, average=None)[0]
    f1_ach = f1_score(p.label_ids, preds, average=None)[1]
    accuracy = accuracy_score(p.label_ids, preds)
    creport = classification_report(p.label_ids, preds)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "f1_ach/abn": f1_ach,
        "f1_normal": f1_normal,
        "accuracy": accuracy,
        "f1_bin": f1_bin,
    }

training_args = TrainingArguments(
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=10,
    learning_rate=0.0002,
    weight_decay=0.002, #increased from 0.001
    evaluation_strategy = "epoch",
    save_strategy= "epoch", # added new for early_stopping_callback
    report_to = "wandb",
    logging_strategy="steps",
    logging_steps = 10,
    fp16 = True,
    lr_scheduler_type="cosine", #changed to cosine from linear
    warmup_steps=500,
    gradient_accumulation_steps=2, #added new
    load_best_model_at_end=True # added new
    )

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics = compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )

trainer.train()
