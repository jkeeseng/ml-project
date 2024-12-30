import pandas as pd
import torchaudio
import librosa
import numpy as np
from datasets import load_dataset, load_metric
from dataclasses import dataclass
import torch
from transformers.file_utils import ModelOutput
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import transformers
from transformers import (
    AutoConfig, 
    Trainer, 
    Wav2Vec2Processor, 
    Wav2Vec2FeatureExtractor, 
    EvalPrediction, 
    TrainingArguments
)
from packaging import version

# Check for AMP support based on PyTorch and Transformers version
_is_native_amp_available = (
    version.parse(transformers.__version__) >= version.parse("4.0.0") and
    version.parse(torch.__version__) >= version.parse("1.6")
)
if _is_native_amp_available:
    from torch.cuda.amp import autocast

class CTCTrainer(Trainer):
    """
    Custom Trainer for CTC models.
    """
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            model (nn.Module): The model being trained.
            inputs (Dict[str, Union[torch.Tensor, Any]]): The inputs and targets.

        Returns:
            torch.Tensor: Training loss for the step.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        # Gradient accumulation adjustment
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Backpropagation
        if self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss

    def compute_loss(self, model, inputs):
        """
        Compute loss by forwarding inputs through the model.

        Args:
            model (nn.Module): The model being trained.
            inputs (Dict): Inputs to the model.

        Returns:
            torch.Tensor: Computed loss.
        """
        labels = inputs.pop("labels")

        with autocast(enabled=_is_native_amp_available):
            outputs = model(**inputs)
            logits = outputs.logits

        if self.label_smoother:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = nn.functional.cross_entropy(logits, labels)

        return loss

# Additional utility functions and classes can be added here as needed.

# Define a callable function to instantiate and train the model
def train_model(model, processor, train_dataset, eval_dataset, training_args):
    """
    Train the Wav2Vec2 model using a custom trainer.

    Args:
        model (nn.Module): The model to train.
        processor (Wav2Vec2Processor): Processor for data preparation.
        train_dataset (Dataset): Training dataset.
        eval_dataset (Dataset): Evaluation dataset.
        training_args (TrainingArguments): Arguments for the training process.

    Returns:
        Trainer: The trained model.
    """
    trainer = CTCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor
    )

    trainer.train()
    return trainer

def run_speech_classification_training(
    model_name_or_path: str,
    train_data_path: str,
    eval_data_path: str,
    output_dir: str,
    num_labels: int,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int
):
    """
    End-to-end function to run speech classification training.

    Args:
        model_name_or_path (str): Path to the pre-trained model.
        train_data_path (str): Path to the training dataset.
        eval_data_path (str): Path to the evaluation dataset.
        output_dir (str): Directory to save trained models.
        num_labels (int): Number of output labels for classification.
        num_train_epochs (int): Number of training epochs.
        per_device_train_batch_size (int): Batch size for training.
        per_device_eval_batch_size (int): Batch size for evaluation.

    Returns:
        Trainer: Trained model.
    """
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
    model = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)

    train_dataset = load_dataset("csv", data_files=train_data_path, split="train")
    eval_dataset = load_dataset("csv", data_files=eval_data_path, split="train")

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs"
    )

    trainer = train_model(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args
    )

    return trainer