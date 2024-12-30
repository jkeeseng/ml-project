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
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import classification_report
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
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

# Check for native AMP support in PyTorch
if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

@dataclass
class SpeechClassifierOutput(ModelOutput):
    """
    Dataclass for speech classification outputs.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    """
    Classification head for Wav2Vec2.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """
    Wav2Vec2 model for speech classification.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True
    ) -> Union[Tuple, SpeechClassifierOutput]:
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state
        if self.pooling_mode == "mean":
            hidden_states = hidden_states.mean(dim=1)
        elif self.pooling_mode == "max":
            hidden_states, _ = hidden_states.max(dim=1)
        else:
            hidden_states = hidden_states[:, 0, :]

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif labels.dtype == torch.long:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            return ((loss,) + (logits,) + outputs[2:]) if loss is not None else (logits,) + outputs[2:]

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

class Wav2Vec2Evaluate:
    """
    Evaluation class for Wav2Vec2 models.
    """
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def evaluate(self, dataset: pd.DataFrame, metric: Any) -> Dict[str, Any]:
        """
        Evaluate the model on a given dataset using the specified metric.

        Args:
            dataset (pd.DataFrame): Dataset containing inputs and labels.
            metric (Any): Evaluation metric (e.g., accuracy, F1-score).

        Returns:
            Dict[str, Any]: Evaluation results.
        """
        predictions = []
        references = []

        for index, row in dataset.iterrows():
            inputs = self.processor(row['audio'], sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)

            predictions.append(predicted_ids.cpu().numpy())
            references.append(row['label'])

        return metric.compute(predictions=predictions, references=references)