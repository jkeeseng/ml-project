import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
import numpy as np
from evaluate_w2v_classifier import Wav2Vec2ForSpeechClassification

def speech_file_to_array_fn(path, target_sampling_rate):
    speech_array, sampling_rate = torchaudio.load(path)
    if sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sampling_rate)
        speech_array = resampler(speech_array)
    return speech_array.squeeze()

def predict(model, config, feature_extractor, device, path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    features = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    # Find the index of the max score
    max_idx = np.argmax(scores)
    # Create output with only the label with the highest score
    output = {"Label": config.id2label[max_idx], "Score": f"{scores[max_idx] * 100:.1f}%"}
    return output


def inference_w2v_model(model_name_or_path, wav_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    sampling_rate = feature_extractor.sampling_rate
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

    prediction = predict(model, config, feature_extractor, device, wav_path, sampling_rate)

    return prediction

if __name__ == "__main__":
    inference_w2v_model(model_name_or_path='trained_models/facebook_wav2vec2-xlsr-53-phon-cv-babel-ft/checkpoint-7090',
                        wav_path='data/filtered_dataset_sepedi/wav/A/0a355431-d0b2-4aa9-b087-1e28d0d1635b.wav')