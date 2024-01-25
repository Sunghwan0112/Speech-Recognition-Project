"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        DATA AUGMENTATION WITH VARIOUS DISTORTIONS PREPARED BY        ║
║                          TORCHAUDIO                                  ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║ Description: This script is designed to augment audio data from the  ║
║ LibriSpeech dataset. It includes pitch shifting, speed control,      ║
║ noise addition, and SpecAugment for robustness enhancement in ASR    ║
║ (Automatic Speech Recognition) models.                               ║
║                                                                      ║
║ Team: Sunghwan Baek & Xiwen Chen & Matthew Saenz                     ║
║                                                                      ║
║ Date: 11/17/2023                                                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import random
import torch
import torchaudio
import librosa
import numpy as np

# I haven't used Librosa for pitch shifting, but you can try. (You need to download the library first)

# ------------------ Configuration and Initial Setup ------------------ #

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You should make your directory.
# Paths for dataset and augmented data directories
dataset_dir = "/ocean/projects/...../yourID/espnet/egs2/librispeech_100/asr1/downloads/LibriSpeech/train-clean-100"
augmented_dir = "/ocean/projects/...../yourID/espnet/egs2/librispeech_100/asr1/downloads/LibriSpeech/train-augmented"

# ------------------ Augmentation Function Definitions ------------------ #

def pitch_shift(waveform, sample_rate, n_steps):
    """
    Apply pitch shifting to the audio.
    :param waveform: Input waveform tensor.
    :param sample_rate: Sample rate of the audio.
    :param n_steps: Steps for pitch shifting.
    :return: Pitch-shifted waveform tensor.
    """
    waveform_np = waveform.cpu().numpy()[0]  
    shifted_waveform_np = librosa.effects.pitch_shift(waveform_np, sample_rate, n_steps)
    return torch.from_numpy(shifted_waveform_np).unsqueeze(0).to(device)  # Convert back to tensor and move to GPU

def change_speed(waveform, sample_rate, speed_factor):
    """
    Change the speed of the audio.
    :param waveform: Input waveform tensor.
    :param sample_rate: Sample rate of the audio.
    :param speed_factor: Factor by which to change the speed.
    :return: Speed-altered waveform tensor.
    """
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=int(sample_rate * speed_factor)).to(device)
    return resampler(waveform)

def add_noise(waveform, noise_level=0.005):
    """
    Add noise to the audio.
    :param waveform: Input waveform tensor.
    :param noise_level: Noise amplitude factor.
    :return: Noise-added waveform tensor.
    """
    noise_amp = noise_level * random.uniform(0.1, 0.3)
    noise = noise_amp * torch.randn(waveform.shape, device=device)
    return waveform + noise

def apply_spec_augment(waveform, sample_rate, n_fft=400, freq_mask_param=20, time_mask_param=30):
    """
    Apply SpecAugment to the audio.
    :param waveform: Input waveform tensor.
    :param sample_rate: Sample rate of the audio.
    :param n_fft: FFT size for spectrogram.
    :param freq_mask_param: Frequency mask parameter.
    :param time_mask_param: Time mask parameter.
    :return: SpecAugmented waveform tensor.
    """
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft).to(device)
    spectrogram = spectrogram_transform(waveform)

    frequency_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param).to(device)
    time_masking = torchaudio.transforms.TimeMasking(time_mask_param).to(device)
    
    augmented_spectrogram = frequency_masking(spectrogram)
    augmented_spectrogram = time_masking(augmented_spectrogram)

    inverse_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft).to(device)
    return inverse_transform(augmented_spectrogram)

# ------------------ Main Augmentation Process ------------------ #

def augment_audio(file_path, save_path):
    """
    Perform audio augmentation on a given file and save the output.
    :param file_path: Path to the original audio file.
    :param save_path: Path to save the augmented audio file.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.to(device)  # Move waveform to GPU if available

    # Apply augmentations
    waveform = add_noise(waveform)
    waveform = apply_spec_augment(waveform, sample_rate)

    # Save the augmented file
    augmented_file_name = os.path.basename(file_path)
    save_dir = os.path.join(save_path, os.path.relpath(os.path.dirname(file_path), dataset_dir))
    os.makedirs(save_dir, exist_ok=True)
    torchaudio.save(os.path.join(save_dir, augmented_file_name), waveform.cpu(), sample_rate,
                    format="flac")  # Move back to CPU for saving


# ------------------ Directory Checking and File Processing ------------------ #

# Check if dataset directory exists
if not os.path.exists(dataset_dir):
    raise Exception(f"Dataset directory does not exist: {dataset_dir}")

# Create augmented directory if it doesn't exist
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Process each audio file in the dataset directory
for subdir, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".flac") or file.endswith(".wav"):
            file_path = os.path.join(subdir, file)
            augment_audio(file_path, augmented_dir)