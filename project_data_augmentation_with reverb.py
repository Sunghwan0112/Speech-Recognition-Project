"""
-----------------------------------------------------------------------------------------
Project Name: Advanced Data Augmentation for ASR with Torchaudio
-----------------------------------------------------------------------------------------
Description:
    This script enhances the robustness of ASR (Automatic Speech Recognition) models
    by augmenting audio data from the LibriSpeech dataset. It integrates a suite of
    sophisticated techniques including pitch shifting, speed control, noise injection,
    reverb augmentation using Room Impulse Responses (RIR), and SpecAugment. This
    diverse set of distortions aims to mimic real-world acoustic variations, thereby
    improving the resilience and accuracy of ASR models.

Author: Sunghwan Baek
Date: December 1st, 2023
Contact: sunghwab@andrew.cmu.edu
-----------------------------------------------------------------------------------------
"""
import os
import random
import torch
import torchaudio
import librosa
import numpy as np

# ------------------ Configuration and Initial Setup ------------------ #

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for dataset and augmented data directories
dataset_dir = "/path/espnet/egs2/librispeech_100/asr1/downloads/LibriSpeech/train-clean-100"
augmented_dir = "/path/espnet/egs2/librispeech_100/asr1/downloads/LibriSpeech/train-augmented"

# Load RIR files
rir_files = [
    "/path/espnet/egs2/librispeech_100/asr1/smallroom/Room001/Room001-00001.wav",
    "/path/espnet/egs2/librispeech_100/asr1/smallroom/Room001/Room001-00002.wav",
    "/path/espnet/egs2/librispeech_100/asr1/smallroom/Room001/Room001-00003.wav",
    "/path/espnet/egs2/librispeech_100/asr1/smallroom/Room001/Room001-00004.wav",
    "/path/espnet/egs2/librispeech_100/asr1/smallroom/Room001/Room001-00005.wav",
    "/path/espnet/egs2/librispeech_100/asr1/smallroom/Room001/Room001-00006.wav",
    "/path/espnet/egs2/librispeech_100/asr1/smallroom/Room001/Room001-00007.wav",
    "/path/espnet/egs2/librispeech_100/asr1/smallroom/Room001/Room001-00008.wav",
    "/path/espnet/egs2/librispeech_100/asr1/smallroom/Room001/Room001-00009.wav",
    "/path/espnet/egs2/librispeech_100/asr1/smallroom/Room001/Room001-00010.wav"
]

rir_waveforms = []
for rir_file in rir_files:
    rir_waveform, _ = torchaudio.load(rir_file)
    rir_waveform = rir_waveform.to(device)
    # Process the RIR (e.g., normalize, ensure mono)
    rir_waveform = rir_waveform / torch.norm(rir_waveform, p=2)
    if rir_waveform.shape[0] > 1:
        rir_waveform = torch.mean(rir_waveform, dim=0, keepdim=True)
    rir_waveforms.append(rir_waveform)

# ------------------ Augmentation Function Definitions ------------------ #
def add_reverb(waveform, rir_waveforms, mix_ratio=0.15):
    """
    Add reverb to the audio using a randomly selected Room Impulse Response (RIR) and mix it with the original signal.
    :param waveform: Input waveform tensor.
    :param rir_waveforms: List of RIR waveforms.
    :param mix_ratio: Ratio of reverberated signal to be mixed with the original signal.
    :return: Waveform tensor with reverb mixed.
    """
    rir_waveform = random.choice(rir_waveforms)

    # Convolve the waveform with the RIR
    reverb_waveform = torch.nn.functional.conv1d(waveform[None, ...], rir_waveform[None, ...]).squeeze(0)

    # Match the length of the reverb waveform to the original waveform
    original_length = waveform.shape[1]
    reverb_length = reverb_waveform.shape[1]

    if reverb_length > original_length:
        # Trim the reverb waveform if it's longer than the original
        reverb_waveform = reverb_waveform[:, :original_length]
    elif reverb_length < original_length:
        # Pad the reverb waveform if it's shorter than the original
        padding_size = original_length - reverb_length
        reverb_waveform = torch.nn.functional.pad(reverb_waveform, (0, padding_size))

    # Mix the original and the reverberated waveforms
    mixed_waveform = (1 - mix_ratio) * waveform + mix_ratio * reverb_waveform

    return mixed_waveform



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
    waveform = add_reverb(waveform, rir_waveforms) 
    waveform = apply_spec_augment(waveform, sample_rate)

    # Save the augmented file
    augmented_file_name = os.path.basename(file_path)
    save_dir = os.path.join(save_path, os.path.relpath(os.path.dirname(file_path), dataset_dir))
    os.makedirs(save_dir, exist_ok=True)
    torchaudio.save(os.path.join(save_dir, augmented_file_name), waveform.cpu(), sample_rate, format="flac")  # Move back to CPU for saving

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