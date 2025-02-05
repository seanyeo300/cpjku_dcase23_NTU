import os
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# Directories for datasets
dir1 = r"D:\Sean\CochlScene\audio"  # Directory containing "street" files
dir2 = r"D:\Sean\DCASE\datasets\Extract_to_Folder\TAU-urban-acoustic-scenes-2022-mobile-development\audio"  # Directory containing "street_pedestrian" and "street_traffic" files

# Parameters for Mel-spectrogram conversion
window_size = 800
hop_size = 320
n_fft = 1024
n_mels = 128

# Limit per class
MAX_SAMPLES_PER_CLASS = 5000

def load_audio_files(directory, label_extraction_fn, max_samples, selected_classes):
    data = []
    labels = []
    label_counts = {}
    targets = selected_classes
    for file in tqdm(os.listdir(directory), desc=f"Processing {directory}"):
        if file.endswith(".wav"):
            filepath = os.path.join(directory, file)
            label = label_extraction_fn(file)
            
            if label not in label_counts:
                label_counts[label] = 0

            if label_counts[label] < max_samples and label in selected_classes:
                waveform, sr = torchaudio.load(filepath)
                if sr != 44100:
                    raise ValueError(f"Unexpected sampling rate {sr} in file {filepath}")

                # Compute Mel-spectrogram
                mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr,
                    n_fft=n_fft,
                    win_length=window_size,
                    hop_length=hop_size,
                    n_mels=n_mels
                )(waveform)

                data.append(mel_spectrogram.mean(dim=-1).squeeze().numpy())
                labels.append(label)
                label_counts[label] += 1

    return data, labels

def extract_label_dir1(filename):
    """Extract label for files in dir1 using underscores."""
    return filename.split("_")[0]

def extract_label_dir2(filename):
    """Extract label for files in dir2 using hyphens."""
    return filename.split("-")[0]

# Load data from both directories
data1, labels1 = load_audio_files(dir1, extract_label_dir1, MAX_SAMPLES_PER_CLASS,["Street"])
data2, labels2 = load_audio_files(dir2, extract_label_dir2, MAX_SAMPLES_PER_CLASS, ["street_pedestrian","street_traffic"])

# Combine data and labels
data = np.array(data1 + data2)
labels = np.array(labels1 + labels2)

# Perform t-SNE
t_sne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
data_2d = t_sne.fit_transform(data)

# Plot t-SNE result
plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    indices = labels == label
    plt.scatter(data_2d[indices, 0], data_2d[indices, 1], label=label, alpha=0.7)
plt.legend()
plt.title("t-SNE Visualisation of Acoustic Scenes")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
plt.show()
