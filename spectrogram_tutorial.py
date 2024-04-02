import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load raw audio waveform
audio_file = r"D:\Sean\DCASE\datasets\Extract_to_Folder\TAU-urban-acoustic-scenes-2022-mobile-development\audio\airport-barcelona-0-0-0-a.wav"
y, sr = librosa.load(audio_file)

# Varying parameters
n_fft_values = [2048, 1024]  # Different values for n_fft
hop_length_values = [512, 256]  # Different values for hop_length
n_mels_values = [64, 128]  # Different values for n_mels

# Plotting
fig, axes = plt.subplots(nrows=len(n_fft_values), ncols=len(hop_length_values), figsize=(12, 8))

for i, n_fft in enumerate(n_fft_values):
    for j, hop_length in enumerate(hop_length_values):
        # Calculate Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)

        # Convert to log scale
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Display the Log Mel spectrogram
        librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[i, j])
        axes[i, j].set_title(f'n_fft={n_fft}, hop_length={hop_length}')

plt.tight_layout()
plt.show()
