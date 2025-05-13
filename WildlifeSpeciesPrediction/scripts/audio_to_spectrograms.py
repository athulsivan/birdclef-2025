import os
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# INPUT: Base audio folder
AUDIO_DIR = Path(r"C:\Users\athul\Desktop\birdclef-2023\train_audio\abethr1")
# OUTPUT: Spectrogram folder
SPEC_DIR = Path("data/processed/spectrograms")
SPEC_DIR.mkdir(parents=True, exist_ok=True)

def save_mel_spectrogram(audio_path, out_path, sr=22050, n_mels=128):
    try:
        y, _ = librosa.load(audio_path, sr=sr)
        y, _ = librosa.effects.trim(y)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot and save as 224x224 image
        plt.figure(figsize=(2.24, 2.24), dpi=100)  # 224x224 pixels
        librosa.display.specshow(S_dB, sr=sr, cmap='magma')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Failed to process {audio_path}: {e}")

# Recursively find all .ogg files and convert
ogg_files = list(AUDIO_DIR.rglob("*.ogg"))
print(f"Found {len(ogg_files)} .ogg files")

for ogg_path in tqdm(ogg_files, desc="Processing spectrograms"):
    species = ogg_path.parent.name
    output_folder = SPEC_DIR / species
    output_folder.mkdir(parents=True, exist_ok=True)

    out_path = output_folder / f"{ogg_path.stem}.png"
    # print the path of the file being processed
    print(f"Processing {ogg_path} to {out_path}")
    save_mel_spectrogram(ogg_path, out_path)
