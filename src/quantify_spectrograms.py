import pandas as pd
import numpy as np
from pathlib import Path
import librosa

AUDIO_ROOT = Path(r"C:\Users\athul\Desktop\birdclef-2023\train_audio")  # your path
METADATA_IN = Path("data/raw/train_metadata.csv")
FEATURES_OUT = Path("data/processed/audio_features.csv")

def extract_features(audio_path, sr_target=None):
    y, sr = librosa.load(audio_path, sr=sr_target)
    y, _ = librosa.effects.trim(y)

    # MFCCs (mean + std of first 13)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std  = mfcc.std(axis=1)

    # Pitch (fundamental) via YIN (more robust than piptrack)
    f0 = librosa.yin(y, fmin=50, fmax=8000, sr=sr)
    pitch_mean = np.nanmean(f0)
    pitch_std  = np.nanstd(f0)

    # “Formant-ish” proxies (true formants need LPC/Praat; we use spectral shape):
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()

    # Temporal/rhythmic patterns
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()

    # Chroma (harmonic profile), summarized
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Pack features
    feats = {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "spec_centroid": spec_centroid,
        "spec_bandwidth": spec_bandwidth,
        "spec_rolloff": spec_rolloff,
        "tempo": tempo,
        "zcr": zcr,
        "rms": rms,
    }
    # Add MFCCs and chroma summaries
    feats.update({f"mfcc_mean_{i+1}": v for i, v in enumerate(mfcc_mean)})
    feats.update({f"mfcc_std_{i+1}": v for i, v in enumerate(mfcc_std)})
    feats.update({f"chroma_mean_{i+1}": v for i, v in enumerate(chroma_mean)})

    return feats

def main():
    df = pd.read_csv(METADATA_IN)
    rows = []
    for _, r in df.iterrows():
        # BirdCLEF filenames live under subfolder = primary_label
        audio_path = AUDIO_ROOT / r["primary_label"] / r["filename"]
        if not audio_path.exists():
            continue
        feats = extract_features(audio_path)
        feats["filename"] = r["filename"]
        feats["primary_label"] = r["primary_label"]
        rows.append(feats)

    out = pd.DataFrame(rows)
    out.to_csv(FEATURES_OUT, index=False)
    print(f"Saved {len(out)} rows -> {FEATURES_OUT}")

if __name__ == "__main__":
    main()
