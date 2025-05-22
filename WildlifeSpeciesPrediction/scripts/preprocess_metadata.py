
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load metadata CSV
df = pd.read_csv(r"C:\Users\athul\Desktop\birdclef-2023\train_metadata.csv")

# # Convert date/time columns
# df['date'] = pd.to_datetime(df['date'])
# df['dayofyear'] = df['date'].dt.dayofyear
# df['month'] = df['date'].dt.month

# df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
# df['hour'] = df['time'].dt.hour

# Encode species label
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['scientific_name'])

# Features to scale
features = ['latitude', 'longitude']
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Final dataframe
df_final = pd.concat([df[['filename', 'scientific_name', 'label']], df_scaled], axis=1)

# Optional: Add path to spectrogram image
df_final['spectrogram_path'] = df_final['filename'].apply(
    lambda f: str((Path.cwd() / "data/processed/spectrograms" / Path(f).parent.name / Path(f).with_suffix(".png").name).resolve())
    
)


# Save processed version
df_final.to_csv("data/processed/metadata_processed.csv", index=False)
print("Metadata preprocessing complete.")
