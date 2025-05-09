import os

# Define the folder structure
folders = [
    "WildlifeSpeciesPrediction/data/raw/audio",
    "WildlifeSpeciesPrediction/data/processed/spectrograms",
    "WildlifeSpeciesPrediction/notebooks",
    "WildlifeSpeciesPrediction/src/preprocessing",
    "WildlifeSpeciesPrediction/src/models",
    "WildlifeSpeciesPrediction/src/train",
    "WildlifeSpeciesPrediction/src/evaluate",
    "WildlifeSpeciesPrediction/outputs/models",
    "WildlifeSpeciesPrediction/outputs/logs",
    "WildlifeSpeciesPrediction/outputs/predictions",
    "WildlifeSpeciesPrediction/scripts"
]

# Create each folder
for folder in folders:
    os.makedirs(folder, exist_ok=True)

"Folder structure created."

python folderstructring.py
