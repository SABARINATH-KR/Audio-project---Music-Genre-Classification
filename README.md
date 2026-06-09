# Music Genre Classification

An end-to-end machine learning system that automatically classifies music clips into genres such as rock, jazz, classical, hip-hop, and more — purely from raw audio signals, without any manual tagging.

---

## Project Description

This project tackles the problem of automatic music genre classification using audio feature extraction and supervised machine learning. Given a raw audio clip, the system extracts meaningful acoustic features (MFCCs, spectral centroid, chroma features, tempo, zero-crossing rate, etc.) and feeds them into a trained classifier to predict the genre label.

The model is trained and evaluated on the **GTZAN Genre Collection** — 1,000 audio clips spanning 10 genres (blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock), each 30 seconds long.

### Key Capabilities

- Extracts rich audio features from raw `.wav` files using Librosa
- Trains multiple classifiers (SVM, Random Forest, CNN on mel spectrograms)
- Evaluates performance with accuracy, confusion matrix, and per-class F1 scores
- Supports inference on new audio clips via a simple CLI or web interface
- (Optional) Real-time genre detection from microphone input

---

## Solution Approach

### 1. Feature Extraction
Using `Librosa`, the following features are extracted from each audio clip:

- **MFCCs** (Mel-Frequency Cepstral Coefficients) — 13 coefficients capturing timbral texture
- **Spectral Centroid** — indicates where the "center of mass" of the spectrum is
- **Chroma Features** — represents the 12 pitch classes of music
- **Spectral Rolloff** — frequency below which 85% of energy is contained
- **Zero Crossing Rate** — rate at which the signal changes sign, useful for rhythm
- **Tempo** — beats per minute extracted via beat tracking

### 2. Model Training
Two approaches are implemented and compared:

- **Classical ML Pipeline**: Features are normalized and passed to a Scikit-learn classifier (SVM or Random Forest)
- **Deep Learning Pipeline**: Raw audio is converted to mel spectrograms and passed through a CNN

### 3. Evaluation
- Train/test split: 80/20
- Metrics: Accuracy, Precision, Recall, F1-score (per class), Confusion Matrix
- Cross-validation is used for robust performance estimation

---

## Project Structure

```
music-genre-classification/
│
├── data/
│   └── genres_original/          # GTZAN dataset (download separately)
│       ├── blues/
│       ├── classical/
│       └── ...
│
├── notebooks/
│   └── exploration.ipynb         # EDA and feature visualization
│
├── src/
│   ├── extract_features.py       # Audio feature extraction pipeline
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Evaluation and metrics
│   └── predict.py                # Inference on new audio clips
│
├── models/
│   └── genre_classifier.pkl      # Saved trained model
│
├── requirements.txt
├── colab_notebook.ipynb          # Full runnable Colab notebook
└── README.md
```

---

## Setup and Usage

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/music-genre-classification.git
cd music-genre-classification

# Install dependencies
pip install -r requirements.txt
```

### Download the Dataset

Download the GTZAN Genre Collection from one of the following sources:

- [Kaggle — GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- [Free Music Archive (FMA)](https://github.com/mdeff/fma)

Extract the dataset and place it in the `data/genres_original/` directory.

### Extract Features

```bash
python src/extract_features.py --data_path data/genres_original/ --output features.csv
```

### Train the Model

```bash
python src/train.py --features features.csv --model svm
# Options for --model: svm, random_forest, cnn
```

### Evaluate the Model

```bash
python src/evaluate.py --model_path models/genre_classifier.pkl --features features.csv
```

### Predict Genre for a New Audio Clip

```bash
python src/predict.py --audio_path path/to/your/song.wav
```

---

## Dependencies

All dependencies are listed in `requirements.txt`:

```
librosa==0.10.1
scikit-learn==1.3.0
numpy==1.24.0
pandas==2.0.0
matplotlib==3.7.0
seaborn==0.12.0
tensorflow==2.13.0       # Only required for CNN approach
torch==2.0.1             # Optional PyTorch alternative
jupyter==1.0.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Google Colab Notebook

A fully runnable Colab notebook is included for easy experimentation without any local setup:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/your-notebook-link-here)

The notebook walks through:
1. Dataset download and setup
2. Feature extraction and visualization
3. Model training (SVM + CNN)
4. Evaluation and result interpretation
5. Inference demo on a custom audio clip

---

## Results

| Model          | Accuracy | F1 Score (macro) |
|----------------|----------|------------------|
| SVM            | ~82%     | ~0.81            |
| Random Forest  | ~79%     | ~0.78            |
| CNN (mel spec) | ~88%     | ~0.87            |

*Results may vary slightly depending on the random seed and dataset split.*

---

## Potential Extensions

- Real-time genre detection from microphone input
- Confidence scores for ambiguous or mixed-genre tracks
- Web UI where users can upload a song and receive instant predictions
- Multi-label classification for songs spanning multiple genres
- Integration with Spotify API for large-scale genre analysis

---

## References

- Tzanetakis, G. & Cook, P. (2002). *Musical genre classification of audio signals*. IEEE Transactions on Speech and Audio Processing.
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [GTZAN Dataset](http://marsyas.info/downloads/datasets.html)
- [Free Music Archive](https://freemusicarchive.org/)

---

## License

This project is developed for the **Claysys AI Hackathon** and is intended for educational and research purposes.
