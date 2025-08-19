# AI Music Generation - Bach Chorales

A sophisticated deep learning system that generates Bach-style four-part chorales using a hybrid Conv1D-LSTM neural network architecture. The model learns from historical Bach chorales and creates new harmonically coherent compositions.

## Author
**Macha Praveen**

## Overview

This project implements an advanced neural network architecture specifically designed for polyphonic music generation. By combining 1D Convolutional Neural Networks with LSTM layers, the system captures both local harmonic patterns and long-term musical dependencies to generate Bach-style chorales with authentic baroque characteristics.

## Features

- **Multi-voice Polyphonic Generation**: Creates 4-part chorales (Soprano, Alto, Tenor, Bass)
- **Advanced Neural Architecture**: 
  - Embedding layers for musical note representation learning
  - Dilated 1D Convolutional layers with increasing receptive fields (2, 4, 8, 16)
  - Batch normalization for training stability
  - LSTM layers for long-term musical sequence modeling
- **Musical Data Processing**: Handles MIDI note values (36-81) with silence representation
- **Model Persistence**: Save and load trained models for inference
- **Interactive Playback**: Generate MIDI files for audio playback using Music21

## Technical Architecture

### Model Structure
The neural network combines several sophisticated components:

```python
# Embedding layer learns 5-dimensional representations for 47 possible note values
model.add(Embedding(input_dim=47, output_dim=5))

# Dilated 1D Convolutions capture patterns at multiple time scales
model.add(Conv1D(32, kernel_size=2, padding="causal", activation="relu"))
model.add(Conv1D(48, kernel_size=2, padding="causal", activation="relu", dilation_rate=2))
model.add(Conv1D(64, kernel_size=2, padding="causal", activation="relu", dilation_rate=4))
model.add(Conv1D(96, kernel_size=2, padding="causal", activation="relu", dilation_rate=8))
model.add(Conv1D(128, kernel_size=2, padding="causal", activation="relu", dilation_rate=16))

# LSTM for sequence modeling
model.add(LSTM(256, return_sequences=True))

# Output layer predicts next note probabilities
model.add(Dense(47, activation='softmax'))
```

## Technology Stack

- **TensorFlow/Keras**: Deep learning framework for model implementation
- **Music21**: Musical data processing and MIDI generation
- **Pandas**: Data manipulation and CSV processing
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization and analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd "AI Music Generation"

# Install required packages
pip install tensorflow pandas numpy music21 matplotlib
```

## Usage

### Data Preparation

Ensure your Bach chorale dataset is organized as CSV files:
```
chorales/
├── train/    # Training chorales (CSV format with 4-note columns)
├── valid/    # Validation chorales  
└── test/     # Test chorales
```

Each CSV file contains chorales with columns: `note0`, `note1`, `note2`, `note3` representing the four voices.

### Training the Model

```python
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, BatchNormalization, LSTM, Dropout
from tensorflow.keras.optimizers import Nadam

# Load and preprocess data
def make_xy(chorales):
    windows = [c[i:i + 33] for c in chorales for i in range(0, len(c) - 32, 16)]
    data = np.array(windows, dtype=int)
    data = np.where(data==0, 0, data - 36 + 1)  # Map MIDI to 0-46
    flat = data.reshape(data.shape[0], -1)
    return flat[:, :-1], flat[:, 1:]

# Build model (454K parameters)
model = Sequential([
    Embedding(input_dim=47, output_dim=5),
    Conv1D(32, 2, padding="causal", activation="relu"),
    BatchNormalization(),
    Conv1D(48, 2, padding="causal", activation="relu", dilation_rate=2),
    BatchNormalization(),
    Conv1D(64, 2, padding="causal", activation="relu", dilation_rate=4),
    BatchNormalization(),
    Conv1D(96, 2, padding="causal", activation="relu", dilation_rate=8),
    BatchNormalization(),
    Conv1D(128, 2, padding="causal", activation="relu", dilation_rate=16),
    BatchNormalization(),
    Dropout(0.05),
    LSTM(256, return_sequences=True),
    Dense(47, activation='softmax')
])

# Train model
optimizer = Nadam(learning_rate=1e-3)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=20, validation_data=[X_valid, Y_valid], batch_size=32)
```

### Generating Music

```python
def generate_chorale(model, seed_chords, length):
    """Generate new chorale from seed material"""
    token_sequence = np.array(seed_chords, dtype=int)
    token_sequence = np.where(token_sequence == 0, token_sequence, token_sequence - 36 + 1)
    token_sequence = token_sequence.reshape(1, -1)

    # Generate note by note
    for _ in range(length * 4):
        next_token_probabilities = model.predict(token_sequence, verbose=0)[0, -1]
        next_token = sample_next_note(next_token_probabilities)
        token_sequence = np.concatenate([token_sequence, [[next_token]]], axis=1)
        
    token_sequence = np.where(token_sequence == 0, token_sequence, token_sequence + 36 - 1)
    return token_sequence.reshape(-1, 4)

# Generate new chorale
seed_chords = test_data[2][:8]  # Use 8 initial chords as seed
new_chorale = generate_chorale(model, seed_chords, 56)

# Convert to MIDI playback
from music21 import stream, chord
s = stream.Stream()
for row in new_chorale.tolist():
    s.append(chord.Chord([n for n in row if n], quarterLength=1))
s.show('midi')  # Opens MIDI player
```

## Model Performance

The trained model demonstrates impressive musical generation capabilities:

- **Training Accuracy**: 91.4% (20 epochs)
- **Validation Accuracy**: 81.7%
- **Model Size**: 454,794 parameters (1.73 MB)
- **Training Time**: ~3-4 seconds per epoch on modern hardware

### Performance Metrics by Epoch
- **Epochs 1-5**: Rapid initial learning (accuracy jumps from 33% to 75%)
- **Epochs 6-10**: Convergence phase (validation accuracy peaks at 82%)
- **Epochs 11-20**: Fine-tuning with slight overfitting tendency

## Key Functions

### `make_xy(chorales)`
Preprocesses chorale data into training sequences:
- Creates sliding windows of 33 chords (32 input + 1 target)
- Maps MIDI notes (36-81) to model indices (1-46), with 0 for silence
- Uses 16-chord offset between windows for data augmentation

### `sample_next_note(probs)`
Probabilistic note selection for generation:
```python
def sample_next_note(probs):
    probabilities = np.asarray(probs, dtype=float)
    probs_sum = probabilities.sum()
    
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        return int(np.argmax(probabilities))
    
    probabilities /= probs_sum
    return np.random.choice(len(probabilities), p=probabilities)
```

## Project Structure

```
AI Music Generation/
├── Main.ipynb                          # Complete implementation notebook
├── README.md                           # This documentation
├── chorales/                           # Training data directory
│   ├── train/                         # Training chorales (CSV format)
│   ├── valid/                         # Validation chorales
│   └── test/                          # Test chorales
└── bach_generation_conv1d_lstm.keras  # Saved trained model
```

## Musical Characteristics

The generated chorales exhibit authentic baroque features:
- **Harmonic Progressions**: Traditional tonal progressions with proper voice leading
- **Melodic Contour**: Natural melodic lines in all four voices
- **Rhythmic Consistency**: Steady quarter-note rhythm typical of chorales
- **Cadential Structure**: Proper phrase endings and harmonic resolutions

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Music21
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook (for running Main.ipynb)

## Architecture Design Decisions

### Dilated Convolutions
The increasing dilation rates (2, 4, 8, 16) create an exponential receptive field:
- **Local patterns**: Captured by early layers (adjacent notes)
- **Medium-term patterns**: Captured by middle layers (phrase segments) 
- **Long-term patterns**: Captured by later layers (structural elements)

### Embedding Layer
- Maps discrete note indices to continuous 5D vectors
- Allows the model to learn meaningful note relationships
- Enables arithmetic operations on musical concepts

### LSTM Integration
- Positioned after convolutional layers for efficiency
- Focuses on long-term dependencies after local patterns are extracted
- 256 units provide sufficient capacity for musical sequence modeling

## Limitations and Future Work

### Current Limitations
- Fixed to 4-voice texture
- Limited note range (MIDI 36-81)
- No explicit control over key or musical form
- Requires substantial training data

### Future Enhancements
- **Conditional Generation**: Control over key, tempo, and style parameters
- **Variable Voice Count**: Support for different ensemble sizes
- **Real-time Generation**: Live music creation interface
- **Style Transfer**: Adapt chorales to different compositional styles

## License

This project is available for educational and research purposes.

---

*This implementation showcases advanced deep learning techniques applied to music generation, demonstrating the intersection of artificial intelligence and musical creativity.*
