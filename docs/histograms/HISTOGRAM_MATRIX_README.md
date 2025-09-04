# Histogram Matrix for CNN Input

This document explains the implementation of concatenated histogram matrices for deepfake recognition using Convolutional Neural Networks (CNNs).

## Overview

The core strategy is to represent image windows as concatenated RGB histograms, creating a single feature matrix suitable for CNN input. Instead of separate red, green, and blue channel histograms, we concatenate them into a single long vector of size 256×3=768 per window.

## Architecture

### Input Processing Flow

1. **Face Detection & Alignment**: Detect faces and align them using facial landmarks
2. **Sliding Window**: Apply sliding windows over the aligned face
3. **Histogram Computation**: Compute 256-bin histograms for each RGB channel per window
4. **Concatenation**: Concatenate RGB histograms into single 768-dimensional feature vectors
5. **Matrix Formation**: Create final matrix of shape (num_windows, 768)

### Matrix Structure

Final Matrix Shape: (L/W, 768)
Where:

- L/W = Number of windows
- 768 = Concatenated RGB histograms (256×3)

Each row represents one window:
Row i: [R_0, R_1, ..., R_255, G_0, G_1, ..., G_255, B_0, B_1, ..., B_255]

## Implementation Details

### Key Functions

#### `concatenate_histograms_to_matrix(histograms)`

- **Input**: List of `HistogramData` dictionaries
- **Output**: NumPy array of shape (num_windows, 768)
- **Process**: Concatenates RGB histograms for each window

#### `get_histogram_matrix_for_cnn(image, cfg)`

- **Input**: Image and configuration
- **Output**: Concatenated histogram matrix
- **Process**: Combines histogram computation and concatenation

#### `get_cnn_histogram_features(image, cfg)`

- **Input**: Image and configuration  
- **Output**: `CNNHistogramFeatures` structured object
- **Process**: Provides complete feature representation with metadata

### Data Structures

#### `HistogramData`

```python
{
    "window_x": int,
    "window_y": int, 
    "hist_r": List[float],  # 256 bins
    "hist_g": List[float],  # 256 bins
    "hist_b": List[float]   # 256 bins
}
```

#### `CNNHistogramFeatures`

```python
{
    "feature_matrix": NumpyArray,  # Shape: (num_windows, 768)
    "num_windows": int,
    "feature_dimension": int,      # Always 768
    "window_size": int
}
```

## Usage Examples

### Basic Usage

```python
from src.preprocessing.histograms import get_histogram_matrix_for_cnn

# Get concatenated matrix
matrix = get_histogram_matrix_for_cnn(image, config)
print(f"Matrix shape: {matrix.shape}")  # (num_windows, 768)
```

### Structured Usage

```python
from src.preprocessing.histograms import get_cnn_histogram_features

# Get structured features
cnn_features = get_cnn_histogram_features(image, config)
print(f"Windows: {cnn_features.num_windows}")
print(f"Features per window: {cnn_features.feature_dimension}")
```

### Manual Concatenation

```python
from src.preprocessing.histograms import (
    compute_histograms_for_window,
    concatenate_histograms_to_matrix
)

# Step by step
histograms = compute_histograms_for_window(image, config)
matrix = concatenate_histograms_to_matrix(histograms)
```

## CNN Integration

### Input Layer

The CNN input layer should expect:

- **Input shape**: (batch_size, num_windows, 768)
- **Feature dimension**: 768 (concatenated RGB histograms)

## Testing

Run the test script to verify functionality:

```bash
uv run scripts/testing/test_histogram_matrix.py
```

This will:

- Test histogram computation
- Verify concatenation structure
- Validate matrix shapes
- Confirm data integrity

## Benefits

1. **Unified Representation**: Single matrix format for CNN input
2. **Efficient Processing**: Vectorized operations on concatenated features
3. **Flexible Architecture**: Easy to modify window sizes and feature dimensions
4. **Structured Output**: Clear metadata about the feature representation
5. **Scalable**: Handles variable numbers of windows per image

## Future Enhancements

- **Normalization**: Add histogram normalization options
- **Feature Selection**: Implement feature importance ranking
- **Dimensionality Reduction**: Add PCA or other reduction methods
- **Multi-scale**: Support for multiple window sizes
- **Batch Processing**: Optimize for multiple images
