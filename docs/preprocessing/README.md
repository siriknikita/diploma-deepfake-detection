# Data Preprocessing Pipeline for Deepfake Recognition

## Overview

This document describes the comprehensive data preprocessing pipeline implemented for the deepfake recognition research project. The pipeline transforms raw facial images into structured feature representations suitable for Convolutional Neural Network (CNN) analysis, enabling robust detection of manipulated facial content.

## Architecture Overview

The preprocessing pipeline consists of four main stages:

1. **Face Detection** - Locating and extracting facial regions using MTCNN
2. **Face Alignment** - Geometric correction based on facial landmarks
3. **Image Normalization** - Squaring, padding, and resizing operations
4. **Feature Extraction** - Histogram computation and matrix construction

```
Raw Image → Face Detection → Face Alignment → Normalization → Histogram Features → CNN Input
```

## 1. Face Detection Module

### Overview

The face detection module utilizes the Multi-task Cascaded Convolutional Networks (MTCNN) architecture to identify facial regions within input images. This stage provides bounding box coordinates and facial landmark points essential for subsequent processing steps.

### Implementation Details

**File**: `src/preprocessing/detection.py`

**Key Functions**:

- `detect_face_features()` - Main detection interface with multiple output formats

**Features**:

- **Multi-face Detection**: Supports detection of multiple faces within a single image
- **Landmark Extraction**: Optional extraction of 68 facial landmark points
- **Flexible Output**: Returns data in dictionary, JSON, or structured object formats
- **Error Handling**: Robust handling of detection failures and edge cases

**Dependencies**:

- `facenet-pytorch` - MTCNN implementation
- `PIL` - Image loading and processing

**Output Structure**:

```python
DetectedFeatures:
    boxes: np.ndarray | None      # Bounding box coordinates [x1, y1, x2, y2]
    landmarks: np.ndarray | None  # Facial landmark coordinates
```

### Configuration Parameters

- **Detection Confidence**: MTCNN internal confidence thresholds
- **Minimum Face Size**: Minimum detectable face dimensions
- **Device Optimization**: CPU/GPU acceleration support

## 2. Face Alignment Module

### Overview

Face alignment corrects geometric distortions by rotating the image based on eye landmark positions. This ensures consistent facial orientation across different input images, improving feature extraction reliability.

### Implementation Details

**File**: `src/preprocessing/normalization.py`

**Key Functions**:

- `align_face()` - Performs geometric alignment based on eye landmarks

**Algorithm**:

1. **Center Calculation**: Compute face center from bounding box
2. **Eye Position Mapping**: Convert relative landmarks to absolute coordinates
3. **Rotation Angle**: Calculate rotation angle from eye vector
4. **Affine Transformation**: Apply rotation matrix to entire image

**Mathematical Foundation**:

```
Rotation Angle = arctan2(Δy, Δx)
where Δx = right_eye_x - left_eye_x
      Δy = right_eye_y - left_eye_y
```

**Benefits**:

- Eliminates head pose variations
- Ensures consistent feature extraction
- Improves CNN training stability

## 3. Image Normalization Module

### Overview

The normalization module transforms aligned faces into standardized square formats with consistent dimensions, ensuring uniform input requirements for the CNN architecture.

### Implementation Details

**File**: `src/preprocessing/normalization.py`

**Key Functions**:

- `square_and_resize_face()` - Squares, pads, and resizes facial regions
- `normalize_face()` - Complete normalization pipeline

**Processing Steps**:

#### 3.1 Face Squaring

- **Dynamic Sizing**: Calculate optimal square dimensions based on face bounding box
- **Padding Application**: Add configurable padding around facial region
- **Size Alignment**: Round dimensions to multiples of window size for histogram processing

#### 3.2 Padding Strategy

```python
# Padding calculation
size = max(width, height) * (1 + padding_factor)
size = round_up_to_multiple(size, window_size)

# Border padding with zero values
padded = cv2.copyMakeBorder(
    image, pad_top, pad_bottom, pad_left, pad_right,
    borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
)
```

#### 3.3 Resizing Operations

- **Interpolation Method**: Uses `cv2.INTER_AREA` for downsampling
- **Aspect Ratio Preservation**: Maintains facial proportions
- **Memory Optimization**: Efficient handling of large images

### Configuration Parameters

- **Window Size**: Base dimension for histogram computation (default: 9)
- **Padding Factor**: Relative padding around face region (default: 0.25)
- **Output Dimensions**: Final image size for CNN processing

## 4. Histogram Feature Extraction

### Overview

The histogram module computes RGB channel histograms using a sliding window approach, creating feature matrices suitable for CNN input. This approach captures local color distribution patterns that are indicative of image manipulation artifacts.

### Implementation Details

**File**: `src/preprocessing/histograms.py`

**Key Functions**:

- `compute_histograms_for_window()` - Sliding window histogram computation
- `concatenate_histograms_to_matrix()` - Feature matrix construction
- `get_cnn_histogram_features()` - Complete feature extraction pipeline

### 4.1 Sliding Window Algorithm

**Window Configuration**:

- **Window Size**: Configurable square window dimensions
- **Stride**: Non-overlapping window movement
- **Coverage**: Complete image coverage without gaps

**Processing Flow**:

```python
for y in range(0, height - window_size + 1, window_size):
    for x in range(0, width - window_size + 1, window_size):
        # Extract RGB channels for current window
        window_r = r_channel[y:y+window_size, x:x+window_size]
        window_g = g_channel[y:y+window_size, x:x+window_size]
        window_b = b_channel[y:y+window_size, x:x+window_size]
        
        # Compute histograms using NumPy bincount
        hist_r = np.bincount(window_r.flatten(), minlength=256)
        hist_g = np.bincount(window_g.flatten(), minlength=256)
        hist_b = np.bincount(window_b.flatten(), minlength=256)
```

### 4.2 Histogram Computation

**Optimization Techniques**:

- **Channel Separation**: Pre-split RGB channels for efficient processing
- **NumPy Bincount**: Fast histogram calculation using optimized array operations
- **Memory Efficiency**: Minimal memory allocation during processing

**Histogram Properties**:

- **Bin Count**: 256 bins per channel (0-255 intensity values)
- **Normalization**: Raw frequency counts (no normalization applied)
- **Channel Independence**: Separate histograms for R, G, B channels

### 4.3 Feature Matrix Construction

**Concatenation Strategy**:

- **Feature Vector**: Each window produces 768-dimensional feature vector
  - Red channel: 256 bins
  - Green channel: 256 bins
  - Blue channel: 256 bins
- **Matrix Format**: (num_windows × 768) feature matrix
- **Data Type**: 64-bit floating point precision

**Output Structure**:

```python
CNNHistogramFeatures:
    feature_matrix: np.ndarray  # Shape: (num_windows, 768)
    num_windows: int           # Number of processed windows
    feature_dimension: int     # Feature vector length (768)
    window_size: int          # Window size used for computation
```

## 5. Data Flow and Integration

### Pipeline Execution

The complete preprocessing pipeline is orchestrated through the main execution flow:

```python
def main():
    # 1. Load configuration
    cfg = load_config(ConfigName.DEFAULT)
    
    # 2. Detect faces and landmarks
    detected_features = detect_face_features(
        image_path=input_file_path,
        with_landmarks=True
    )
    
    # 3. Process each detected face
    for box, landmarks in zip(boxes, landmarks):
        # 4. Normalize face (align + square + resize)
        normalized_face = normalize_face(
            image=original_img,
            face_box=box,
            landmarks=landmarks,
            cfg=cfg
        )
        
        # 5. Extract CNN features
        cnn_features = get_cnn_histogram_features(
            image=normalized_face,
            cfg=cfg
        )
```

### Configuration Management

**File**: `configs/default.yaml`

**Parameters**:

```yaml
window_size: 9          # Histogram window size
padding: 0.25          # Face padding factor
input_file_path: "./data/test-dataset/girl-in-sunlight.jpg"
```

## 6. Performance Characteristics

### Computational Complexity

- **Face Detection**: O(n²) where n is image dimension
- **Histogram Computation**: O(w² × h²) where w, h are image dimensions
- **Memory Usage**: Linear with image size and window count

### Optimization Features

- **Vectorized Operations**: NumPy-based histogram computation
- **Efficient Memory Management**: Minimal intermediate storage
- **Parallel Processing**: Channel-wise independent operations

## 7. Quality Assurance

### Error Handling

- **Detection Failures**: Graceful handling of no-face scenarios
- **Landmark Validation**: Verification of landmark quality
- **Memory Management**: Efficient handling of large images

### Validation Checks

- **Bounding Box Validation**: Ensures valid face regions
- **Landmark Consistency**: Verifies landmark point validity
- **Output Verification**: Confirms feature matrix integrity

## 8. Research Applications

### Deepfake Detection

The preprocessing pipeline is specifically designed for deepfake detection research:

- **Artifact Preservation**: Maintains manipulation artifacts during processing
- **Feature Consistency**: Ensures uniform input for CNN training
- **Scalability**: Supports large-scale dataset processing

## 10. Dependencies and Requirements

### Core Dependencies

```toml
facenet-pytorch >= 2.5.3    # MTCNN face detection
opencv-python >= 4.12.0.88 # Computer vision operations
pydantic >= 2.11.7         # Data validation and serialization
numpy                       # Numerical computations
PIL                         # Image processing
```

### Development Tools

```toml
ruff                        # Code linting
black                       # Code formatting
mypy                        # Static type checking
pre-commit                  # Git hooks
```

## 11. Usage Examples

### Basic Usage

```python
from src.preprocessing.detection import detect_face_features
from src.preprocessing.normalization import normalize_face
from src.preprocessing.histograms import get_cnn_histogram_features

# Detect faces
features = detect_face_features("image.jpg", with_landmarks=True)

# Process each face
for box, landmarks in zip(features.boxes, features.landmarks):
    # Normalize face
    normalized = normalize_face(image, box, landmarks, config)
    
    # Extract features
    cnn_features = get_cnn_histogram_features(normalized, config)
```

### Configuration Customization

```python
from src.config import load_config
from src.schemas.enums.config_paths import ConfigName

# Load custom configuration
config = load_config(ConfigName.EXPERIMENT)

# Modify parameters
config.window_size = 16
config.padding = 0.5
```

## 12. Conclusion

The data preprocessing pipeline provides a robust, research-grade foundation for deepfake detection research. Through careful implementation of face detection, alignment, normalization, and feature extraction, it delivers consistent, high-quality input representations for CNN-based analysis.

The modular architecture enables easy extension and modification, while the comprehensive error handling and validation ensure reliable operation across diverse datasets. This implementation represents a significant contribution to the field of digital forensics and deepfake detection research.

---

*This documentation is part of the deepfake recognition research project and serves as both academic documentation and implementation guide for researchers and developers in the field of digital media forensics.*
