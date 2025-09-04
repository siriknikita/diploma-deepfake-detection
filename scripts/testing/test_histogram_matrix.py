#!/usr/bin/env python3
"""
Test script to demonstrate the concatenated histogram matrix functionality.
This script shows how the RGB histograms are combined into a single matrix
suitable for CNN input.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path so we can import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image

from src.config import load_config
from src.preprocessing.histograms import (
    compute_histograms_for_window,
    concatenate_histograms_to_matrix,
    get_histogram_matrix_for_cnn,
    get_cnn_histogram_features
)
from src.schemas.enums.config_paths import ConfigName


def test_histogram_concatenation():
    """Test the histogram concatenation functionality."""
    print("Testing Histogram Matrix Concatenation")
    print("=" * 50)
    
    # Load configuration
    cfg = load_config(ConfigName.DEFAULT)
    print(f"Window size: {cfg.window_size}")
    print(f"Input file: {cfg.input_file_path}")
    
    # Load test image
    try:
        pil_image = Image.open(cfg.input_file_path).convert('RGB')
        print(f"PIL Image loaded: {pil_image.size}")
        
        # Convert PIL Image to NumPy array (OpenCV format)
        image = np.array(pil_image)
        print(f"Converted to NumPy array: {image.shape}")
        
    except FileNotFoundError:
        print(f"Test image not found: {cfg.input_file_path}")
        print("Please ensure the test image exists in the data directory.")
        return
    
    # Test 1: Individual histograms
    print("\n1. Computing individual histograms...")
    histograms = compute_histograms_for_window(image, cfg)
    print(f"   Number of windows: {len(histograms)}")
    
    if histograms:
        first_hist = histograms[0]
        print(f"   First window position: ({first_hist['window_x']}, {first_hist['window_y']})")
        print(f"   Red channel histogram length: {len(first_hist['hist_r'])}")
        print(f"   Green channel histogram length: {len(first_hist['hist_g'])}")
        print(f"   Blue channel histogram length: {len(first_hist['hist_b'])}")
        
        # Verify histogram properties
        assert len(first_hist['hist_r']) == 256, "Red histogram should have 256 bins"
        assert len(first_hist['hist_g']) == 256, "Green histogram should have 256 bins"
        assert len(first_hist['hist_b']) == 256, "Blue histogram should have 256 bins"
        print("   ✓ Histogram bin counts verified")
    
    # Test 2: Concatenated matrix
    print("\n2. Creating concatenated matrix...")
    matrix = concatenate_histograms_to_matrix(histograms)
    print(f"   Matrix shape: {matrix.shape}")
    print(f"   Expected shape: ({len(histograms)}, 768)")
    
    if matrix.size > 0:
        expected_shape = (len(histograms), 768)
        assert matrix.shape == expected_shape, f"Matrix shape mismatch: {matrix.shape} != {expected_shape}"
        print("   ✓ Matrix shape verified")
        
        # Verify concatenation structure
        first_row = matrix[0, :]
        red_part = first_row[:256]
        green_part = first_row[256:512]
        blue_part = first_row[512:768]
        
        print(f"   First row red part (0-255): sum={np.sum(red_part):.2f}, mean={np.mean(red_part):.2f}")
        print(f"   First row green part (256-511): sum={np.sum(green_part):.2f}, mean={np.mean(green_part):.2f}")
        print(f"   First row blue part (512-767): sum={np.sum(blue_part):.2f}, mean={np.mean(blue_part):.2f}")
        
        # Verify the concatenation matches original histograms
        np.testing.assert_array_almost_equal(red_part, first_hist['hist_r'], decimal=10)
        np.testing.assert_array_almost_equal(green_part, first_hist['hist_g'], decimal=10)
        np.testing.assert_array_almost_equal(blue_part, first_hist['hist_b'], decimal=10)
        print("   ✓ Concatenation structure verified")
    
    # Test 3: Direct CNN matrix function
    print("\n3. Testing direct CNN matrix function...")
    cnn_matrix = get_histogram_matrix_for_cnn(image, cfg)
    print(f"   CNN matrix shape: {cnn_matrix.shape}")
    
    if matrix.size > 0 and cnn_matrix.size > 0:
        np.testing.assert_array_almost_equal(matrix, cnn_matrix, decimal=10)
        print("   ✓ Direct CNN function matches concatenation")
    
    # Test 4: Structured CNN features
    print("\n4. Testing structured CNN features...")
    cnn_features = get_cnn_histogram_features(image, cfg)
    print(f"   Structured features:")
    print(f"     - Matrix shape: {cnn_features.feature_matrix.shape}")
    print(f"     - Number of windows: {cnn_features.num_windows}")
    print(f"     - Feature dimension: {cnn_features.feature_dimension}")
    print(f"     - Window size: {cnn_features.window_size}")
    
    if matrix.size > 0:
        np.testing.assert_array_almost_equal(matrix, cnn_features.feature_matrix, decimal=10)
        assert cnn_features.num_windows == len(histograms)
        assert cnn_features.feature_dimension == 768
        assert cnn_features.window_size == cfg.window_size
        print("   ✓ Structured features verified")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("\nThe histogram matrix is now ready for CNN input:")
    print(f"- Input shape: {matrix.shape}")
    print(f"- Each row represents one window with 768 features (256×3 RGB channels)")
    print(f"- Ready to feed into a CNN with input dimension 768")


if __name__ == "__main__":
    test_histogram_concatenation() 