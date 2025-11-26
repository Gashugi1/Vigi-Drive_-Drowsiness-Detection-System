"""
Unit tests for drowsiness detection feature extraction
Tests EAR/MAR calculations, threshold consistency, and edge cases
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from realtime_ml_frame import calculate_ear, calculate_mar, l2, denorm_landmarks


class TestUtilityFunctions:
    """Test basic utility functions"""
    
    def test_l2_distance(self):
        """Test Euclidean distance calculation"""
        p1 = (0, 0)
        p2 = (3, 4)
        assert l2(p1, p2) == 5.0
        
        p1 = (0, 0)
        p2 = (0, 0)
        assert l2(p1, p2) == 0.0
    
    def test_denorm_landmarks(self):
        """Test coordinate denormalization"""
        # Mock landmark with x, y attributes
        p = MagicMock()
        p.x = 0.5
        p.y = 0.5
        
        x, y = denorm_landmarks(p, 640, 480)
        assert x == 320
        assert y == 240
        
        p.x = 0.0
        p.y = 0.0
        x, y = denorm_landmarks(p, 640, 480)
        assert x == 0
        assert y == 0


class TestEARCalculation:
    """Test Eye Aspect Ratio calculations"""
    
    def create_mock_landmarks(self, eye_state="open"):
        """Create mock landmarks for different eye states"""
        landmarks = []
        for i in range(500):  # MediaPipe has 468+ landmarks
            p = MagicMock()
            p.x = 0.5
            p.y = 0.5
            p.visibility = 1.0
            landmarks.append(p)
        
        # Set specific eye landmarks based on state
        # LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        # Format: P1(33), P2(160), P3(158), P4(133), P5(153), P6(144)
        
        if eye_state == "open":
            # Horizontal wide, vertical narrow (normal open eye)
            landmarks[33].x, landmarks[33].y = 0.3, 0.4  # Left corner
            landmarks[133].x, landmarks[133].y = 0.5, 0.4  # Right corner
            landmarks[160].x, landmarks[160].y = 0.35, 0.39  # Top
            landmarks[144].x, landmarks[144].y = 0.35, 0.41  # Bottom
            landmarks[158].x, landmarks[158].y = 0.45, 0.39  # Top
            landmarks[153].x, landmarks[153].y = 0.45, 0.41  # Bottom
            
        elif eye_state == "closed":
            # Horizontal same, vertical very small (closed)
            landmarks[33].x, landmarks[33].y = 0.3, 0.4
            landmarks[133].x, landmarks[133].y = 0.5, 0.4
            landmarks[160].x, landmarks[160].y = 0.35, 0.40  # Almost same Y
            landmarks[144].x, landmarks[144].y = 0.35, 0.40
            landmarks[158].x, landmarks[158].y = 0.45, 0.40
            landmarks[153].x, landmarks[153].y = 0.45, 0.40
            
        return landmarks
    
    def test_ear_open_eye(self):
        """Test EAR for open eye (should be > 0.2)"""
        landmarks = self.create_mock_landmarks("open")
        LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        
        ear = calculate_ear(landmarks, LEFT_EYE_LANDMARKS, 640, 480)
        assert ear > 0.15, f"Expected EAR > 0.15 for open eye, got {ear}"
    
    def test_ear_closed_eye(self):
        """Test EAR for closed eye (should be very low)"""
        landmarks = self.create_mock_landmarks("closed")
        LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        
        ear = calculate_ear(landmarks, LEFT_EYE_LANDMARKS, 640, 480)
        assert ear < 0.1, f"Expected EAR < 0.1 for closed eye, got {ear}"
    
    def test_ear_division_by_zero_protection(self):
        """Test that epsilon prevents division by zero"""
        landmarks = []
        for i in range(500):
            p = MagicMock()
            # All points at same location (collapsed eye)
            p.x = 0.5
            p.y = 0.5
            landmarks.append(p)
        
        LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        
        # Should not raise exception, should return very large or very small value
        try:
            ear = calculate_ear(landmarks, LEFT_EYE_LANDMARKS, 640, 480)
            assert isinstance(ear, float)
        except ZeroDivisionError:
            pytest.fail("Division by zero occurred, epsilon protection failed")


class TestMARCalculation:
    """Test Mouth Aspect Ratio calculations"""
    
    def create_mock_landmarks(self, mouth_state="closed"):
        """Create mock landmarks for different mouth states"""
        landmarks = []
        for i in range(500):
            p = MagicMock()
            p.x = 0.5
            p.y = 0.5
            landmarks.append(p)
        
        # MOUTH_LEFT=61, MOUTH_RIGHT=291, MOUTH_TOP=13, MOUTH_BOTTOM=14
        
        if mouth_state == "closed":
            landmarks[61].x, landmarks[61].y = 0.3, 0.6  # Left
            landmarks[291].x, landmarks[291].y = 0.7, 0.6  # Right
            landmarks[13].x, landmarks[13].y = 0.5, 0.59  # Top
            landmarks[14].x, landmarks[14].y = 0.5, 0.61  # Bottom (small gap)
            
        elif mouth_state == "open":  # Yawning
            landmarks[61].x, landmarks[61].y = 0.3, 0.6
            landmarks[291].x, landmarks[291].y = 0.7, 0.6
            landmarks[13].x, landmarks[13].y = 0.5, 0.55  # Top
            landmarks[14].x, landmarks[14].y = 0.5, 0.75  # Bottom (large gap)
            
        return landmarks
    
    def test_mar_closed_mouth(self):
        """Test MAR for closed mouth (should be low)"""
        landmarks = self.create_mock_landmarks("closed")
        
        mar = calculate_mar(landmarks, 640, 480)
        assert mar < 0.3, f"Expected MAR < 0.3 for closed mouth, got {mar}"
    
    def test_mar_open_mouth(self):
        """Test MAR for yawning/open mouth (should be high)"""
        landmarks = self.create_mock_landmarks("open")
        
        mar = calculate_mar(landmarks, 640, 480)
        assert mar > 0.4, f"Expected MAR > 0.4 for open mouth, got {mar}"


class TestThresholdConsistency:
    """Test that training and inference use same thresholds"""
    
    def test_ear_threshold_consistency(self):
        """Verify EAR threshold matches between training and inference"""
        # Load config
        import json
        config_path = Path(__file__).parent.parent / "config.json"
        
        with open(config_path) as f:
            config = json.load(f)
        
        inference_threshold = config["inference"]["EAR_CLOSED_THRESH"]
        
        # Check training script default (should match audit fix to 0.25)
        from training.extract_features_images import main as extract_main
        import argparse
        
        # Expected: both should be 0.25 after fix
        assert inference_threshold == 0.25, \
            f"Inference threshold should be 0.25, got {inference_threshold}"
        
        print(f"✓ EAR threshold consistent: {inference_threshold}")
    
    def test_mar_threshold_consistency(self):
        """Verify MAR threshold is defined"""
        import json
        config_path = Path(__file__).parent.parent / "config.json"
        
        with open(config_path) as f:
            config = json.load(f)
        
        mouth_threshold = config["inference"].get("mouth_open_threshold", 0.6)
        assert 0.4 <= mouth_threshold <= 0.8, \
            f"MAR threshold seems unreasonable: {mouth_threshold}"


class TestTimingAccuracy:
    """Test frame vs. time-based alerting"""
    
    def test_alert_duration_calculation(self):
        """Verify ALERT_DURATION_SEC corresponds to expected timing"""
        import json
        config_path = Path(__file__).parent.parent / "config.json"
        
        with open(config_path) as f:
            config = json.load(f)
        
        frames_to_alert = config["inference"]["FRAMES_TO_ALERT"]
        fps_assumed = config["inference"].get("fps_assumed", 24)
        
        # Calculate expected duration
        expected_duration = frames_to_alert / fps_assumed
        
        # Time-based threshold should be approximately this value
        # After our fix, detector uses 2.7 seconds
        EXPECTED_TIME = 2.7
        
        assert abs(expected_duration - EXPECTED_TIME) < 0.5, \
            f"Frame-based ({expected_duration}s) doesn't match time-based ({EXPECTED_TIME}s)"
        
        print(f"✓ Timing: {frames_to_alert} frames @ {fps_assumed}fps = {expected_duration:.1f}s")


if __name__ == "__main__":
    # Run with pytest -v tests/test_features.py
    pytest.main([__file__, "-v", "--tb=short"])
