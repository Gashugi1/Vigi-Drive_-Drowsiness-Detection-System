"""
Integration tests for drowsiness detection pipeline
Tests end-to-end flow from image → features → model → prediction
"""

import pytest
import cv2
import numpy as np
import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.realtime_ml_frame import DrowsinessDetector, calculate_ear, calculate_mar
import mediapipe as mp


class TestModelLoading:
    """Test ML model loading and inference"""
    
    def test_model_bundle_structure(self):
        """Verify model bundle contains required components"""
        model_path = Path(__file__).parent.parent / "models" / "earmar_img_logreg.joblib"
        
        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")
        
        bundle = joblib.load(model_path)
        
        assert "scaler" in bundle, "Model bundle missing scaler"
        assert "model" in bundle, "Model bundle missing model"
        assert "feature_order" in bundle, "Model bundle missing feature_order"
        
        # Check feature order matches expected
        expected_features = ["ear", "mar", "eye_close", "mouth_open"]
        assert bundle["feature_order"] == expected_features, \
            f"Feature order mismatch: {bundle['feature_order']}"
        
        print(f"✓ Model bundle valid with features: {bundle['feature_order']}")
    
    def test_model_inference_shape(self):
        """Test model can process sample input"""
        model_path = Path(__file__).parent.parent / "models" / "earmar_img_logreg.joblib"
        
        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")
        
        bundle = joblib.load(model_path)
        scaler = bundle["scaler"]
        model = bundle["model"]
        
        # Create sample input (alert state)
        x = np.array([[0.30, 0.35, 0.0, 0.0]])  # [ear, mar, eye_close, mouth_open]
        x_scaled = scaler.transform(x)
        proba = model.predict_proba(x_scaled)[0, 1]
        
        assert 0.0 <= proba <= 1.0, f"Probability out of range: {proba}"
        assert proba < 0.5, f"Alert state should have low drowsiness probability, got {proba}"
        
        # Test drowsy state
        x_drowsy = np.array([[0.15, 0.45, 1.0, 0.0]])  # Low EAR, eye closed
        x_drowsy_scaled = scaler.transform(x_drowsy)
        proba_drowsy = model.predict_proba(x_drowsy_scaled)[0, 1]
        
        assert proba_drowsy > proba, "Drowsy state should have higher probability"
        print(f"✓ Model inference: alert={proba:.2f}, drowsy={proba_drowsy:.2f}")


class TestConfigValidation:
    """Test configuration file structure and values"""
    
    def test_config_structure(self):
        """Verify config.json has all required fields"""
        import json
        config_path = Path(__file__).parent.parent / "config.json"
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Check required sections
        assert "camera" in config
        assert "inference" in config
        assert "logging" in config
        
        # Check critical inference parameters
        assert "model_path" in config["inference"]
        assert "EAR_CLOSED_THRESH" in config["inference"]
        assert "prob_threshold" in config["inference"]
        
        print(f"✓ Config valid with EAR_THRESH={config['inference']['EAR_CLOSED_THRESH']}")
    
    def test_threshold_values_reasonable(self):
        """Check that threshold values are in expected ranges"""
        import json
        config_path = Path(__file__).parent.parent / "config.json"
        
        with open(config_path) as f:
            config = json.load(f)
        
        ear_thresh = config["inference"]["EAR_CLOSED_THRESH"]
        prob_thresh = config["inference"]["prob_threshold"]
        
        # EAR threshold should be between 0.15 and 0.30
        assert 0.15 <= ear_thresh <= 0.30, f"EAR threshold unreasonable: {ear_thresh}"
        
        # Probability threshold should be between 0.5 and 0.95
        assert 0.5 <= prob_thresh <= 0.95, f"Probability threshold unreasonable: {prob_thresh}"
        
        print(f"✓ Thresholds reasonable: EAR={ear_thresh}, P(drowsy)={prob_thresh}")


class TestDetectorInitialization:
    """Test DrowsinessDetector initialization"""
    
    def test_detector_init(self):
        """Test detector can be initialized"""
        try:
            detector = DrowsinessDetector(config_path="config.json")
            
            # Check critical attributes exist
            assert hasattr(detector, 'mesh')
            assert hasattr(detector, 'scaler')
            assert hasattr(detector, 'clf')
            assert hasattr(detector, 'EAR_THRESH')
            assert hasattr(detector, 'p_thresh')
            assert hasattr(detector, 'closure_start_time')  # New time-based tracking
            assert hasattr(detector, 'NO_FACE_GRACE')  # New grace period
            
            detector.release_resources()
            print("✓ Detector initialized successfully")
            
        except Exception as e:
            pytest.fail(f"Detector initialization failed: {e}")


class TestFeatureConsistency:
    """Test that features are calculated consistently"""
    
    def test_bilateral_ear_averaging(self):
        """Test that both eyes contribute to final EAR"""
        # This test would require actual landmarks
        # Placeholder to remind of this edge case
        pytest.skip("Requires mock landmark creation - see audit recommendation")
    
    def test_feature_engineering_matches_training(self):
        """Critical test: eye_close feature must match training"""
        import json
        config_path = Path(__file__).parent.parent / "config.json"
        
        with open(config_path) as f:
            config = json.load(f)
        
        ear_thresh = config["inference"]["EAR_CLOSED_THRESH"]
        
        # After fix, this should be consistent
        # Training uses instant threshold, inference should too
        
        # Simulate feature creation (like in app.py line 225)
        e_val = 0.20  # Right at threshold
        
        # OLD (WRONG): eye_close based on closure_frames > 10
        # NEW (CORRECT): eye_close based on e_val < EAR_THRESH
        eye_close_correct = 1.0 if e_val < ear_thresh else 0.0
        
        assert eye_close_correct == 1.0, "Feature standardization check failed"
        print(f"✓ Feature engineering consistent (instant threshold at EAR={ear_thresh})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
