import unittest
import numpy as np
from src.core.realtime_ml_frame import calculate_ear, calculate_mar
from src.core.fatigue_classifier import FatigueClassifier

class TestDrowsinessLogic(unittest.TestCase):
    def setUp(self):
        # Mock landmarks (normalized coordinates)
        # Create a simple structure to mimic MediaPipe landmarks
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        self.Point = Point
        self.W, self.H = 100, 100 # Simple 100x100 grid

    def test_ear_calculation_open_eye(self):
        # Simulate open eye
        # Horizontal: (10, 50) to (30, 50) -> dist = 20
        # Vertical 1: (20, 45) to (20, 55) -> dist = 10
        # Vertical 2: (20, 45) to (20, 55) -> dist = 10
        # EAR = (10 + 10) / (2 * 20) = 20 / 40 = 0.5
        
        landmarks = [self.Point(0,0)] * 400 # Dummy list
        indices = [0, 1, 2, 3, 4, 5]
        
        # Map indices to points
        # 0: P1 (Left), 3: P4 (Right)
        # 1: P2 (Top1), 5: P6 (Bottom1)
        # 2: P3 (Top2), 4: P5 (Bottom2)
        
        landmarks[0] = self.Point(0.1, 0.5) # P1
        landmarks[3] = self.Point(0.3, 0.5) # P4
        
        landmarks[1] = self.Point(0.2, 0.45) # P2
        landmarks[5] = self.Point(0.2, 0.55) # P6
        
        landmarks[2] = self.Point(0.2, 0.45) # P3
        landmarks[4] = self.Point(0.2, 0.55) # P5
        
        ear = calculate_ear(landmarks, indices, self.W, self.H)
        self.assertAlmostEqual(ear, 0.5, places=2)

    def test_ear_calculation_closed_eye(self):
        # Simulate closed eye (vertical distance near 0)
        landmarks = [self.Point(0,0)] * 400
        indices = [0, 1, 2, 3, 4, 5]
        
        landmarks[0] = self.Point(0.1, 0.5)
        landmarks[3] = self.Point(0.3, 0.5)
        
        # Verticals are same point (closed)
        landmarks[1] = self.Point(0.2, 0.5)
        landmarks[5] = self.Point(0.2, 0.5)
        
        landmarks[2] = self.Point(0.2, 0.5)
        landmarks[4] = self.Point(0.2, 0.5)
        
        ear = calculate_ear(landmarks, indices, self.W, self.H)
        self.assertAlmostEqual(ear, 0.0, places=2)

    def test_fatigue_classifier_alert(self):
        classifier = FatigueClassifier()
        features = {
            'ear': 0.35, # Wide open
            'mar': 0.0,
            'blink_rate': 15,
            'closure_frames': 0,
            'p_drowsy': 0.0
        }
        level, score, conf = classifier.classify(features)
        self.assertEqual(level, 0) # Should be Alert

    def test_fatigue_classifier_drowsy(self):
        classifier = FatigueClassifier()
        features = {
            'ear': 0.18, # Low EAR
            'mar': 0.0,
            'blink_rate': 15,
            'closure_frames': 50, # Sustained closure > 1.5s
            'p_drowsy': 0.8
        }
        level, score, conf = classifier.classify(features)
        # Should be at least level 2 or 3
        self.assertGreaterEqual(level, 2)

if __name__ == '__main__':
    unittest.main()
