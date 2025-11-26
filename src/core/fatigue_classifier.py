"""
Fatigue Level Classifier for Multi-Modal Drowsiness Detection
Implements 5-level fatigu classification based on multiple indicators:
- Level 0: Fully Alert (0-20%)
- Level 1: Mild Fatigue (20-40%)
- Level 2: Moderate Fatigue (40-65%)
- Level 3: Severe Drowsiness (65-85%)
- Level 4: Critical Microsleep (85-100%)
"""

import numpy as np

class FatigueClassifier:
    """Multi-level fatigue classification with weighted scoring"""
    
    LEVELS = {
        0: {"name": "Fully Alert", "color": (0, 255, 0), "action": "Continue driving safely"},
        1: {"name": "Mild Fatigue", "color": (100, 255, 100), "action": "Consider a short break soon"},
        2: {"name": "Moderate Fatigue", "color": (0, 165, 255), "action": "⚠️ Recommended: Stop at next safe location"},
        3: {"name": "Severe Drowsiness", "color": (0, 100, 255), "action": "⚠️ Pull over safely - drowsiness detected"},
        4: {"name": "Critical Microsleep", "color": (0, 0, 255), "action": "🚨 DANGER: Stop immediately!"}
    }
    
    def __init__(self, weights=None):
        # Feature weights for scoring
        self.weights = {
            'ear': 40,          # Eye aspect ratio (Primary indicator now)
            'mar': 20,          # Mouth aspect ratio (Contributor)
            'blink_rate': 10,   # Blink frequency
            'closure_duration': 30,  # Sustained eye closure (Critical override)
            'ml_prob': 20       # ML Model Probability
        }
        
        # Allow overriding weights
        if weights:
            self.weights.update(weights)
    
    def classify(self, features):
        """
        Classify fatigue level based on multiple features.
        Uses a 'Max Critical' approach: The final score is driven by the most severe indicator.
        """
        score = 0.0
        total_weight = 0.0
        
        # Track individual scores to enforce "Max Critical" logic
        critical_scores = []
        
        # EAR contribution (lower EAR = higher fatigue)
        # EAR contribution (lower EAR = higher fatigue)
        # Only count low EAR if it persists longer than a blink (e.g., > 5 frames)
        if 'ear' in features:
            ear = features['ear']
            closure_frames = features.get('closure_frames', 0)
            
            # If eyes are closed but it's a short duration (blink), ignore or reduce score
            if closure_frames < 5 and ear < 0.20:
                ear_score = 0
            elif ear < 0.15:
                ear_score = 100
            elif ear < 0.20:
                ear_score = 80
            elif ear < 0.24: # Slightly increased threshold for sensitivity
                ear_score = 60
            else:
                ear_score = max(0, (0.32 - ear) * 100)
            
            score += ear_score * self.weights['ear']
            total_weight += self.weights['ear']
            # EAR is instantaneous, so we don't add to critical_scores to avoid blink triggers.
        
        # MAR contribution (yawning)
        if 'mar' in features and 'yawn_count' in features:
            mar = features['mar']
            yawn_count = features['yawn_count']
            
            # Sustained high MAR or multiple yawns
            mar_score = min(100, (mar - 0.4) * 200) if mar > 0.4 else 0
            yawn_score = min(100, yawn_count * 20)  # 5+ yawns = max score
            
            combined_mar_score = max(mar_score, yawn_score)
            score += combined_mar_score * self.weights['mar']
            total_weight += self.weights['mar']
            # MAR removed from critical_scores as requested (contributor only)
        
        # Blink rate contribution
        if 'blink_rate' in features:
            blink_rate = features['blink_rate']
            # Normal: 10-30 bpm
            if blink_rate < 8:
                blink_score = 70
            elif blink_rate > 40:
                blink_score = 50
            elif 12 <= blink_rate <= 28:
                blink_score = 0
            else:
                blink_score = 20
            
            score += blink_score * self.weights['blink_rate']
            total_weight += self.weights['blink_rate']
        
        # Closure duration contribution
        if 'closure_frames' in features:
            closure = features['closure_frames']
            # Assuming ~30 fps:
            if closure > 150:  # > 5 seconds
                closure_score = 100  # Critical
            elif closure > 90:  # > 3 seconds
                closure_score = 90   # Severe
            elif closure > 45:  # > 1.5 seconds
                closure_score = 60   # Moderate
            elif closure > 20:  # > 0.7 seconds
                closure_score = 30   # Mild
            else:
                closure_score = 0
            
            score += closure_score * self.weights['closure_duration']
            total_weight += self.weights['closure_duration']
            critical_scores.append(closure_score)
        
        # ML Probability contribution (GATED by sustained closure)
        # Only allow ML to contribute if there's already sustained drowsiness
        if 'p_drowsy' in features and 'closure_frames' in features:
            p_d = features['p_drowsy']
            closure = features['closure_frames']
            
            # GATE: Only use ML probability if eyes have been closed for >20 frames
            # This ensures rule-based detection has already flagged sustained drowsiness
            # ML acts as a "confirmation" signal, not a primary trigger
            if closure > 20:
                # Map probability (0-1) to score (0-100)
                if p_d > 0.90:
                    ml_score = 100
                elif p_d > 0.75:
                    ml_score = 70
                elif p_d > 0.50:
                    ml_score = 30
                else:
                    ml_score = 0
                
                score += ml_score * self.weights.get('ml_prob', 0)
                total_weight += self.weights.get('ml_prob', 0)
                
                # High ML probability is a critical indicator ONLY if extremely high
                if p_d > 0.95:
                    critical_scores.append(90)
            # else: ML contribution is 0 when closure_frames <= 20 (prevents blink triggers)
        # Calculate Weighted Average
        weighted_avg = (score / total_weight) if total_weight > 0 else 0
        
        # --- MAX CRITICAL LOGIC ---
        # Only closure_duration is now a critical override.
        # We take the maximum of the weighted average and the highest critical score.
        max_critical = max(critical_scores) if critical_scores else 0
        
        # Final score is the MAX of average and the highest single indicator
        # This solves the issue where "0 yawns" drags down "eyes closed".
        fatigue_score = max(weighted_avg, max_critical)
        
        # Map score to fatigue level - Adjusted for sensitivity
        if fatigue_score < 20:
            level = 0
        elif fatigue_score < 35: # Lowered from 40
            level = 1
        elif fatigue_score < 55: # Lowered from 65/50
            level = 2
        elif fatigue_score < 75: # Lowered from 85/70
            level = 3
        else:
            level = 4
            
        # --- SAFETY OVERRIDES ---
        # Keep these as a fail-safe
        if 'closure_frames' in features and features['closure_frames'] > 150:
            level = 4
            fatigue_score = 100.0
        elif 'closure_frames' in features and features['closure_frames'] > 90:
            level = max(level, 3)
            fatigue_score = max(fatigue_score, 90.0)
        
        # Calculate confidence
        confidence = min(1.0, total_weight / sum(self.weights.values()))
        
        return level, fatigue_score, confidence
    
    def get_level_info(self, level):
        """Get information about a fatigue level"""
        return self.LEVELS.get(level, self.LEVELS[0])
    
    def get_progressive_alert(self, level):
        """Get appropriate alert type for the fatigue level"""
        alerts = {
            0: None,  # No alert
            1: "gentle_chime",
            2: "moderate_beep",
            3: "strong_alert",
            4: "critical_alarm"
        }
        return alerts.get(level)
