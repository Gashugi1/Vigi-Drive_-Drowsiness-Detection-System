import cv2
import time
import csv
import os
import json
import numpy as np
import joblib
import mediapipe as mp
# Use conditional import for serial
try:
    import serial
except ImportError:
    serial = None

# --- CONSTANTS (Defined once for clarity) ---
# Landmarks for Eye Aspect Ratio (EAR)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [263, 387, 385, 362, 380, 373]
# Landmarks for Mouth Aspect Ratio (MAR)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
# ---------------------------------------------

# --- UTILITY FUNCTIONS (Kept separate for reusability) ---

def l2(p1, p2):
    """Calculates the Euclidean distance between two points (p1 and p2)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def denorm_landmarks(p_norm, W, H):
    """Converts normalized (0.0 to 1.0) coordinates to pixel coordinates."""
    return int(p_norm.x * W), int(p_norm.y * H)

def calculate_ear(landmarks, indices, W, H):
    """Calculates the Eye Aspect Ratio (EAR)."""
    # Denormalize landmarks for the eye indices
    points = [denorm_landmarks(landmarks[i], W, H) for i in indices]
    
    # Calculate vertical and horizontal distances
    P2, P6, P3, P5 = points[1], points[5], points[2], points[4] # Vertical
    P1, P4 = points[0], points[3] # Horizontal
    
    vertical_dist = l2(P2, P6) + l2(P3, P5)
    horizontal_dist = l2(P1, P4)
    
    # Add a tiny epsilon to prevent division by zero
    return vertical_dist / (2.0 * horizontal_dist + 1e-6)

def calculate_mar(landmarks, W, H):
    """Calculates the Mouth Aspect Ratio (MAR)."""
    # Denormalize mouth landmarks
    L = denorm_landmarks(landmarks[MOUTH_LEFT], W, H)
    R = denorm_landmarks(landmarks[MOUTH_RIGHT], W, H)
    T = denorm_landmarks(landmarks[MOUTH_TOP], W, H)
    B = denorm_landmarks(landmarks[MOUTH_BOTTOM], W, H)
    
    # Vertical (Top-Bottom) / Horizontal (Left-Right)
    return l2(T, B) / (l2(L, R) + 1e-6)





class YawnDetector:
    """Detects and counts yawns based on MAR threshold and duration"""
    def __init__(self, mar_threshold=0.6, min_duration_frames=15, cooldown_frames=30):
        self.mar_threshold = mar_threshold
        self.min_duration_frames = min_duration_frames
        self.cooldown_frames = cooldown_frames
        self.yawn_frames = 0
        self.yawn_count = 0
        self.cooldown_counter = 0
        self.yawn_timestamps = []
    
    def update(self, mar_value, current_time):
        """Update yawn detection with current MAR value"""
        # Cooldown period after detecting a yawn
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        # Check if mouth is open enough
        if mar_value > self.mar_threshold:
            self.yawn_frames += 1
        else:
            # Check if we had a sustained yawn
            if self.yawn_frames >= self.min_duration_frames:
                self.yawn_count += 1
                self.yawn_timestamps.append(current_time)
                self.cooldown_counter = self.cooldown_frames
                self.yawn_frames = 0
                return True  # Yawn detected
            self.yawn_frames = 0
        
        return False
    
    def get_yawns_per_minute(self, current_time, window_seconds=60):
        """Get number of yawns in the last window_seconds"""
        cutoff_time = current_time - window_seconds
        self.yawn_timestamps = [t for t in self.yawn_timestamps if t > cutoff_time]
        return len(self.yawn_timestamps)


class BlinkDetector:
    """Detects and analyzes blink rate"""
    def __init__(self, ear_threshold=0.21, min_blink_frames=2, max_blink_frames=10):
        self.ear_threshold = ear_threshold
        self.min_blink_frames = min_blink_frames
        self.max_blink_frames = max_blink_frames
        self.blink_frames = 0
        self.was_blinking = False
        self.blink_count = 0
        self.blink_timestamps = []
    
    def update(self, ear_value, current_time):
        """Update blink detection with current EAR value"""
        is_blinking = ear_value < self.ear_threshold
        
        if is_blinking:
            self.blink_frames += 1
        else:
            # Check if we had a valid blink
            if self.was_blinking and self.min_blink_frames <= self.blink_frames <= self.max_blink_frames:
                self.blink_count += 1
                self.blink_timestamps.append(current_time)
            self.blink_frames = 0
        
        self.was_blinking = is_blinking
        return self.was_blinking and self.blink_frames == self.min_blink_frames
    
    def get_blink_rate(self, current_time, window_seconds=60):
        """Get blinks per minute in the last window"""
        cutoff_time = current_time - window_seconds
        self.blink_timestamps = [t for t in self.blink_timestamps if t > cutoff_time]
        # Convert to blinks per minute
        return len(self.blink_timestamps) * (60.0 / window_seconds)

# ---------------------------------------------

# --- MAIN CLASS ---

class DrowsinessDetector:
    def __init__(self, config_path="../config/config.json"):
        # 1. Load Configuration
        self.CFG = self._load_config(config_path)
        
        # 2. Setup Logging
        os.makedirs("../data/logs", exist_ok=True)
        self.log_path = self.CFG["logging"]["events_csv"]
        self._setup_csv_log()
        
        # 3. Load ML Model
        self.scaler, self.clf, self.feat_order = self._load_model(self.CFG["inference"]["model_path"])
        
        # --- MODIFICATION: Load new alert parameters ---
        # Get the probability threshold (for display)
        self.p_thresh = float(self.CFG["inference"].get("prob_threshold", 0.65))
        # Get the EAR threshold for physical closure
        self.EAR_THRESH = float(self.CFG["inference"].get("EAR_CLOSED_THRESH", 0.22))
        # Get the number of frames for a microsleep
        self.FRAMES_TO_ALERT = int(self.CFG["inference"].get("FRAMES_TO_ALERT", 30))
        # Buffer to prevent flickering
        self.EAR_RESET_THRESH = self.EAR_THRESH + 0.03 
        
        # 4. Initialize State Variables
        self.closure_frames = 0  # Frame counter for closure detection
        self.closure_start_time = None  # Time-based tracking (more reliable than frames)
        self.alert_on = False
        self.show_enhanced = False  # Toggle enhanced frame view for debugging
        
        # Face tracking continuity
        self.no_face_frames = 0
        self.NO_FACE_GRACE = 15  # ~0.6s grace period at 24fps before reset
        
        # Time-based threshold (seconds)
        self.ALERT_DURATION_SEC = 2.7  # Replaces frame-based FRAMES_TO_ALERT
        
        # Landmark quality thresholds
        self.MIN_LANDMARK_VISIBILITY = 0.5  # Confidence threshold for landmark quality
        
        # 5. Initialize Video Capture and MediaPipe
        # Set camera properties from config
        self.cap = cv2.VideoCapture(self.CFG["camera"]["index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CFG["camera"]["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CFG["camera"]["height"])
        self.cap.set(cv2.CAP_PROP_FPS, self.CFG["camera"]["fps_target"])

        if not self.cap.isOpened():
            raise IOError(f"[Error] Camera index {self.CFG['camera']['index']} not found.")
        
        self.mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        
        # --- NEW: Initialize CLAHE for low-light enhancement ---
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 6. Initialize Serial Communication
        self.ser = self._setup_serial()

    def _load_config(self, path):
        """Loads and validates the configuration file."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"[ERROR] Could not load/decode config.json: {e}")
            exit()
        except Exception as e:
            print(f"[ERROR] Unexpected error loading config: {e}")
            exit()

    def _load_model(self, path):
        """Loads the trained model and associated scaler/features."""
        try:
            bundle = joblib.load(path)
            return bundle["scaler"], bundle["model"], bundle["feature_order"]
        except FileNotFoundError:
            print(f"[ERROR] Model file not found at: {path}")
            exit()
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            exit()

    def _setup_csv_log(self):
        """Creates the CSV log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "ear", "mar", "p_drowsy", "state"])

    def _setup_serial(self):
        """Initializes the serial connection if enabled."""
        if self.CFG.get("serial", {}).get("enabled", False) and serial:
            try:
                ser_obj = serial.Serial(self.CFG["serial"]["port"], self.CFG["serial"]["baud"], timeout=0.2)
                print(f"[Serial] Connected on {self.CFG['serial']['port']}")
                return ser_obj
            except Exception as e:
                print(f"[Serial] Could not open port or serial library is missing: {e}")
        elif self.CFG.get("serial", {}).get("enabled", False) and not serial:
             print("[Serial] Warning: Serial enabled in config, but 'pyserial' not installed.")
        return None

    def _log_event(self, e_val, m_val, p_d, state="drowsy"):
        """Appends a new event record to the CSV log file."""
        try:
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{e_val:.3f}", f"{m_val:.3f}", f"{p_d:.2f}", state
                ])
        except Exception as e:
            print(f"[ERROR] Failed to write to CSV log: {e}")

    def _send_buzzer_signal(self, signal="1\n"):
        """Sends the buzzer signal over the serial port."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(signal.encode('utf-8'))
            except Exception as e:
                print(f"[Serial] Write failed: {e}")
        else:
             print("[Info] Serial not enabled/connected.")

    def run(self):
        """The main video processing and inference loop."""
        print("[Info] Press 'q' to quit, 'b' for buzzer test (if enabled), 'e' to toggle debug view")
        
        while True:
            ok, frame = self.cap.read()
            if not ok:
                cv2.waitKey(1)
                continue
            
            # --- LOW-LIGHT PRE-PROCESSING ---
            # 1. Convert BGR to LAB color space (separates lightness from color)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # 2. Split the LAB channels
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # 3. Apply CLAHE to the L-channel (Luminance/Lightness)
            l_channel_eq = self.clahe.apply(l_channel)
            
            # 4. Merge the enhanced L-channel back with original A and B channels
            lab_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
            
            # 5. Convert the enhanced LAB image back to BGR
            frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            # --- END PRE-PROCESSING ---

            # Now, use the ENHANCED frame (frame_eq) for MediaPipe processing
            H, W = frame_eq.shape[:2]
            rgb = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2RGB)
            res = self.mesh.process(rgb)

            e_val, m_val, p_d = 0.0, 0.0, 0.0
            
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                self.no_face_frames = 0  # Reset no-face counter when face detected
                
                # Check landmark quality (visibility/confidence) - for logging only
                # MediaPipe provides visibility scores for each landmark
                critical_landmarks = LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM]
                visibility_scores = [lm[i].visibility if hasattr(lm[i], 'visibility') else 1.0 for i in critical_landmarks]
                min_visibility = min(visibility_scores) if visibility_scores else 1.0
                
                # Log low-confidence frames but continue processing
                if min_visibility < self.MIN_LANDMARK_VISIBILITY and min_visibility < 1.0:
                    cv2.putText(frame, f"Low confidence ({min_visibility:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                # 1. Feature Calculation
                left_ear = calculate_ear(lm, LEFT_EYE_LANDMARKS, W, H)
                right_ear = calculate_ear(lm, RIGHT_EYE_LANDMARKS, W, H)
                e_val = (left_ear + right_ear) / 2.0
                m_val = calculate_mar(lm, W, H)
                
                # 2. Feature Engineering
                feats = {
                    "ear": e_val, 
                    "mar": m_val,
                    "eye_close": 1.0 if e_val < self.EAR_THRESH else 0.0, # Use configured thresh
                    "mouth_open": 1.0 if m_val > self.CFG["inference"].get("mouth_open_threshold", 0.60) else 0.0 # Use configured thresh
                }
                
                # 3. Model Inference (FOR DISPLAY ONLY)
                x = np.array([[feats[k] for k in self.feat_order]])
                x_scaled = self.scaler.transform(x)
                p_d = self.clf.predict_proba(x_scaled)[0, 1]

                # --- DROWSINESS LOGIC (Time-based with ML confirmation) ---
                
                # 4. Time-based eye closure tracking
                if e_val < self.EAR_THRESH:
                    if self.closure_start_time is None:
                        self.closure_start_time = time.time()
                    closure_duration = time.time() - self.closure_start_time
                    self.closure_frames += 1  # Keep for backward compatibility
                else:
                    self.closure_start_time = None
                    closure_duration = 0.0
                    self.closure_frames = 0
                    
                # 5. Alert Trigger (DUAL-GATE: time-based + ML confirmation)
                # BOTH conditions must be met to trigger alert
                if (closure_duration >= self.ALERT_DURATION_SEC and 
                    p_d >= self.p_thresh and  # ML confirmation gate
                    not self.alert_on):
                    self.alert_on = True
                    self._log_event(e_val, m_val, p_d, "drowsy_microsleep")
                    self._send_buzzer_signal(signal="1\n")
                    
                # 6. Reset Alert (Only when eyes are clearly open again)
                if e_val >= self.EAR_RESET_THRESH:
                    self.alert_on = False
                    self.closure_frames = 0 # Reset counter when alert is off
                
                # --- END LOGIC ---

            else:
                # No face detected - use grace period before resetting
                self.no_face_frames += 1
                cv2.putText(frame, f"No face ({self.no_face_frames})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Only reset drowsiness state after grace period
                if self.no_face_frames > self.NO_FACE_GRACE:
                    self.closure_frames = 0
                    self.closure_start_time = None
                    self.alert_on = False 
                
            # --- Display & Visualization ---
            
            # Determine which frame to display: enhanced (debug) or original (natural)
            display_frame = frame_eq if self.show_enhanced else frame

            # Display EAR, MAR, and probability (drawn on the display_frame)
            cv2.putText(display_frame, f"EAR: {e_val:.3f}", (10, H - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"MAR: {m_val:.3f}", (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, f"p(drowsy): {p_d:.2f}", (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Display the closure frame count (helpful for debugging)
            cv2.putText(display_frame, f"Closure Frames: {self.closure_frames}", (W - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the view mode
            # cv2.putText(display_frame, f"View: {'Enhanced (CLAHE)' if self.show_enhanced else 'Original (BGR)'} (Press 'e')", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)

            # Draw the Alert Banner
            if self.alert_on:
                # Always draw the alert on the actual frame shown to the user
                cv2.rectangle(display_frame, (0, 0), (W, 50), (0, 0, 255), -1)
                cv2.putText(display_frame, "DROWSINESS DETECTED!", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            cv2.imshow("Vigi-Drive (Image-trained model)", display_frame)

            # --- Key Event Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                self._send_buzzer_signal(signal="1\n")
                print("[Info] Test buzz signal sent.")
            elif key == ord("e"):
                # Toggle between enhanced and original frame view
                self.show_enhanced = not self.show_enhanced
                print(f"[Debug] Showing enhanced frame: {self.show_enhanced}")

        self.release_resources()

    def release_resources(self):
        """Releases the camera and closes the serial port."""
        print("[Info] Releasing resources...")
        self.cap.release()
        cv2.destroyAllWindows()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[Info] Serial port closed.")

# --- Execution Block ---
if __name__ == "__main__":
    try:
        # Use a specific config file name if your script isn't named 'realtime_ml_frame.py'
        detector = DrowsinessDetector(config_path="../config/config.json")
        detector.run()
    except IOError as e:
        print(f"[FATAL ERROR] {e}")
    except Exception as e:
        print(f"[FATAL ERROR] An unhandled exception occurred: {e}")
        # Ensure resources are released even if an error occurs early
        try:
            detector.release_resources()
        except NameError:
            pass # Detector was never created
