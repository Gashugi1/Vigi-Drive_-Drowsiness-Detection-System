import cv2
import time
import csv
import os
import json
import numpy as np
import joblib
import mediapipe as mp

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

# --- MAIN CLASS ---

class DrowsinessDetector:
    def __init__(self, config_path="config.json"):
        # 1. Load Configuration
        self.CFG = self._load_config(config_path)
        
        # 2. Setup Logging
        os.makedirs("logs", exist_ok=True)
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
        self.closure_frames = 0 # Replaces 'self.hit'
        self.alert_on = False
        
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
        print("[Info] Press 'q' to quit, 'b' for buzzer test (if enabled)")
        
        while True:
            ok, frame = self.cap.read()
            if not ok:
                cv2.waitKey(1)
                continue
            
            # --- NEW: LOW-LIGHT PRE-PROCESSING ---
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
                    "mouth_open": 1.0 if m_val > 0.60 else 0.0 # Can also configure this
                }
                
                # 3. Model Inference (FOR DISPLAY ONLY)
                x = np.array([[feats[k] for k in self.feat_order]])
                x_scaled = self.scaler.transform(x)
                p_d = self.clf.predict_proba(x_scaled)[0, 1]

                # --- NEW DROWSINESS LOGIC (Based on EAR duration) ---
                
                # 4. Check for physical eye closure
                if e_val < self.EAR_THRESH:
                    self.closure_frames += 1
                else:
                    self.closure_frames = 0
                    
                # 5. Alert Trigger and Action
                if self.closure_frames >= self.FRAMES_TO_ALERT and not self.alert_on:
                    self.alert_on = True
                    self._log_event(e_val, m_val, p_d, "drowsy_microsleep")
                    self._send_buzzer_signal(signal="1\n")
                    
                # 6. Reset Alert (Only when eyes are clearly open again)
                if e_val >= self.EAR_RESET_THRESH:
                    self.alert_on = False
                    self.closure_frames = 0 # Reset counter when alert is off
                
                # --- END NEW LOGIC ---

            else:
                cv2.putText(frame, "No face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # Reset counter when face is lost
                self.closure_frames = 0
                self.alert_on = False 
                
            # --- Display & Visualization ---
            
            # We draw overlays on the ORIGINAL 'frame', so the user sees a natural image
            cv2.putText(frame, f"EAR: {e_val:.3f}", (10, H - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {m_val:.3f}", (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"p(drowsy): {p_d:.2f}", (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            # Display the closure frame count (helpful for debugging)
            cv2.putText(frame, f"Closure Frames: {self.closure_frames}", (W - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            # Draw the Alert Banner on the original frame
            if self.alert_on:
                cv2.rectangle(frame, (0, 0), (W, 50), (0, 0, 255), -1)
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Show the original, natural-looking frame to the user
            cv2.imshow("Vigi-Drive (Image-trained model)", frame)
            
            # (Optional) Uncomment the line below to see the enhanced frame
            # cv2.imshow("Enhanced Frame (for MediaPipe)", frame_eq)

            # --- Key Event Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                self._send_buzzer_signal(signal="1\n")
                print("[Info] Test buzz signal sent.")

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
        detector = DrowsinessDetector(config_path="config.json")
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

