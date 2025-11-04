import argparse, cv2, mediapipe as mp, numpy as np, pandas as pd
from pathlib import Path
import sys

# MediaPipe landmark indices for EAR and MAR calculation
LEFT_EYE_LANDMARKS = [33,160,158,133,153,144]
RIGHT_EYE_LANDMARKS = [263,387,385,362,380,373]
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14

def l2(a, b): 
    """Calculates the Euclidean distance between two 2D points."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def denorm(p, w, h): 
    """Converts normalized (0.0-1.0) landmark coordinates to pixel values."""
    return int(p.x * w), int(p.y * h)

def ear(lm, idx, w, h):
    """Calculates the Eye Aspect Ratio (EAR)."""
    # P1, P4 are horizontal points. P2, P6, P3, P5 are vertical points.
    P1, P2, P3, P4, P5, P6 = [denorm(lm[i], w, h) for i in idx]
    vertical_dist = l2(P2, P6) + l2(P3, P5)
    horizontal_dist = l2(P1, P4)
    return vertical_dist / (2 * horizontal_dist + 1e-6)

def mar(lm, w, h):
    """Calculates the Mouth Aspect Ratio (MAR)."""
    L = denorm(lm[MOUTH_LEFT], w, h)
    R = denorm(lm[MOUTH_RIGHT], w, h)
    T = denorm(lm[MOUTH_TOP], w, h)
    B = denorm(lm[MOUTH_BOTTOM], w, h)
    # Vertical distance (T-B) / Horizontal distance (L-R)
    return l2(T, B) / (l2(L, R) + 1e-6)

def main():
    ap = argparse.ArgumentParser(description="Extracts EAR and MAR features from images for drowsiness detection training.")
    ap.add_argument("--root", required=True, help="Root directory containing 'alert/' (label 0) and 'drowsy/' (label 1) subfolders.")
    ap.add_argument("--out", default="features/features_images.csv", help="Output path for the generated CSV file.")
    ap.add_argument("--perclos_thr", type=float, default=0.20, help="EAR threshold for binary 'eye_close' feature.")
    ap.add_argument("--yawn_thr",    type=float, default=0.60, help="MAR threshold for binary 'mouth_open' feature.")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    output_path = Path(args.out) # Define output path here
    
    print(f"\n[DEBUG PATHS]")
    print(f"  Input Dataset Root (Resolved): {root.resolve()}")
    print(f"  Output CSV Path (Resolved): {output_path.resolve()}")
    print(f"  Execution Directory: {Path.cwd().resolve()}")
    print(f"-------------------------\n")
    
    if not root.exists():
        print(f"Error: Root directory not found: {root}")
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    # Initialize MediaPipe FaceMesh for static images (faster processing)
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)

    rows = []
    
    print(f"Starting feature extraction...") # Simplified start message
    
    # Process images in 'alert' (label 0) and 'drowsy' (label 1) folders
    for label_name, lab in [("alert", 0), ("drowsy", 1)]:
        folder = root / label_name
        if not folder.exists():
            print(f"Warning: Missing folder: {folder}. Skipping {label_name} data.")
            continue

        print(f"Processing {label_name} images in {folder}...")

        image_count = 0
        
        # Iterate over all files recursively in the subfolder
        for p in folder.rglob("*"):
            if p.suffix.lower() not in exts:
                # Skip files that are not common image extensions
                continue

            # print(f"Processing {p}...") # Optional: uncomment for verbose debugging
            img = cv2.imread(str(p))
            if img is None:
                print(f"Skipping {p}, unable to read image.")
                continue
                
            h, w = img.shape[:2]
            # Convert to RGB as required by MediaPipe
            res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
            
            if not res.multi_face_landmarks:
                # print(f"No face detected in {p}") # Optional: uncomment for verbose debugging
                continue
                
            # Get landmarks and calculate features
            lm = res.multi_face_landmarks[0].landmark
            e = (ear(lm, LEFT_EYE_LANDMARKS, w, h) + ear(lm, RIGHT_EYE_LANDMARKS, w, h)) / 2.0
            m = mar(lm, w, h)
            
            rows.append({
                "ear": e, "mar": m,
                "eye_close": 1.0 if e < args.perclos_thr else 0.0,
                "mouth_open": 1.0 if m > args.yawn_thr else 0.0,
                "label": lab, 
                "class": label_name, 
                "image": p.name
            })
            image_count += 1
            
        print(f"Finished processing {image_count} images for {label_name}.")

    if rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\n[SUCCESS] Saved feature CSV to {output_path.resolve()}, processed total {len(rows)} images.")
    else:
        print("\n[WARNING] No usable images with faces were found. CSV file was not created.")

if __name__ == "__main__":
    main()
