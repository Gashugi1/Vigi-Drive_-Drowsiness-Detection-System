# 🚗 Vigi-Drive: Advanced Drowsiness Detection System

**Vigi-Drive** is a production-grade, real-time drowsiness detection system designed to enhance road safety. Powered by computer vision and machine learning, it monitors driver fatigue levels using multi-modal analysis of facial features (eyes, mouth, blink rate) and provides progressive alerts to prevent accidents.

![Vigi-Drive Dashboard](https://via.placeholder.com/800x400?text=Vigi-Drive+Dashboard+Preview)

## 🌟 Key Features

*   **Real-Time Monitoring:** High-performance video processing using OpenCV and MediaPipe.
*   **Multi-Modal Detection:**
    *   **EAR (Eye Aspect Ratio):** Detects droopy eyes and sustained closure.
    *   **MAR (Mouth Aspect Ratio):** Detects yawning frequency and duration.
    *   **Blink Rate Analysis:** Identifies abnormal blinking patterns (too slow or too fast).
*   **5-Level Fatigue Classification:**
    *   🟢 **Level 0 (Fully Alert):** Safe driving state.
    *   🟢 **Level 1 (Mild Fatigue):** Early signs, gentle monitoring.
    *   🟡 **Level 2 (Moderate Fatigue):** Visible signs, warning advised.
    *   🟠 **Level 3 (Severe Drowsiness):** Strong indicators, immediate action required.
    *   🔴 **Level 4 (Critical Microsleep):** Sustained eye closure (>3s), critical alarm.
*   **Smart Alerting:** Progressive audio-visual alerts (Beeps -> Alarms) based on fatigue severity.
*   **Event Logging:** Automatically records severe drowsiness events to a secure database for review.
*   **Secure Authentication:** Integrated Google OAuth 2.0 for secure user access.
*   **Modern UI:** Responsive, dark-themed dashboard with real-time metrics and visualizations.

## 🛠️ Tech Stack

*   **Backend:** Python 3.12+, Flask, SQLAlchemy
*   **Computer Vision:** OpenCV, MediaPipe Face Mesh
*   **Machine Learning:** Scikit-learn (Random Forest Classifier)
*   **Database:** SQLite (Production-ready relational DB)
*   **Frontend:** HTML5, CSS3 (Custom Dark Theme), JavaScript (Fetch API)

## 🚀 Installation & Setup

### Prerequisites
*   Python 3.10 or higher
*   A webcam (built-in or USB)
*   Google Cloud Console Account (for OAuth)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/vigidrive.git
cd vigidrive
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Google OAuth
To enable secure login, you need Google OAuth credentials:
1.  Go to [Google Cloud Console](https://console.cloud.google.com/).
2.  Create a new project and enable the **Google People API**.
3.  Configure the **OAuth Consent Screen** (External).
4.  Create **OAuth Client ID** credentials (Web Application).
    *   **Authorized Redirect URIs:** `http://127.0.0.1:5001/login/callback`
5.  Download the `client_secret_XXXX.json` file.
6.  Rename it to `client_secret.json` and place it in the project root.

### 5. Initialize the Database
The application will automatically create `instance/vigidrive.db` on the first run.

## 🏃‍♂️ Usage

1.  **Start the Application:**
    ```bash
    python app.py
    ```
2.  **Access the Dashboard:**
    Open your browser and navigate to `http://127.0.0.1:5001`.
3.  **Login:** Sign in with your Google account.
4.  **Start Monitoring:**
    *   Go to the **Live Monitor** tab.
    *   Click **Start Monitoring**.
    *   Allow camera access when prompted.
5.  **Stop Monitoring:** Click **Stop Monitoring** to release the camera and save resources.

## ⚙️ Configuration (`config.json`)

You can fine-tune the detection parameters in `config.json`:

```json
{
    "camera": {
        "index": 0,
        "width": 640,
        "height": 480,
        "fps_target": 30
    },
    "inference": {
        "model_path": "models/drowsiness_model.pkl",
        "prob_threshold": 0.65,
        "EAR_CLOSED_THRESH": 0.21,
        "FRAMES_TO_ALERT": 30
    }
}
```

## 📊 Project Structure

```
vigidrive/
├── app.py                 # Main Flask application
├── fatigue_classifier.py  # Core fatigue logic & scoring
├── realtime_ml_frame.py   # Standalone script for testing
├── config.json            # Configuration settings
├── client_secret.json     # Google OAuth credentials (ignored)
├── requirements.txt       # Python dependencies
├── instance/              # SQLite database
├── static/
│   ├── css/style.css      # Custom styling
│   └── js/dashboard.js    # Frontend logic
├── templates/             # HTML templates
│   ├── home.html
│   ├── monitor.html
│   └── ...
└── models/                # Trained ML models
```

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed with ❤️ for Road Safety.*
