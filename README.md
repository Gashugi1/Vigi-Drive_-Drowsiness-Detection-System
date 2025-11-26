# 🚗 Vigi-Drive: Advanced Drowsiness Detection System

**Vigi-Drive** is a production-grade, real-time drowsiness detection system designed to enhance road safety. Powered by computer vision and machine learning, it monitors driver fatigue levels using multi-modal analysis of facial features (eyes, mouth, blink rate) and provides progressive alerts to prevent accidents.

## 🌟 Key Features

*   **Real-Time Monitoring:** High-performance video processing using OpenCV and MediaPipe
*   **Multi-Modal Detection:**
    *   **EAR (Eye Aspect Ratio):** Detects droopy eyes and sustained closure
    *   **MAR (Mouth Aspect Ratio):** Detects yawning frequency and duration
    *   **Blink Rate Analysis:** Identifies abnormal blinking patterns
*   **5-Level Fatigue Classification:**
    *   🟢 **Level 0-1:** Alert / Mild Fatigue
    *   🟡 **Level 2:** Moderate Fatigue
    *   🟠 **Level 3:** Severe Drowsiness
    *   🔴 **Level 4:** Critical Microsleep (\u003e2.7s eye closure)
*   **Smart Alerting:** Progressive audio-visual alerts based on fatigue severity
*   **Event Logging:** Automatically records drowsiness events for authenticated users
*   **Secure Authentication:** Integrated Google OAuth 2.0
*   **Modern UI:** Responsive dashboard with real-time analytics

## 🛠️ Tech Stack

*   **Backend:** Python 3.10+, Flask, SQLAlchemy
*   **Computer Vision:** OpenCV, MediaPipe Face Mesh
*   **Machine Learning:** Logistic Regression (trained on image features)
*   **Database:** SQLite with optimized indices
*   **Frontend:** HTML5, CSS3, JavaScript

## 🚀 Quick Start

### Prerequisites
*   Python 3.10 or higher
*   Webcam (built-in or USB)
*   Google OAuth credentials (see docs/OAUTH_SETUP.md)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/vigidrive.git
cd vigidrive

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp config/.env.example .env
# Edit .env and add your Google OAuth credentials

# 5. Run application
./run.sh  # or: cd src && python app.py
```

The application will be available at `http://localhost:5001`

## 📊 Project Structure

```
vigidrive/
├── src/                      # Source code
│   ├── app.py               # Main Flask application
│   ├── core/                # Core detection modules
│   │   ├── fatigue_classifier.py
│   │   └── realtime_ml_frame.py
│   ├── models/              # Trained ML models
│   ├── static/              # Web assets (CSS, JS, sounds, images)
│   └── templates/           # HTML templates
├── config/                   # Configuration files
│   ├── config.json          # Detection parameters
│   └── .env.example         # Environment template
├── data/                     # Application data
│   ├── features/            # Feature CSVs
│   ├── logs/                # Event logs
│   └── instance/            # SQLite database
├── docs/                     # Documentation
│   ├── OAUTH_SETUP.md
│   ├── SYSTEM_REVIEW.md
│   └── DOCUMENTATION_WALKTHROUGH.md
├── tools/                    # Development tools
│   ├── training/            # Model training scripts
│   └── benchmarks/          # Performance tests
├── tests/                    # Test suite
├── run.sh                    # Production startup script
└── requirements.txt          # Python dependencies
```

## ⚙️ Configuration

Edit `config/config.json` to adjust detection parameters:

```json
{
  "camera": { "index": 0, "width": 640, "height": 480, "fps_target": 24 },
  "inference": {
    "model_path": "../src/models/earmar_img_logreg.joblib",
    "prob_threshold": 0.70,
    "EAR_CLOSED_THRESH": 0.20,
    "FRAMES_TO_ALERT": 30
  }
}
```

## 🏃 Usage

1. Start the application: `./run.sh`
2. Navigate to `http://localhost:5001`
3. Sign in with Google OAuth
4. Go to **Monitor** tab
5. Click **Start Monitoring** to begin detection
6. View **Analytics** for performance metrics
7. Check **Events** for drowsiness history

## 🧪 Testing

```bash
# Run test suite
pytest tests/ -v

# Run specific test category
pytest tests/test_features.py -v

# Run performance benchmarks
python tools/benchmarks/performance_test.py
```

## 📈 Performance

* **Frame Processing:** ~47ms latency
* **ML Inference:** ~18ms per frame
* **Memory Usage:** ~150MB
* **CPU Usage:** ~15-25% (single core)
* **Detection Rate:** 94%+ accuracy

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 📚 Documentation

* [System Architecture](docs/SYSTEM_REVIEW.md)
* [OAuth Setup Guide](docs/OAUTH_SETUP.md)
* [Development Walkthrough](docs/DOCUMENTATION_WALKTHROUGH.md)

---

*Developed with ❤️ for Road Safety*
