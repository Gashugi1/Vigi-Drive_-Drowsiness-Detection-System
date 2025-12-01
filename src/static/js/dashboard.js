// Dashboard real-time updates with Multi-Modal Fatigue Detection
class DashboardMonitor {
  constructor() {
    this.refreshInterval = 1000; // 1 second
    this.soundEnabled = false;
    this.alertAudio = null;
    this.lastAlertLevel = 0;
    this.alertShown = false;
    this.init();
  }

  init() {
    this.startPolling();
    this.setupSoundToggle();
  }

  async fetchStatus() {
    try {
      const response = await fetch('/status');
      if (!response.ok) throw new Error('Failed to fetch status');
      return await response.json();
    } catch (error) {
      console.error('Error fetching status:', error);
      return null;
    }
  }

  updateMetrics(data) {
    // Update EAR
    const earEl = document.getElementById('ear');
    if (earEl) {
      earEl.textContent = data.ear.toFixed(3);
      this.animateValue(earEl);
    }

    // Update MAR
    const marEl = document.getElementById('mar');
    if (marEl) {
      marEl.textContent = data.mar.toFixed(3);
      this.animateValue(marEl);
    }

    // Update Fatigue Level (5 levels)
    const fatigueLevel = data.fatigue_level || 0;
    const fatigueScore = data.fatigue_score || 0;
    const fatigueLevelEl = document.getElementById('fatigue_level');
    const fatigueScoreEl = document.getElementById('fatigue_score');

    if (fatigueLevelEl) {
      const levelNames = ['Fully Alert', 'Mild Fatigue', 'Moderate Fatigue', 'Severe Drowsiness', 'Critical!'];
      const levelColors = ['success', 'info', 'warning', 'danger', 'danger'];

      fatigueLevelEl.textContent = levelNames[fatigueLevel];
      fatigueLevelEl.className = `badge rounded-pill bg-${levelColors[fatigueLevel]}`;
      this.animateValue(fatigueLevelEl);

      // Trigger alert for Level 3+ (Severe or Critical)
      if (fatigueLevel >= 3 && !this.alertShown) {
        this.showDrowsinessAlert(fatigueLevel, levelNames[fatigueLevel]);
        this.triggerAlert(fatigueLevel);
        this.alertShown = true;
        this.lastAlertLevel = fatigueLevel;
      } else if (fatigueLevel < 3) {
        // Reset alert flag when fatigue drops
        this.alertShown = false;
      }
    }

    if (fatigueScoreEl) {
      fatigueScoreEl.textContent = fatigueScore.toFixed(1) + '%';
    }

    // Update State
    const stateEl = document.getElementById('state');
    if (stateEl) {
      const stateText = data.state.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      stateEl.textContent = stateText;
      stateEl.className = 'badge rounded-pill ';

      // Color based on fatigue level
      if (fatigueLevel >= 4) {
        stateEl.classList.add('bg-danger');
      } else if (fatigueLevel === 3) {
        stateEl.classList.add('bg-warning');
      } else if (fatigueLevel === 2) {
        stateEl.classList.add('bg-info');
      } else if (fatigueLevel === 1) {
        stateEl.classList.add('bg-light', 'text-dark');
      } else {
        stateEl.classList.add('bg-success');
      }
    }

    // Update Running Status
    const runningEl = document.getElementById('running');
    if (runningEl) {
      runningEl.textContent = data.running ? 'Running' : 'Stopped';
      runningEl.className = 'badge rounded-pill ';
      runningEl.classList.add(data.running ? 'bg-success' : 'bg-secondary');

      // Sync toggle switch
      const toggle = document.getElementById('detectionToggle');
      if (toggle && toggle.checked !== data.running) {
        toggle.checked = data.running;
        document.getElementById('toggleLabel').textContent = data.running ? 'Detection On' : 'Detection Off';
        document.getElementById('toggleLabel').style.color = data.running ? '#10b981' : '#6b7280';
      }
    }

    // Update Multi-Modal Metrics
    const headPitchEl = document.getElementById('head_pitch');
    if (headPitchEl) {
      headPitchEl.textContent = data.head_pitch ? data.head_pitch.toFixed(1) + '°' : '0.0°';
    }

    const yawnCountEl = document.getElementById('yawn_count');
    if (yawnCountEl) {
      yawnCountEl.textContent = data.yawn_count || 0;
    }

    const blinkRateEl = document.getElementById('blink_rate');
    if (blinkRateEl) {
      blinkRateEl.textContent = data.blink_rate ? data.blink_rate.toFixed(1) + ' bpm' : '0.0 bpm';
    }
  }

  showDrowsinessAlert(level, levelName) {
    // Call the global function from monitor.html
    if (typeof showDrowsinessAlert === 'function') {
      showDrowsinessAlert(level, levelName);
    }
  }

  animateValue(element) {
    element.style.transform = 'scale(1.1)';
    setTimeout(() => {
      element.style.transform = 'scale(1)';
    }, 200);
  }

  triggerAlert(fatigueLevel) {
    if (this.soundEnabled && this.alertAudio) {
      this.alertAudio.play().catch(e => console.log('Audio play failed:', e));
    }

    // Visual alert with intensity based on fatigue level
    const videoContainer = document.querySelector('.video-container');
    if (videoContainer && !videoContainer.classList.contains('alert-flash')) {
      videoContainer.classList.add('alert-flash');
      const duration = fatigueLevel === 4 ? 2000 : 1000;
      setTimeout(() => {
        videoContainer.classList.remove('alert-flash');
      }, duration);
    }
  }

  setupSoundToggle() {
    const soundToggle = document.getElementById('sound-toggle');
    if (soundToggle) {
      soundToggle.addEventListener('click', () => {
        this.soundEnabled = !this.soundEnabled;
        soundToggle.textContent = this.soundEnabled ? '🔊 Sound On' : '🔇 Sound Off';
        soundToggle.classList.toggle('btn-success', this.soundEnabled);
        soundToggle.classList.toggle('btn-secondary', !this.soundEnabled);
      });
    }
  }

  startPolling() {
    this.fetchAndUpdate();
    setInterval(() => this.fetchAndUpdate(), this.refreshInterval);
  }

  async fetchAndUpdate() {
    const data = await this.fetchStatus();
    if (data) {
      this.updateMetrics(data);
    }
  }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('ear') || document.getElementById('mar')) {
    new DashboardMonitor();
  }
});

// Add visual alert flash animation
const style = document.createElement('style');
style.textContent = `
  .alert-flash {
    animation: flashBorder 1s ease-in-out infinite;
  }
  
  @keyframes flashBorder {
    0%, 100% {
      border-color: var(--dark-border);
    }
    50% {
      border-color: #ef4444;
      box-shadow: 0 0 30px rgba(239, 68, 68, 0.8);
    }
  }
  
  .metric-card .badge,
  .metric-card .metric-value {
    transition: transform 0.2s ease;
  }
`;
document.head.appendChild(style);
