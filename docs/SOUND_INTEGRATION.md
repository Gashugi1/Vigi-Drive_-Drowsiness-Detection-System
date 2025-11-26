# Sound Integration Guide

## Overview

The VigiDrive system uses browser-based audio alerts to notify drivers when drowsiness is detected. This replaces the previous Arduino buzzer integration with a more accessible and cross-platform solution.

## Architecture

### Components

**Frontend Audio System:**
- HTML5 Audio API for playback
- Status polling mechanism (`/status` endpoint)
- Alert state monitoring
- Audio file management

**Backend Alert Trigger:**
- Dual-gate detection logic (EAR + ML confirmation)
- Real-time alert state in `/status` JSON
- Configuration-driven sound settings

### Data Flow

```
Drowsiness Detected → Alert State (alert: true) → Frontend Poll → Play Audio
    ↓
Eyes Open → Alert Clear (alert: false) → Frontend Poll → Stop Audio
```

## Configuration

### Enable Sound Alerts

Edit `config/config.json`:

```json
{
  "features": {
    "sound_alerts": {
      "enabled": true,
      "sound_file": "sounds/drowsiness_alert.mp3",
      "loop": true
    }
  }
}
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable/disable sound alerts |
| `sound_file` | string | `"sounds/drowsiness_alert.mp3"` | Path to audio file (relative to `src/static/`) |
| `loop` | boolean | `true` | Whether to loop audio until alert cleared |

### Supported Audio Formats

- **MP3** (recommended) - Best browser compatibility
- **WAV** - Highest quality, larger file size
- **AIFF** - Apple ecosystem, good quality

**Audio Files Location:** `src/static/sounds/`

---

## Frontend Implementation

### Status Polling

The frontend polls the `/status` endpoint every 500ms to monitor alert state:

```javascript
// Monitor alert state
setInterval(async () => {
    const response = await fetch('/status');
    const data = await response.json();
    
    if (data.alert && !audioPlaying) {
        playAlertSound();
    } else if (!data.alert && audioPlaying) {
        stopAlertSound();
    }
}, 500);
```

### Audio Playback

**Play Alert:**
```javascript
const audio = new Audio('/static/sounds/drowsiness_alert.mp3');
audio.loop = true;  // From config

function playAlertSound() {
    audio.play().catch(err => {
        console.error('Audio playback failed:', err);
        // Fallback: visual-only alert
    });
    audioPlaying = true;
}
```

**Stop Alert:**
```javascript
function stopAlertSound() {
    audio.pause();
    audio.currentTime = 0;  // Reset to beginning
    audioPlaying = false;
}
```

### Browser Autoplay Policies

Modern browsers restrict autoplay. Handle gracefully:

```javascript
// Require user interaction before audio can play
document.addEventListener('click', () => {
    audio.load();  // Prime the audio context
}, { once: true });
```

---

## Backend Alert Logic

### Dual-Gate Trigger

Alert triggered when **BOTH** conditions met:

1. **Temporal Gate**: Eyes closed for ≥ 2.7 seconds
2. **ML Gate**: Drowsiness probability ≥ 0.8

```python
# From src/app.py
if (closure_duration >= self.ALERT_DURATION_SEC and 
    p_d >= self.p_thresh and 
    not self.alert_on):
    
    self.alert_on = True
    self._log_event(e_val, m_val, p_d, "drowsy_microsleep")
```

### Alert State in /status Endpoint

```json
{
  "ear": 0.18,
  "mar": 0.25,
  "p_drowsy": 0.85,
  "alert": true,  // ← Frontend monitors this
  "state": "drowsy",
  "fatigue_level": 4
}
```

---

## Testing

### Manual Testing Checklist

**Basic Functionality:**
- [ ] Sound plays when drowsiness detected
- [ ] Sound stops when eyes open
- [ ] Volume is appropriate (not too loud/quiet)
- [ ] No audio overlap from multiple alerts

**Cross-Browser:**
- [ ] Chrome (desktop + Android)
- [ ] Firefox (desktop + Android)
- [ ] Safari (macOS + iOS)
- [ ] Edge

**Edge Cases:**
- [ ] Alert triggered → page refresh → no orphaned audio
- [ ] Background tab → alert → audio still plays
- [ ] Mobile autoplay restrictions handled
- [ ] Network latency doesn't cause delays

### Performance Metrics

**Target Performance:**
- Alert trigger → audio start: **< 300ms**
- CPU usage during playback: **< 5%**
- Memory: No leaks after 100+ alerts

**Measurement:**
```javascript
const startTime = performance.now();
audio.play().then(() => {
    const latency = performance.now() - startTime;
    console.log(`Audio latency: ${latency}ms`);
});
```

### Automated Testing

See `tests/test_sound_integration.py` (to be created) for:
- Configuration validation
- Status endpoint response verification
- Alert state transitions

---

## Troubleshooting

### Audio Doesn't Play

**Possible Causes:**
1. Browser autoplay policy blocked it
2. Audio file path incorrect
3. Network error loading file
4. User muted system/browser

**Solutions:**
```javascript
audio.play().catch(err => {
    if (err.name === 'NotAllowedError') {
        // Show visual prompt: "Click to enable sound alerts"
        showAudioEnablePrompt();
    } else if (err.name === 'NotSupportedError') {
        // Audio format not supported
        console.error('Audio format not supported');
    }
});
```

### Audio Continues After Alert Cleared

**Cause:** Frontend polling missed the state change

**Solution:** Ensure `audio.pause()` is called unconditionally:
```javascript
if (!data.alert) {
    if (audioPlaying) {
        stopAlertSound();
    }
}
```

### High CPU Usage

**Cause:** Inefficient polling or audio codec

**Solutions:**
- Increase polling interval (500ms → 750ms)
- Use MP3 instead of uncompressed WAV
- Implement WebSocket for real-time updates (future enhancement)

---

## Production Deployment

### Checklist

- [ ] Sound files committed to `src/static/sounds/`
- [ ] Config verified in `config/config.json`
- [ ] Cross-browser testing complete
- [ ] Performance benchmarks meet targets
- [ ] Fallback for no-audio scenarios
- [ ] User documentation updated

### Optimization

**Preload Audio:**
```html
<audio preload="auto" id="alertSound">
    <source src="/static/sounds/drowsiness_alert.mp3" type="audio/mpeg">
    <source src="/static/sounds/drowsiness_alert.wav" type="audio/wav">
</audio>
```

**Lazy Loading:**
Only load audio file after first user interaction to save bandwidth.

---

## Future Enhancements

- Progressive alert volume (gradual increase)
- Multiple alert sounds based on fatigue level
- WebSocket for real-time alerts (eliminate polling)
- Voice announcements ("Please pull over safely")
- Adjustable volume in UI settings

---

## Related Issues

- [#15 - Implement sound triggering when drowsiness detected](https://github.com/Gashugi1/Vigi-Drive_-Drowsiness-Detection-System/issues/15)
- [#16 - Ensure sound stops when alert dismissed](https://github.com/Gashugi1/Vigi-Drive_-Drowsiness-Detection-System/issues/16)
- [#17 - Cross-browser audio testing](https://github.com/Gashugi1/Vigi-Drive_-Drowsiness-Detection-System/issues/17)

**Milestone:** [Sound Integration](https://github.com/Gashugi1/Vigi-Drive_-Drowsiness-Detection-System/milestone/9)

---

**Last Updated:** 2025-11-26  
**Author:** VigiDrive Development Team
