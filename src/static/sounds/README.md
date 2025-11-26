Note: Sound File Required
=========================

A sound file needs to be placed at this location:
/Users/Itanzi/Projects/vigidrive/static/sounds/drowsiness_alert.wav

Options for obtaining a sound file:

1. Use macOS system sounds:
   cp /System/Library/Sounds/Sosumi.aiff drowsiness_alert.wav
   (Note: May need conversion from .aiff to .wav)

2. Download from free resources:
   - FreeSound.org (search "alarm" or "alert")
   - ZapSplat.com
   
3. Generate programmatically:
   - Use Python library like pydub or scipy
   
4. Use text-to-speech:
   say "Drowsiness detected. Please pull over safely." -o drowsiness_alert.aiff
   # Then convert to wav

For now, a placeholder file will be created in the next step.
