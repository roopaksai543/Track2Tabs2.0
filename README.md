# Track2Tabs 2.0

## Description
Track2Tabs 2.0 is a web-based application that analyzes audio files and generates musical insights such as chord progressions and tempo. The goal of the project is to make it easier for musicians—especially beginners—to understand and learn songs directly from audio.

Users can upload an audio file, and the app processes it using audio analysis techniques to extract meaningful musical information in a simple and intuitive interface.

This project currently runs locally on my machine while deployment is being finalized. Public hosting will be added in a future update.

## Live Website
[Track2Tabs 2.0](https://track2tabs20-git-main-roopaksais-projects-f8ff8358.vercel.app/)

## Changelog
See full version history and updates [here](https://github.com/roopaksai543/Track2Tabs2.0/blob/main/CHANGELOG.md)

## Current Status (v0.2.0)
- Supports MP# and WAV file uploads
- Performs stem seperation to isolate melody and drums
- Detects tempo (BPM) automatically
- Uses a sequence-based PyTorch model for chord prediction
- Displays a chord timeline synced with audio playback

### Works well for: 
- Audio with clear guitar or harmonic background
- Simpler chord progression

### Limitations:
- Less accurate with dense mixes/heavy melodies
- Struggles with complex layered instrumentation

## Future Plans
- Improve accuracy on full songs with complex arrangements
- Train on more realistic and diverse data
- Better handling fo melody vs harmony seperation
- Improve chord labeling consistency
- Add support for more chord types and extensions
- Add backend hosted on Railway

<p align="center">
  <sub>
    <i>
      Built by <b>Roopaksai Sivakumar</b><br>
      Computer Engineering @ UC Irvine
    </i>
  </sub>
</p>