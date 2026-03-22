# Track2Tabs 2.0

## Description
Track2Tabs 2.0 is a web-based application that analyzes audio files and generates musical insights such as chord progressions and tempo. The goal of the project is to make it easier for musicians—especially beginners—to understand and learn songs directly from audio.

Users can upload an audio file, and the app processes it using audio analysis techniques to extract meaningful musical information in a simple and intuitive interface.

## Live Website
[Track2Tabs 2.0]([https://example.com](https://track2tabs20-git-main-roopaksais-projects-f8ff8358.vercel.app/))

## Current Status
The project is currently in an early working state.

### What it can do:
- Upload and play audio files
- Process audio using a backend pipeline
- Analyze chord progressions (currently works best for audio with only guitar chords)
- Estimate tempo (BPM)
- Supports `.mp3` and `.wav` audio files
- Provide a simple UI for interaction

### Current limitations:
- Works reliably only for audio that contains mostly isolated guitar chords
- Accuracy of chord detection can vary with complex mixes (drums, vocals, etc.)
- Backend processing may be slow for longer tracks
- Deployment is split (frontend vs backend)
- Some edge-case errors still occur during analysis

## Future Plans
- Improve chord detection accuracy using better ML models
- Add proper stem separation (separating drums, vocals, etc.)
- Optimize audio processing speed
- Add support for full-song analysis (better segmentation)
- Enhance UI/UX for a more polished experience
- Add visualizations (waveform, chord timeline, etc.)
- Deploy a stable, fully integrated backend
- Allow users to download results (tabs, chord sheets, etc.)



<p align="center">
  <sub>
    <i>
      Built by <b>Roopaksai Sivakumar</b><br>
      Computer Engineering @ UC Irvine
    </i>
  </sub>
</p>
