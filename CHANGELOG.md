# Changelog

## v0.2.0 - 2026-03-23

### Added
- Sequence-based chord recognition using a CNN + BiGRU PyTorch model
- Feature extraction pipeline using harmonic audio (chroma CQT, chroma CENS, tonnetz)
- Parallel dataset precomputation for faster training
- Apple Metal (`mps`) support for faster model training
- Stronger inference smoothing and chord segment merging

### Changed
- Replaced old per-window chord classifier with sequence prediction model
- Updated training pipeline to use realistic multi-chord sequences
- Introduced transition regions and "no chord" (`N`) labels in training data
- Improved model architecture for temporal chord tracking
- Fixed chord label generation bug causing pitch offset errors

### Improved
- Reduced single-chord prediction bias
- Improved chord transition timing
- Improved stability of chord segmentation
- Reduced jittery predictions using smoothing techniques
- Better performance on guitar-based audio

### Known Issues
- Still struggles with dense mixes and complex instrumentation
- Background melodies can interfere with chord detection
- Needs more realistic training data for full-song accuracy

---

## v0.1.0 - Initial Version

### Added
- Basic chord detection pipeline
- Audio upload (MP3/WAV)
- Stem separation using Demucs
- Tempo (BPM) detection
- Basic frontend UI with player and timeline

### Changed
- Initial implementation

### Improved
- N/A

### Known Issues
- Poor chord accuracy
- Strong bias toward incorrect repeated chords
- Very unstable chord transitions
- Limited to simple audio with clear harmony