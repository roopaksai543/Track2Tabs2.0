# Track2Tabs 2.0

## Description
Track2Tabs 2.0 is a web-based application that analyzes audio files and generates musical insights such as chord progressions and tempo. The goal of the project is to make it easier for musicians—especially beginners—to understand and learn songs directly from audio.

Users can upload an audio file, and the app processes it using audio analysis techniques to extract meaningful musical information in a simple and intuitive interface.

The Vite frontend can be hosted on Vercel and the audio-analysis API on Railway.

## Live Website
[Track2Tabs 2.0](https://track2tabs20-git-main-roopaksais-projects-f8ff8358.vercel.app/)

## Deploy the backend to Railway

1. Push this repository to GitHub.
2. In Railway, choose **New Project → Deploy from GitHub repo** and select this repository. Railway automatically detects the root `Dockerfile`.
3. In the service settings, set the health-check path to `/health` and generate a public domain under **Networking**.
4. In the Railway service's **Variables** tab, add:

   ```text
   ALLOWED_ORIGINS=https://your-frontend-domain.vercel.app
   ```

   Multiple allowed origins can be supplied as a comma-separated list. Do not add a trailing slash.

5. In the Vercel project's environment variables, add the Railway public URL (without a trailing slash):

   ```text
   VITE_API_BASE=https://your-backend.up.railway.app
   ```

6. Redeploy the Vercel frontend so Vite includes the new value in its production build.
7. Verify the backend before uploading a song:

   ```sh
   curl https://your-backend.up.railway.app/health
   ```

   It should return `{"ok":true}`.

The Docker image includes Demucs's pretrained model. Audio processing is CPU- and memory-intensive, so increase the Railway service's resources if it is terminated while analyzing longer tracks. Uploaded files and generated stems are removed after every request.

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
