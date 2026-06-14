# Gesture Audio Mixer

A browser-based performance tool that maps webcam hand motion to live microphone audio effects.

## Controls

- The app is pitch-only.
- Put one hand near the center ring for normal voice.
- Raise your hand to pitch up.
- Lower your hand to pitch down.
- The dashed box shows the full pitch range.

## Diagnose Pitch Quality

- Dry mic: verifies the microphone and output path are clean.
- Fixed +7: tests the pitch shifter without camera movement.
- Live hand: tests camera-driven pitch changes.
- Voice mix: controls how much shifted voice is applied.

## Run

```bash
npm install
npm run dev
```

Open `http://localhost:5175` for local desktop testing, then press **Start camera + mic**. Use headphones to avoid acoustic feedback.

For phones or tablets on the same local network, use the HTTPS LAN URL printed by Vite. Mobile camera/microphone permissions require HTTPS. The dev certificate is self-signed, so the browser may ask you to accept a warning before the app can start devices.
