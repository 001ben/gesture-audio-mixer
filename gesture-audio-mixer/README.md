# Gesture Audio Mixer

A browser-based performance tool that maps webcam hand motion to live microphone audio effects.

## Controls

- Pitch mode is the default: put one hand near the center ring for normal voice.
- Raise your hand to pitch up, or lower it to pitch down.
- Advanced mode splits the camera into two lanes.
- The left lane controls pitch, brightness, and vibrato.
- The right lane controls robot tone, echo, grit, and tremolo.
- Voice mix controls how much shifted voice is applied.
- The app warns before starting if live monitoring is on, because microphone monitoring through speakers can create loud feedback.

## Run

```bash
npm install
npm run dev
```

Open `http://localhost:5175` for local desktop testing, then press **Start camera + mic**. Use headphones to avoid acoustic feedback.

For phones or tablets on the same local network, use the HTTPS LAN URL printed by Vite. Mobile camera/microphone permissions require HTTPS. The dev certificate is self-signed, so the browser may ask you to accept a warning before the app can start devices.
