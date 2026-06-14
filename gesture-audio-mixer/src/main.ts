import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
  type NormalizedLandmark
} from "@mediapipe/tasks-vision";
import { SoundTouchNode, type ProcessorMetrics } from "@soundtouchjs/audio-worklet";
import soundTouchProcessorUrl from "@soundtouchjs/audio-worklet/processor?url";
import "./styles.css";

const MODEL_VERSION = "0.10.35";
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MODEL_VERSION}/wasm`;
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const CENTER_DEAD_ZONE = 0.04;
const TONE_CENTER_X = 0.25;
const TEXTURE_CENTER_X = 0.75;
const LANE_EFFECT_HALF_WIDTH = 0.14;
const LANE_EFFECT_HALF_HEIGHT = 0.22;

type GestureControls = {
  pitchSemitones: number;
  pitchWet: number;
  brightnessHz: number;
  robotAmount: number;
  robotRateHz: number;
  echoAmount: number;
  distortion: number;
  vibrato: number;
  tremolo: number;
  activity: number;
};

type HandRole = {
  name: "tone" | "texture";
  x: number;
  y: number;
  pinch: number;
  speed: number;
  landmarks: NormalizedLandmark[];
};

type MeterFrame = {
  waveform: Uint8Array<ArrayBuffer>;
  loudnessSamples: Float32Array<ArrayBuffer>;
};

type LoudnessPoint = {
  at: number;
  meanSquare: number;
};

const LOUDNESS_WINDOW_MS = 10000;

const neutralControls: GestureControls = {
  pitchSemitones: 0,
  pitchWet: 0,
  brightnessHz: 12000,
  robotAmount: 0,
  robotRateHz: 42,
  echoAmount: 0,
  distortion: 0,
  vibrato: 0,
  tremolo: 0,
  activity: 0
};

const app = document.querySelector<HTMLDivElement>("#app");

if (!app) {
  throw new Error("Missing app root");
}

app.innerHTML = `
  <main class="shell">
    <section class="stage-panel" aria-label="Camera tracking surface">
      <div class="topbar">
        <div>
          <p class="eyebrow">Gesture Audio Mixer</p>
          <h1>Move your hands. Bend your voice.</h1>
        </div>
        <button class="primary-button" id="startButton" type="button">
          <span class="button-icon" aria-hidden="true">▶</span>
          <span>Start camera + mic</span>
        </button>
      </div>

      <div class="stage">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="overlay"></canvas>
        <div class="stage-empty" id="stageEmpty">
          <strong>Ready for input</strong>
          <span>Use headphones, then allow camera and microphone when prompted.</span>
        </div>
        <div class="center-guide" aria-hidden="true">
          <span class="pitch-range"></span>
          <span class="pitch-center-line"></span>
          <span class="pitch-guide-ring"></span>
        </div>
        <div class="stage-grid" aria-hidden="true"></div>
      </div>

      <div class="status-row">
        <span id="trackingStatus">Hand tracker idle</span>
        <span id="audioStatus">Audio engine idle</span>
        <span id="fpsStatus">0 fps</span>
      </div>
    </section>

    <aside class="mixer-panel" aria-label="Gesture audio controls">
      <div class="meter-wrap">
        <canvas id="meter" width="560" height="130" aria-label="Audio level visualizer"></canvas>
        <div class="meter-stats" aria-label="Output loudness estimate">
          <div>
            <span>Now LUFS</span>
            <strong id="loudnessReadout">-inf LUFS</strong>
          </div>
          <div>
            <span>10s avg</span>
            <strong id="averageLufsReadout">-inf LUFS</strong>
          </div>
          <div>
            <span>10s max</span>
            <strong id="maxLufsReadout">-inf LUFS</strong>
          </div>
          <div>
            <span>Peak now</span>
            <strong id="peakReadout">-inf dB</strong>
          </div>
        </div>
      </div>

      <section class="control-cluster" aria-label="Tone hand mapping">
        <div class="cluster-heading">
          <span class="role-dot tone-dot"></span>
          <div>
            <h2>Pitch hand</h2>
            <p>Raise or lower one hand</p>
          </div>
        </div>
        <div class="readout-grid">
          <div class="readout">
            <span>Pitch</span>
            <strong id="pitchReadout">0 st</strong>
          </div>
          <div class="readout">
            <span>Voice mix</span>
            <strong id="shiftReadout">0%</strong>
          </div>
          <div class="readout">
            <span>Mode</span>
            <strong id="brightnessReadout">Pitch only</strong>
          </div>
        </div>
      </section>

      <section class="control-cluster" aria-label="Texture hand mapping" hidden>
        <div class="cluster-heading">
          <span class="role-dot texture-dot"></span>
          <div>
            <h2>Texture hand</h2>
            <p>Right side of frame</p>
          </div>
        </div>
        <div class="readout-grid">
          <div class="readout">
            <span>Robot</span>
            <strong id="robotReadout">0%</strong>
          </div>
          <div class="readout">
            <span>Echo</span>
            <strong id="echoReadout">0%</strong>
          </div>
          <div class="readout">
            <span>Grit</span>
            <strong id="distortionReadout">0%</strong>
          </div>
        </div>
      </section>

      <section class="control-cluster compact" aria-label="Output controls">
        <div class="mode-control pitch-source-control" aria-label="Pitch diagnostic source">
          <button class="mode-option is-active" id="livePitchButton" type="button" aria-pressed="true">Live hand</button>
          <button class="mode-option" id="fixedPitchButton" type="button" aria-pressed="false">Fixed +7</button>
          <button class="mode-option" id="dryMicButton" type="button" aria-pressed="false">Dry mic</button>
        </div>
        <label class="slider-line" for="pitchMix">
          <span>Voice mix</span>
          <input id="pitchMix" type="range" min="0" max="1" step="0.01" value="1" />
        </label>
        <label class="slider-line" for="inputGain">
          <span>Input gain</span>
          <input id="inputGain" type="range" min="0.5" max="4" step="0.01" value="1.5" />
        </label>
        <div class="toggle-line">
          <label for="monitorToggle">Monitor output</label>
          <input id="monitorToggle" type="checkbox" checked />
        </div>
        <label class="slider-line" for="outputGain">
          <span>Output</span>
          <input id="outputGain" type="range" min="0" max="1" step="0.01" value="0.72" />
        </label>
      </section>

      <p class="hint">
        Keep your hand on the center line for normal voice. Raise it to pitch up, lower it to pitch down. Headphones strongly recommended.
      </p>
    </aside>
  </main>
`;

const startButton = getElement<HTMLButtonElement>("startButton");
const video = getElement<HTMLVideoElement>("video");
const overlay = getElement<HTMLCanvasElement>("overlay");
const meter = getElement<HTMLCanvasElement>("meter");
const stageEmpty = getElement<HTMLDivElement>("stageEmpty");
const trackingStatus = getElement<HTMLSpanElement>("trackingStatus");
const audioStatus = getElement<HTMLSpanElement>("audioStatus");
const fpsStatus = getElement<HTMLSpanElement>("fpsStatus");
const monitorToggle = getElement<HTMLInputElement>("monitorToggle");
const outputGain = getElement<HTMLInputElement>("outputGain");
const pitchMix = getElement<HTMLInputElement>("pitchMix");
const inputGain = getElement<HTMLInputElement>("inputGain");
const loudnessReadout = getElement<HTMLElement>("loudnessReadout");
const averageLufsReadout = getElement<HTMLElement>("averageLufsReadout");
const maxLufsReadout = getElement<HTMLElement>("maxLufsReadout");
const peakReadout = getElement<HTMLElement>("peakReadout");
const livePitchButton = getElement<HTMLButtonElement>("livePitchButton");
const fixedPitchButton = getElement<HTMLButtonElement>("fixedPitchButton");
const dryMicButton = getElement<HTMLButtonElement>("dryMicButton");

const readouts = {
  pitch: getElement<HTMLElement>("pitchReadout"),
  shift: getElement<HTMLElement>("shiftReadout"),
  brightness: getElement<HTMLElement>("brightnessReadout"),
  robot: getElement<HTMLElement>("robotReadout"),
  echo: getElement<HTMLElement>("echoReadout"),
  distortion: getElement<HTMLElement>("distortionReadout")
};

const overlayContext = getCanvasContext(overlay);
const meterContext = getCanvasContext(meter);

let landmarker: HandLandmarker | undefined;
let audioEngine: GestureAudioEngine | undefined;
let mediaStream: MediaStream | undefined;
let animationStarted = false;
let isLive = false;
let pitchSource: "live" | "fixed" | "dry" = "live";
let pitchMixAmount = 1;
let previousPalms = new Map<number, { x: number; y: number; at: number }>();
let smoothedControls = { ...neutralControls };
let smoothedLoudnessDb = -70;
let smoothedPeakDb = -70;
let loudnessHistory: LoudnessPoint[] = [];
let lastFrameAt = performance.now();
let fpsAverage = 0;

startButton.addEventListener("click", () => {
  void toggleExperience();
});

monitorToggle.addEventListener("change", () => {
  audioEngine?.setMonitoring(monitorToggle.checked);
});

outputGain.addEventListener("input", () => {
  audioEngine?.setOutputGain(Number(outputGain.value));
});

inputGain.addEventListener("input", () => {
  audioEngine?.setInputGain(Number(inputGain.value));
  resetMeterState();
});

pitchMix.addEventListener("input", () => {
  pitchMixAmount = Number(pitchMix.value);
  resetMeterState();
  updateDiagnosticReadouts();
});

livePitchButton.addEventListener("click", () => {
  setPitchSource("live");
});

fixedPitchButton.addEventListener("click", () => {
  setPitchSource("fixed");
});

dryMicButton.addEventListener("click", () => {
  setPitchSource("dry");
});

startButton.innerHTML = `<span class="button-icon" aria-hidden="true">&gt;</span><span>Start camera + mic</span>`;

function setPitchSource(source: "live" | "fixed" | "dry") {
  pitchSource = source;
  const buttons = [
    [livePitchButton, source === "live"],
    [fixedPitchButton, source === "fixed"],
    [dryMicButton, source === "dry"]
  ] as const;

  for (const [button, active] of buttons) {
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }

  resetMeterState();
  updateDiagnosticReadouts();
}

function resetMeterState() {
  smoothedLoudnessDb = -70;
  smoothedPeakDb = -70;
  loudnessHistory = [];
  updateLoudnessReadouts(-70, -70, -70, -70);
}

function updateDiagnosticReadouts() {
  readouts.pitch.textContent = pitchSource === "fixed" ? "7 st" : "0 st";
  readouts.shift.textContent = `${Math.round((pitchSource === "dry" ? 0 : pitchMixAmount) * 100)}%`;
  readouts.brightness.textContent = pitchSourceLabel();
}

async function toggleExperience() {
  if (isLive) {
    await stopExperience();
    return;
  }

  await startExperience();
}

async function startExperience() {
  startButton.disabled = true;
  startButton.innerHTML = `<span class="button-icon" aria-hidden="true">...</span><span>Starting</span>`;
  trackingStatus.textContent = "Loading hand tracker";

  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera and microphone need HTTPS on mobile. Open the https LAN URL and accept the dev certificate warning.");
    }

    landmarker ??= await createHandLandmarker();
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user"
      },
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    });

    video.srcObject = mediaStream;
    await video.play();

    const nextAudioEngine = new GestureAudioEngine();
    await nextAudioEngine.start(mediaStream);
    audioEngine = nextAudioEngine;
    audioEngine.setOutputGain(Number(outputGain.value));
    audioEngine.setInputGain(Number(inputGain.value));
    audioEngine.setMonitoring(monitorToggle.checked);

    stageEmpty.hidden = true;
    audioStatus.textContent = "Audio engine live";
    trackingStatus.textContent = "Tracking hands";
    startButton.innerHTML = `<span class="button-icon" aria-hidden="true">✓</span><span>Live</span>`;

    if (!animationStarted) {
      animationStarted = true;
      requestAnimationFrame(tick);
      requestAnimationFrame(drawMeter);
    }

    isLive = true;
    startButton.disabled = false;
    startButton.innerHTML = `<span class="button-icon" aria-hidden="true">x</span><span>Stop feed</span>`;
  } catch (error) {
    console.error(error);
    await releaseLiveResources();
    startButton.disabled = false;
    startButton.innerHTML = `<span class="button-icon" aria-hidden="true">▶</span><span>Start camera + mic</span>`;
    trackingStatus.textContent = "Could not start devices";
    audioStatus.textContent = error instanceof Error ? error.message : "Unknown error";
    startButton.innerHTML = `<span class="button-icon" aria-hidden="true">&gt;</span><span>Start camera + mic</span>`;
  }
}

async function stopExperience() {
  startButton.disabled = true;
  startButton.innerHTML = `<span class="button-icon" aria-hidden="true">...</span><span>Stopping</span>`;
  await releaseLiveResources();
  resetLiveUi();
}

async function releaseLiveResources() {
  isLive = false;
  await audioEngine?.dispose();
  audioEngine = undefined;

  mediaStream?.getTracks().forEach((track) => {
    track.stop();
  });
  mediaStream = undefined;

  video.pause();
  video.srcObject = null;
  previousPalms.clear();
  smoothedControls = { ...neutralControls };
  resetMeterState();
  fpsAverage = 0;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
}

function resetLiveUi() {
  stageEmpty.hidden = false;
  updateReadouts(neutralControls, 0);
  trackingStatus.textContent = "Hand tracker idle";
  audioStatus.textContent = "Audio engine idle";
  fpsStatus.textContent = "0 fps";
  startButton.disabled = false;
  startButton.innerHTML = `<span class="button-icon" aria-hidden="true">&gt;</span><span>Start camera + mic</span>`;
}

async function createHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_URL,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
}

function tick(now: number) {
  requestAnimationFrame(tick);

  resizeOverlay();
  if (!landmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  const result = landmarker.detectForVideo(video, now);
  const roles = mapHandsToRoles(result, now);
  const nextControls = controlsFromHands(roles);
  smoothedControls = smoothControls(smoothedControls, nextControls, 0.22);

  audioEngine?.applyControls(smoothedControls);
  drawHands(result, roles);
  updateReadouts(smoothedControls, roles.length);
  updateFps(now);
}

function mapHandsToRoles(result: HandLandmarkerResult, now: number): HandRole[] {
  const hands = result.landmarks.map((landmarks, index) => {
    const palm = landmarks[9];
    const mirroredX = 1 - palm.x;
    const previous = previousPalms.get(index);
    const dt = previous ? Math.max(16, now - previous.at) : 16;
    const distance = previous ? Math.hypot(mirroredX - previous.x, palm.y - previous.y) : 0;
    const speed = clamp((distance / dt) * 120, 0, 1);
    previousPalms.set(index, { x: mirroredX, y: palm.y, at: now });
    return {
      x: mirroredX,
      y: palm.y,
      pinch: pinchAmount(landmarks),
      speed,
      landmarks
    };
  });

  if (hands.length === 0) {
    previousPalms.clear();
    return [];
  }

  const sorted = [...hands].sort((a, b) => a.x - b.x);
  if (sorted.length === 1) {
    const only = sorted[0];
    return [{ ...only, name: only.x < 0.5 ? "tone" : "texture" }];
  }

  return [
    { ...sorted[0], name: "tone" },
    { ...sorted[sorted.length - 1], name: "texture" }
  ];
}

function controlsFromHands(roles: HandRole[]): GestureControls {
  const controls = { ...neutralControls };
  controls.activity = roles.length > 0 ? 1 : 0;

  if (pitchSource === "dry") {
    controls.pitchWet = 0;
    return controls;
  }

  if (pitchSource === "fixed") {
    controls.pitchSemitones = 7;
    controls.pitchWet = pitchMixAmount;
    return controls;
  }

  const pitchHand = roles[0];
  if (!pitchHand) {
    return controls;
  }

  const vertical = centeredAxis(1 - pitchHand.y, 0.5, LANE_EFFECT_HALF_HEIGHT);
  controls.pitchSemitones = vertical * 12;
  controls.pitchWet = pitchMixAmount;
  controls.brightnessHz = neutralControls.brightnessHz;
  controls.robotAmount = 0;
  controls.echoAmount = 0;
  controls.distortion = 0;
  controls.vibrato = 0;
  controls.tremolo = 0;
  return controls;
}

function pinchAmount(landmarks: NormalizedLandmark[]) {
  const thumb = landmarks[4];
  const index = landmarks[8];
  const distance = Math.hypot(thumb.x - index.x, thumb.y - index.y);
  return clamp(1 - (distance - 0.025) / 0.13, 0, 1);
}

function centeredAxis(value: number, center: number, halfRange: number) {
  const distance = value - center;
  const magnitude = Math.abs(distance);

  if (magnitude <= CENTER_DEAD_ZONE) {
    return 0;
  }

  const scaled = (magnitude - CENTER_DEAD_ZONE) / (halfRange - CENTER_DEAD_ZONE);
  return Math.sign(distance) * clamp(scaled, 0, 1);
}

function smoothControls(previous: GestureControls, next: GestureControls, amount: number): GestureControls {
  return {
    pitchSemitones: lerp(previous.pitchSemitones, next.pitchSemitones, 0.1),
    pitchWet: lerp(previous.pitchWet, next.pitchWet, amount),
    brightnessHz: lerp(previous.brightnessHz, next.brightnessHz, amount),
    robotAmount: lerp(previous.robotAmount, next.robotAmount, amount),
    robotRateHz: lerp(previous.robotRateHz, next.robotRateHz, amount),
    echoAmount: lerp(previous.echoAmount, next.echoAmount, amount),
    distortion: lerp(previous.distortion, next.distortion, amount),
    vibrato: lerp(previous.vibrato, next.vibrato, amount),
    tremolo: lerp(previous.tremolo, next.tremolo, amount),
    activity: lerp(previous.activity, next.activity, amount)
  };
}

function drawHands(result: HandLandmarkerResult, roles: HandRole[]) {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  overlayContext.lineWidth = Math.max(3, overlay.width / 300);
  overlayContext.lineCap = "round";
  overlayContext.lineJoin = "round";

  for (const role of roles) {
    const color = "#f2c94c";
    overlayContext.strokeStyle = color;
    overlayContext.fillStyle = color;
    drawHandSkeleton(role.landmarks);
  }

  if (result.landmarks.length === 0) {
    overlayContext.fillStyle = "rgba(255, 255, 255, 0.76)";
    overlayContext.font = "600 18px Inter, system-ui, sans-serif";
    overlayContext.fillText("Raise one or two hands into frame", 28, 42);
  }
}

function drawHandSkeleton(landmarks: NormalizedLandmark[]) {
  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17]
  ];

  for (const [from, to] of connections) {
    const start = toCanvas(landmarks[from]);
    const end = toCanvas(landmarks[to]);
    overlayContext.beginPath();
    overlayContext.moveTo(start.x, start.y);
    overlayContext.lineTo(end.x, end.y);
    overlayContext.stroke();
  }

  for (const landmark of landmarks) {
    const point = toCanvas(landmark);
    overlayContext.beginPath();
    overlayContext.arc(point.x, point.y, Math.max(4, overlay.width / 230), 0, Math.PI * 2);
    overlayContext.fill();
  }
}

function drawPinchHalo(role: HandRole) {
  const thumb = toCanvas(role.landmarks[4]);
  const index = toCanvas(role.landmarks[8]);
  const x = (thumb.x + index.x) / 2;
  const y = (thumb.y + index.y) / 2;
  const radius = lerp(16, 54, role.pinch);

  overlayContext.save();
  overlayContext.globalAlpha = 0.18 + role.pinch * 0.35;
  overlayContext.beginPath();
  overlayContext.arc(x, y, radius, 0, Math.PI * 2);
  overlayContext.fill();
  overlayContext.restore();
}

function drawMeter() {
  requestAnimationFrame(drawMeter);
  const width = meter.width;
  const height = meter.height;
  meterContext.clearRect(0, 0, width, height);

  const data = audioEngine?.getMeterFrame();
  updatePitchProcessorStatus(audioEngine?.getPitchMetrics());
  const gradient = meterContext.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "#00d0a2");
  gradient.addColorStop(0.5, "#f2c94c");
  gradient.addColorStop(1, "#ff7a59");

  meterContext.fillStyle = "rgba(255, 255, 255, 0.05)";
  meterContext.fillRect(0, 0, width, height);

  meterContext.strokeStyle = gradient;
  meterContext.lineWidth = 3;
  meterContext.beginPath();

  if (data) {
    const loudness = estimateLufs(data);
    const now = performance.now();
    smoothedLoudnessDb = smoothDb(smoothedLoudnessDb, loudness.lufs, 0.16);
    smoothedPeakDb = smoothDb(smoothedPeakDb, loudness.peakDb, 0.28);
    loudnessHistory = updateLoudnessHistory(loudnessHistory, loudness.meanSquare, now);
    const averageLufs = averageLufsFromHistory(loudnessHistory);
    const maxLufs = maxLufsFromHistory(loudnessHistory);
    updateLoudnessReadouts(smoothedLoudnessDb, averageLufs, maxLufs, loudness.peakDb);

    for (let i = 0; i < data.waveform.length; i += 1) {
      const x = (i / (data.waveform.length - 1)) * width;
      const y = ((255 - data.waveform[i]) / 255) * height;
      if (i === 0) {
        meterContext.moveTo(x, y);
      } else {
        meterContext.lineTo(x, y);
      }
    }
  } else {
    meterContext.moveTo(0, height / 2);
    meterContext.lineTo(width, height / 2);
  }

  meterContext.stroke();
  drawLoudnessStrip(smoothedLoudnessDb, smoothedPeakDb);
}

function estimateLufs(data: MeterFrame) {
  let sumSquares = 0;
  let peak = 0;

  for (const sample of data.loudnessSamples) {
    sumSquares += sample * sample;
  }

  for (const sample of data.waveform) {
    const centered = (sample - 128) / 128;
    peak = Math.max(peak, Math.abs(centered));
  }

  const meanSquare = sumSquares / data.loudnessSamples.length;
  return {
    lufs: meanSquareToLufs(meanSquare),
    meanSquare,
    peakDb: amplitudeToDb(peak)
  };
}

function drawLoudnessStrip(rmsDb: number, peakDb: number) {
  const width = meter.width;
  const height = meter.height;
  const stripHeight = 10;
  const y = height - stripHeight;
  const rmsWidth = dbToMeterWidth(rmsDb, width);
  const peakX = dbToMeterWidth(peakDb, width);
  const gradient = meterContext.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "#00d0a2");
  gradient.addColorStop(0.72, "#f2c94c");
  gradient.addColorStop(1, "#ff4f64");

  meterContext.save();
  meterContext.fillStyle = "rgba(255, 255, 255, 0.08)";
  meterContext.fillRect(0, y, width, stripHeight);
  meterContext.fillStyle = gradient;
  meterContext.fillRect(0, y, rmsWidth, stripHeight);
  meterContext.strokeStyle = "rgba(255, 255, 255, 0.92)";
  meterContext.lineWidth = 2;
  meterContext.beginPath();
  meterContext.moveTo(peakX, y - 3);
  meterContext.lineTo(peakX, height);
  meterContext.stroke();
  meterContext.restore();
}

function updateLoudnessHistory(history: LoudnessPoint[], meanSquare: number, now: number) {
  const next = [...history, { at: now, meanSquare }];
  return next.filter((point) => now - point.at <= LOUDNESS_WINDOW_MS);
}

function averageLufsFromHistory(history: LoudnessPoint[]) {
  if (history.length === 0) {
    return -70;
  }

  const meanSquare = history.reduce((sum, point) => sum + point.meanSquare, 0) / history.length;
  return meanSquareToLufs(meanSquare);
}

function maxLufsFromHistory(history: LoudnessPoint[]) {
  if (history.length === 0) {
    return -70;
  }

  const maxMeanSquare = Math.max(...history.map((point) => point.meanSquare));
  return meanSquareToLufs(maxMeanSquare);
}

function updateLoudnessReadouts(momentaryLufs: number, averageLufs: number, maxLufs: number, peakDb: number) {
  loudnessReadout.textContent = formatLufs(momentaryLufs);
  averageLufsReadout.textContent = formatLufs(averageLufs);
  maxLufsReadout.textContent = formatLufs(maxLufs);
  peakReadout.textContent = formatDb(peakDb);
}

function updatePitchProcessorStatus(metrics: ProcessorMetrics | null | undefined) {
  if (!isLive) {
    return;
  }

  if (!metrics) {
    audioStatus.textContent = "Audio worklet starting";
    return;
  }

  if (metrics.underrunCount > 0) {
    audioStatus.textContent = `Audio worklet underruns ${metrics.underrunCount}`;
    return;
  }

  audioStatus.textContent = "Audio worklet stable";
}

function smoothDb(previous: number, next: number, amount: number) {
  return lerp(previous, next, amount);
}

function amplitudeToDb(value: number) {
  return 20 * Math.log10(Math.max(value, 0.00001));
}

function meanSquareToLufs(meanSquare: number) {
  return -0.691 + 10 * Math.log10(Math.max(meanSquare, 0.0000000001));
}

function dbToMeterWidth(db: number, width: number) {
  return clamp((db + 70) / 70, 0, 1) * width;
}

function formatDb(db: number) {
  if (db <= -59.5) {
    return "-inf dB";
  }

  return `${Math.round(db)} dB`;
}

function formatLufs(lufs: number) {
  if (lufs <= -69.5) {
    return "-inf LUFS";
  }

  return `${Math.round(lufs)} LUFS`;
}

function updateReadouts(controls: GestureControls, handCount: number) {
  readouts.pitch.textContent = `${Math.round(controls.pitchSemitones)} st`;
  readouts.shift.textContent = `${Math.round(controls.pitchWet * 100)}%`;
  readouts.brightness.textContent = pitchSourceLabel();
  readouts.robot.textContent = `${Math.round(controls.robotAmount * 100)}%`;
  readouts.echo.textContent = `${Math.round(controls.echoAmount * 100)}%`;
  readouts.distortion.textContent = `${Math.round(controls.distortion * 100)}%`;
  if (pitchSource === "fixed") {
    trackingStatus.textContent = "Fixed pitch test";
  } else if (pitchSource === "dry") {
    trackingStatus.textContent = "Dry mic test";
  } else {
    trackingStatus.textContent = handCount === 0 ? "Searching for hands" : `${handCount} hand${handCount === 1 ? "" : "s"} tracked`;
  }
}

function pitchSourceLabel() {
  if (pitchSource === "dry") {
    return "Dry mic";
  }

  if (pitchSource === "fixed") {
    return "Fixed +7";
  }

  return "Live hand";
}

function updateFps(now: number) {
  const dt = Math.max(1, now - lastFrameAt);
  const instant = 1000 / dt;
  fpsAverage = fpsAverage === 0 ? instant : lerp(fpsAverage, instant, 0.08);
  fpsStatus.textContent = `${Math.round(fpsAverage)} fps`;
  lastFrameAt = now;
}

function resizeOverlay() {
  const width = video.videoWidth || overlay.clientWidth;
  const height = video.videoHeight || overlay.clientHeight;
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
}

function toCanvas(landmark: NormalizedLandmark) {
  return {
    x: (1 - landmark.x) * overlay.width,
    y: landmark.y * overlay.height
  };
}

class GestureAudioEngine {
  private context!: AudioContext;
  private source!: MediaStreamAudioSourceNode;
  private inputGain!: GainNode;
  private pitchShifter!: SoundTouchPitchShifter;
  private directPitchGain!: GainNode;
  private shiftedPitchGain!: GainNode;
  private toneFilter!: BiquadFilterNode;
  private robotDryGain!: GainNode;
  private ringGain!: GainNode;
  private robotWetGain!: GainNode;
  private robotOsc!: OscillatorNode;
  private voiceBus!: GainNode;
  private cleanGain!: GainNode;
  private distortionNode!: WaveShaperNode;
  private distortionGain!: GainNode;
  private distortionCompensation!: GainNode;
  private distortionBus!: GainNode;
  private delay!: DelayNode;
  private feedback!: GainNode;
  private echoGain!: GainNode;
  private master!: GainNode;
  private outputVolume!: GainNode;
  private monitorGate!: GainNode;
  private analyser!: AnalyserNode;
  private loudnessHighpass!: BiquadFilterNode;
  private loudnessShelf!: BiquadFilterNode;
  private loudnessAnalyser!: AnalyserNode;
  private waveform!: Uint8Array<ArrayBuffer>;
  private loudnessSamples!: Float32Array<ArrayBuffer>;
  private vibratoOsc!: OscillatorNode;
  private vibratoGain!: GainNode;
  private tremoloOsc!: OscillatorNode;
  private tremoloDepth!: GainNode;
  private tremoloBase!: ConstantSourceNode;
  private disposed = false;

  async start(stream: MediaStream) {
    this.disposed = false;
    this.context = new AudioContext({ latencyHint: "interactive" });
    this.source = this.context.createMediaStreamSource(stream);
    this.inputGain = this.context.createGain();
    this.inputGain.gain.value = 0.9;

    await SoundTouchNode.register(this.context, soundTouchProcessorUrl);
    this.pitchShifter = new SoundTouchPitchShifter(this.context);
    this.directPitchGain = this.context.createGain();
    this.shiftedPitchGain = this.context.createGain();
    this.toneFilter = this.context.createBiquadFilter();
    this.toneFilter.type = "lowpass";
    this.toneFilter.frequency.value = neutralControls.brightnessHz;
    this.toneFilter.Q.value = 0.7;

    this.robotDryGain = this.context.createGain();
    this.ringGain = this.context.createGain();
    this.robotWetGain = this.context.createGain();
    this.robotOsc = this.context.createOscillator();
    this.robotOsc.type = "square";
    this.robotOsc.frequency.value = neutralControls.robotRateHz;
    this.voiceBus = this.context.createGain();

    this.cleanGain = this.context.createGain();
    this.distortionNode = this.context.createWaveShaper();
    this.distortionGain = this.context.createGain();
    this.distortionCompensation = this.context.createGain();
    this.distortionBus = this.context.createGain();
    this.delay = this.context.createDelay(1.4);
    this.feedback = this.context.createGain();
    this.echoGain = this.context.createGain();
    this.master = this.context.createGain();
    this.outputVolume = this.context.createGain();
    this.monitorGate = this.context.createGain();
    this.analyser = this.context.createAnalyser();
    this.loudnessHighpass = this.context.createBiquadFilter();
    this.loudnessShelf = this.context.createBiquadFilter();
    this.loudnessAnalyser = this.context.createAnalyser();
    this.analyser.fftSize = 1024;
    this.loudnessAnalyser.fftSize = 2048;
    this.loudnessHighpass.type = "highpass";
    this.loudnessHighpass.frequency.value = 60;
    this.loudnessHighpass.Q.value = 0.5;
    this.loudnessShelf.type = "highshelf";
    this.loudnessShelf.frequency.value = 1500;
    this.loudnessShelf.gain.value = 4;
    this.waveform = new Uint8Array(this.analyser.frequencyBinCount);
    this.loudnessSamples = new Float32Array(this.loudnessAnalyser.fftSize);

    this.vibratoOsc = this.context.createOscillator();
    this.vibratoOsc.type = "sine";
    this.vibratoOsc.frequency.value = 5.4;
    this.vibratoGain = this.context.createGain();
    this.vibratoGain.gain.value = 0;

    this.tremoloOsc = this.context.createOscillator();
    this.tremoloOsc.type = "sine";
    this.tremoloOsc.frequency.value = 7;
    this.tremoloDepth = this.context.createGain();
    this.tremoloDepth.gain.value = 0;
    this.tremoloBase = this.context.createConstantSource();
    this.tremoloBase.offset.value = 1;
    this.master.gain.value = 0;

    this.source.connect(this.inputGain);
    this.inputGain.connect(this.directPitchGain);
    this.inputGain.connect(this.pitchShifter.input);
    this.pitchShifter.output.connect(this.shiftedPitchGain);
    this.directPitchGain.connect(this.toneFilter);
    this.shiftedPitchGain.connect(this.toneFilter);

    this.vibratoOsc.connect(this.vibratoGain);
    this.vibratoGain.connect(this.toneFilter.detune);

    this.toneFilter.connect(this.robotDryGain);
    this.toneFilter.connect(this.ringGain);
    this.robotOsc.connect(this.ringGain.gain);
    this.robotDryGain.connect(this.voiceBus);
    this.ringGain.connect(this.robotWetGain);
    this.robotWetGain.connect(this.voiceBus);

    this.voiceBus.connect(this.cleanGain);
    this.voiceBus.connect(this.distortionNode);
    this.cleanGain.connect(this.distortionBus);
    this.distortionNode.connect(this.distortionGain);
    this.distortionGain.connect(this.distortionBus);
    this.distortionBus.connect(this.distortionCompensation);

    this.distortionCompensation.connect(this.delay);
    this.delay.connect(this.feedback);
    this.feedback.connect(this.delay);
    this.delay.connect(this.echoGain);

    this.tremoloBase.connect(this.master.gain);
    this.tremoloOsc.connect(this.tremoloDepth);
    this.tremoloDepth.connect(this.master.gain);

    this.distortionCompensation.connect(this.master);
    this.echoGain.connect(this.master);
    this.master.connect(this.analyser);
    this.master.connect(this.loudnessHighpass);
    this.master.connect(this.outputVolume);
    this.loudnessHighpass.connect(this.loudnessShelf);
    this.loudnessShelf.connect(this.loudnessAnalyser);
    this.outputVolume.connect(this.monitorGate);
    this.monitorGate.connect(this.context.destination);

    this.robotOsc.start();
    this.vibratoOsc.start();
    this.tremoloOsc.start();
    this.tremoloBase.start();
    this.applyControls(neutralControls);
    await this.context.resume();
  }

  applyControls(controls: GestureControls) {
    if (this.disposed || this.context.state === "closed") {
      return;
    }

    const now = this.context.currentTime;
    const pitchWet = clamp(controls.pitchWet, 0, 1);
    const robot = clamp(controls.robotAmount, 0, 1);
    const distortion = clamp(controls.distortion, 0, 1);
    const echo = clamp(controls.echoAmount, 0, 1);
    const effectivePitchWet = pitchWet;
    const dryGain = Math.cos(effectivePitchWet * Math.PI * 0.5);
    const shiftedGain = Math.sin(effectivePitchWet * Math.PI * 0.5);

    this.pitchShifter.setSemitones(controls.pitchSemitones);
    setParam(this.directPitchGain.gain, dryGain, now);
    setParam(this.shiftedPitchGain.gain, shiftedGain, now);
    setParam(this.toneFilter.frequency, controls.brightnessHz, now);
    setParam(this.vibratoGain.gain, controls.vibrato * 26, now);

    setParam(this.robotDryGain.gain, 1 - robot * 0.65, now);
    setParam(this.robotWetGain.gain, robot * 0.92, now);
    setParam(this.robotOsc.frequency, controls.robotRateHz, now);

    this.distortionNode.curve = makeDistortionCurve(distortion);
    const distortionWet = distortion * 0.9;
    setParam(this.cleanGain.gain, Math.cos(distortionWet * Math.PI * 0.5), now);
    setParam(this.distortionGain.gain, Math.sin(distortionWet * Math.PI * 0.5) * 0.86, now);
    setParam(this.distortionCompensation.gain, lerp(1, 0.9, distortion), now);

    setParam(this.delay.delayTime, lerp(0.08, 0.42, echo), now);
    setParam(this.feedback.gain, echo * 0.58, now);
    setParam(this.echoGain.gain, echo * 0.7, now);

    setParam(this.tremoloDepth.gain, controls.tremolo * 0.35, now);
  }

  setMonitoring(enabled: boolean) {
    if (!this.context || this.disposed || this.context.state === "closed") {
      return;
    }
    setParam(this.monitorGate.gain, enabled ? 1 : 0, this.context.currentTime);
  }

  setOutputGain(value: number) {
    if (!this.context || this.disposed || this.context.state === "closed") {
      return;
    }
    setParam(this.outputVolume.gain, clamp(value, 0, 1), this.context.currentTime);
  }

  setInputGain(value: number) {
    if (!this.context || this.disposed || this.context.state === "closed") {
      return;
    }
    setParam(this.inputGain.gain, clamp(value, 0.5, 4), this.context.currentTime);
  }

  getPitchMetrics() {
    return this.pitchShifter.metrics;
  }

  getMeterFrame(): MeterFrame | undefined {
    if (!this.analyser || this.disposed || this.context.state === "closed") {
      return undefined;
    }
    this.analyser.getByteTimeDomainData(this.waveform);
    this.loudnessAnalyser.getFloatTimeDomainData(this.loudnessSamples);
    return {
      waveform: this.waveform,
      loudnessSamples: this.loudnessSamples
    };
  }

  async dispose() {
    if (this.disposed) {
      return;
    }

    this.disposed = true;
    this.pitchShifter.dispose();
    stopAudioSource(this.robotOsc);
    stopAudioSource(this.vibratoOsc);
    stopAudioSource(this.tremoloOsc);
    stopAudioSource(this.tremoloBase);

    if (this.context.state !== "closed") {
      await this.context.close();
    }
  }
}

class SoundTouchPitchShifter {
  readonly input: AudioNode;
  readonly output: AudioNode;

  private context: AudioContext;
  private node: SoundTouchNode;

  constructor(context: AudioContext) {
    this.context = context;
    this.node = new SoundTouchNode({
      context,
      outputChannelCount: 2,
      sampleBufferType: "circular",
      interpolationStrategy: "lanczos"
    });
    this.input = this.node;
    this.output = this.node;

    this.node.pitch.value = 1;
    this.node.pitchSemitones.value = 0;
    this.node.playbackRate.value = 1;
    this.node.setStretchParameters({
      sequenceMs: 55,
      seekWindowMs: 18,
      overlapMs: 12,
      quickSeek: true
    });
  }

  get metrics(): ProcessorMetrics | null {
    return this.node.metrics;
  }

  setSemitones(semitones: number) {
    const now = this.context.currentTime;
    const target = clamp(semitones, -12, 12);
    this.node.pitchSemitones.cancelScheduledValues(now);
    this.node.pitchSemitones.setTargetAtTime(target, now, 0.055);
  }

  dispose() {
    this.node.disconnect();
  }
}

function stopAudioSource(source: AudioScheduledSourceNode) {
  try {
    source.stop();
  } catch (_error) {
    // Source may already be stopped when the audio context is closing.
  }
}

function makeDistortionCurve(amount: number) {
  const samples = 1024;
  const curve = new Float32Array(samples);
  const drive = 1 + amount * 22;
  const trim = lerp(1, 0.9, amount);

  for (let i = 0; i < samples; i += 1) {
    const x = (i * 2) / samples - 1;
    curve[i] = (((Math.PI + drive) * x) / (Math.PI + drive * Math.abs(x))) * trim;
  }

  return curve;
}

function setParam(param: AudioParam, value: number, now: number) {
  param.cancelScheduledValues(now);
  param.setTargetAtTime(value, now, 0.035);
}

function getElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element #${id}`);
  }
  return element as T;
}

function getCanvasContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas context is unavailable");
  }
  return context;
}

function lerp(start: number, end: number, amount: number) {
  return start + (end - start) * amount;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
