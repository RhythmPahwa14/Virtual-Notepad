import { LANDMARKS, VISUAL } from './constants.js';

const HAND_CONNECTIONS = [
  [LANDMARKS.WRIST, LANDMARKS.THUMB_CMC],
  [LANDMARKS.THUMB_CMC, LANDMARKS.THUMB_MCP],
  [LANDMARKS.THUMB_MCP, LANDMARKS.THUMB_IP],
  [LANDMARKS.THUMB_IP, LANDMARKS.THUMB_TIP],
  [LANDMARKS.WRIST, LANDMARKS.INDEX_MCP],
  [LANDMARKS.INDEX_MCP, LANDMARKS.INDEX_PIP],
  [LANDMARKS.INDEX_PIP, LANDMARKS.INDEX_DIP],
  [LANDMARKS.INDEX_DIP, LANDMARKS.INDEX_TIP],
  [LANDMARKS.WRIST, LANDMARKS.MIDDLE_MCP],
  [LANDMARKS.MIDDLE_MCP, LANDMARKS.MIDDLE_PIP],
  [LANDMARKS.MIDDLE_PIP, LANDMARKS.MIDDLE_DIP],
  [LANDMARKS.MIDDLE_DIP, LANDMARKS.MIDDLE_TIP],
  [LANDMARKS.WRIST, LANDMARKS.RING_MCP],
  [LANDMARKS.RING_MCP, LANDMARKS.RING_PIP],
  [LANDMARKS.RING_PIP, LANDMARKS.RING_DIP],
  [LANDMARKS.RING_DIP, LANDMARKS.RING_TIP],
  [LANDMARKS.WRIST, LANDMARKS.PINKY_MCP],
  [LANDMARKS.PINKY_MCP, LANDMARKS.PINKY_PIP],
  [LANDMARKS.PINKY_PIP, LANDMARKS.PINKY_DIP],
  [LANDMARKS.PINKY_DIP, LANDMARKS.PINKY_TIP],
  [LANDMARKS.INDEX_MCP, LANDMARKS.MIDDLE_MCP],
  [LANDMARKS.MIDDLE_MCP, LANDMARKS.RING_MCP],
  [LANDMARKS.RING_MCP, LANDMARKS.PINKY_MCP]
];

export class HandVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context');
    this.ctx = ctx;
    this.cursorPulse = 0;
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  render(landmarks, gestureState, currentColor, deltaTime) {
    this.clear();

    if (!landmarks) return;

    this.cursorPulse += deltaTime * VISUAL.CURSOR_PULSE_SPEED;

    this.renderSkeleton(landmarks);

    const indexTip = landmarks.landmarks[LANDMARKS.INDEX_TIP];
    this.renderCursor(indexTip.x, indexTip.y, currentColor, gestureState);
  }

  renderSkeleton(landmarks) {
    this.ctx.save();
    this.ctx.globalAlpha = VISUAL.HAND_SKELETON_OPACITY;
    this.ctx.strokeStyle = 'white';
    this.ctx.lineWidth = VISUAL.HAND_SKELETON_WIDTH;
    this.ctx.lineCap = 'round';

    for (const [from, to] of HAND_CONNECTIONS) {
      const start = landmarks.landmarks[from];
      const end = landmarks.landmarks[to];

      this.ctx.beginPath();
      this.ctx.moveTo(start.x, start.y);
      this.ctx.lineTo(end.x, end.y);
      this.ctx.stroke();
    }

    this.ctx.fillStyle = 'white';
    for (const landmark of landmarks.landmarks) {
      this.ctx.beginPath();
      this.ctx.arc(landmark.x, landmark.y, 3, 0, Math.PI * 2);
      this.ctx.fill();
    }

    this.ctx.restore();
  }

  renderCursor(x, y, color, gestureState) {
    const isReadyToDraw = gestureState.current === 'draw' || gestureState.current === 'none';
    const pulseAmount = isReadyToDraw ? Math.sin(this.cursorPulse) * 0.3 + 1 : 1;

    this.ctx.save();

    const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, VISUAL.CURSOR_GLOW_SIZE * pulseAmount);
    gradient.addColorStop(0, color);
    gradient.addColorStop(0.3, this.hexToRgba(color, 0.5));
    gradient.addColorStop(1, this.hexToRgba(color, 0));

    this.ctx.fillStyle = gradient;
    this.ctx.beginPath();
    this.ctx.arc(x, y, VISUAL.CURSOR_GLOW_SIZE * pulseAmount, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.fillStyle = color;
    this.ctx.shadowColor = color;
    this.ctx.shadowBlur = 10;
    this.ctx.beginPath();
    this.ctx.arc(x, y, (VISUAL.CURSOR_SIZE / 2) * pulseAmount, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.fillStyle = 'white';
    this.ctx.beginPath();
    this.ctx.arc(x, y, VISUAL.CURSOR_SIZE / 4, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.restore();
  }

  hexToRgba(hex, alpha) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return `rgba(255, 255, 255, ${alpha})`;
    const r = parseInt(result[1], 16);
    const g = parseInt(result[2], 16);
    const b = parseInt(result[3], 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
}
