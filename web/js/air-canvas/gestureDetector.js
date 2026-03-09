import { LANDMARKS, GESTURE } from './constants.js';

export class GestureDetector {
  constructor() {
    this.lastLandmarks = null;
    this.lastTime = 0;
    this.gestureStartTime = 0;
    this.currentGesture = 'none';
    this.previousGesture = 'none';
    this.palmHistory = [];
    this.velocityHistory = [];
  }

  detect(landmarks) {
    const now = performance.now();
    const dt = this.lastTime > 0 ? (now - this.lastTime) / 1000 : 0;
    this.lastTime = now;

    if (!landmarks) {
      return this.createState('none', { x: 0, y: 0 }, 0);
    }

    const velocity = this.calculateVelocity(landmarks, dt);
    const detectedGesture = this.detectGestureType(landmarks, velocity);

    if (detectedGesture !== this.currentGesture) {
      this.previousGesture = this.currentGesture;
      this.currentGesture = detectedGesture;
      this.gestureStartTime = now;
    }

    const duration = now - this.gestureStartTime;
    this.lastLandmarks = landmarks;

    return this.createState(this.currentGesture, velocity, duration);
  }

  createState(gesture, velocity, duration) {
    return {
      current: gesture,
      previous: this.previousGesture,
      duration,
      velocity,
      confidence: 1.0
    };
  }

  calculateVelocity(landmarks, dt) {
    if (!this.lastLandmarks || dt === 0) {
      return { x: 0, y: 0 };
    }

    const currentPalm = this.getPalmCenter(landmarks);
    const lastPalm = this.getPalmCenter(this.lastLandmarks);

    const velocity = {
      x: (currentPalm.x - lastPalm.x) / dt,
      y: (currentPalm.y - lastPalm.y) / dt
    };

    this.velocityHistory.push(velocity);
    if (this.velocityHistory.length > 2) {
      this.velocityHistory.shift();
    }

    const avgVelocity = this.velocityHistory.reduce(
      (acc, v) => ({ x: acc.x + v.x, y: acc.y + v.y }),
      { x: 0, y: 0 }
    );

    return {
      x: avgVelocity.x / this.velocityHistory.length,
      y: avgVelocity.y / this.velocityHistory.length
    };
  }

  getPalmCenter(landmarks) {
    const wrist = landmarks.landmarks[LANDMARKS.WRIST];
    const indexMcp = landmarks.landmarks[LANDMARKS.INDEX_MCP];
    const pinkyMcp = landmarks.landmarks[LANDMARKS.PINKY_MCP];

    return {
      x: (wrist.x + indexMcp.x + pinkyMcp.x) / 3,
      y: (wrist.y + indexMcp.y + pinkyMcp.y) / 3
    };
  }

  detectGestureType(landmarks, velocity) {
    const lm = landmarks.landmarks;

    const speed = Math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
    if (speed > GESTURE.SWIPE_VELOCITY && Math.abs(velocity.x) > Math.abs(velocity.y) * 1.5) {
      return 'swipe';
    }

    const pinchDistance = this.distance(lm[LANDMARKS.THUMB_TIP], lm[LANDMARKS.INDEX_TIP]);
    if (pinchDistance < GESTURE.PINCH_THRESHOLD) {
      return 'pinch';
    }

    if (this.isFist(landmarks)) {
      return 'fist';
    }

    if (this.isOpenPalm(landmarks)) {
      this.palmHistory.push(this.getPalmCenter(landmarks));
      if (this.palmHistory.length > 6) {
        this.palmHistory.shift();
      }
      if (this.palmHistory.length >= 3 && this.isPalmStable()) {
        return 'palm';
      }
    } else {
      this.palmHistory = [];
    }

    if (this.isPointingIndex(landmarks)) {
      return 'draw';
    }

    return 'none';
  }

  distance(p1, p2) {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  isFingerExtended(landmarks, tipIdx, pipIdx, _mcpIdx) {
    const lm = landmarks.landmarks;
    const tip = lm[tipIdx];
    const pip = lm[pipIdx];
    const tipToPalm = this.distance(tip, lm[LANDMARKS.WRIST]);
    const pipToPalm = this.distance(pip, lm[LANDMARKS.WRIST]);
    return tipToPalm > pipToPalm * GESTURE.FINGER_CURL_THRESHOLD;
  }

  isThumbExtended(landmarks) {
    const lm = landmarks.landmarks;
    const thumbTip = lm[LANDMARKS.THUMB_TIP];
    const thumbIp = lm[LANDMARKS.THUMB_IP];
    const indexMcp = lm[LANDMARKS.INDEX_MCP];
    const distFromIndex = this.distance(thumbTip, indexMcp);
    const thumbLength = this.distance(thumbTip, thumbIp);
    return distFromIndex > thumbLength * 1.5;
  }

  isPointingIndex(landmarks) {
    const indexExtended = this.isFingerExtended(
      landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP, LANDMARKS.INDEX_MCP
    );
    const middleCurled = !this.isFingerExtended(
      landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP, LANDMARKS.MIDDLE_MCP
    );
    return indexExtended && middleCurled;
  }

  isOpenPalm(landmarks) {
    return (
      this.isFingerExtended(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP, LANDMARKS.INDEX_MCP) &&
      this.isFingerExtended(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP, LANDMARKS.MIDDLE_MCP) &&
      this.isFingerExtended(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP, LANDMARKS.RING_MCP) &&
      this.isFingerExtended(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP, LANDMARKS.PINKY_MCP) &&
      this.isThumbExtended(landmarks)
    );
  }

  isFist(landmarks) {
    return (
      !this.isFingerExtended(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP, LANDMARKS.INDEX_MCP) &&
      !this.isFingerExtended(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP, LANDMARKS.MIDDLE_MCP) &&
      !this.isFingerExtended(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP, LANDMARKS.RING_MCP) &&
      !this.isFingerExtended(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP, LANDMARKS.PINKY_MCP) &&
      !this.isThumbExtended(landmarks)
    );
  }

  isPalmStable() {
    if (this.palmHistory.length < 3) return false;
    const recent = this.palmHistory.slice(-3);
    const first = recent[0];
    for (const point of recent) {
      if (this.distance(point, first) > GESTURE.PALM_STABILITY_THRESHOLD) {
        return false;
      }
    }
    return true;
  }

  getIndexTip(landmarks) {
    return landmarks.landmarks[LANDMARKS.INDEX_TIP];
  }

  getThumbTip(landmarks) {
    return landmarks.landmarks[LANDMARKS.THUMB_TIP];
  }

  getPinchCenter(landmarks) {
    const thumb = this.getThumbTip(landmarks);
    const index = this.getIndexTip(landmarks);
    return {
      x: (thumb.x + index.x) / 2,
      y: (thumb.y + index.y) / 2
    };
  }
}
