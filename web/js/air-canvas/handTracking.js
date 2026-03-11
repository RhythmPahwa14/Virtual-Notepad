// HandTracker: wraps MediaPipe Hands (loaded as global via CDN script tag)
export class HandTracker {
  constructor(videoElement) {
    this.videoElement = videoElement;
    this.callback = null;
    this.isRunning = false;
    this.isSending = false; // backpressure: don't queue more frames than MediaPipe can handle
    this.animationId = null;
    this.canvasWidth = 640;
    this.canvasHeight = 480;

    // window.Hands is provided by the @mediapipe/hands CDN script tag
    this.hands = new window.Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    });

    this.hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 0,        // lightweight model: ~2x faster, no noticeable quality loss for tip tracking
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5
    });

    this.hands.onResults((results) => this.onResults(results));
  }

  setCanvasSize(width, height) {
    this.canvasWidth = width;
    this.canvasHeight = height;
  }

  onResults(results) {
    this.isSending = false; // previous frame fully processed — ready for the next one
    if (!this.callback) return;

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];
      const worldLandmarks = results.multiHandWorldLandmarks?.[0];

      const convertedLandmarks = landmarks.map((lm) => ({
        x: (1 - lm.x) * this.canvasWidth,
        y: lm.y * this.canvasHeight
      }));

      const convertedWorldLandmarks = worldLandmarks?.map((lm) => ({
        x: -lm.x,
        y: -lm.y,
        z: lm.z
      }));

      this.callback({
        landmarks: convertedLandmarks,
        worldLandmarks: convertedWorldLandmarks
      });
    } else {
      this.callback(null);
    }
  }

  async start(callback) {
    this.callback = callback;

    if (this.isRunning) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
          facingMode: 'user'
        }
      });

      this.videoElement.srcObject = stream;
      await this.videoElement.play();

      this.isRunning = true;

      const processFrame = () => {
        if (!this.isRunning) return;

        // Only send if MediaPipe has finished the previous frame.
        // Without this guard, frames pile up and create compounding lag.
        if (!this.isSending && this.videoElement.readyState >= 2) {
          this.isSending = true;
          this.hands.send({ image: this.videoElement });
        }

        this.animationId = requestAnimationFrame(processFrame);
      };

      processFrame();
    } catch (error) {
      console.error('Failed to start hand tracking:', error);
      throw error;
    }
  }

  stop() {
    this.isRunning = false;

    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    const stream = this.videoElement.srcObject;
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
  }

  isActive() {
    return this.isRunning;
  }
}
