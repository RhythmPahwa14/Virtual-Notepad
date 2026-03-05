export class HandTracker {
  constructor(videoElement) {
    this.videoElement = videoElement;
    this.callback = null;
    this.isRunning = false;
    this.animationId = null;
    this.canvasWidth = 640;
    this.canvasHeight = 480;

    // Hands is loaded from CDN as a global
    this.hands = new window.Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    });

    this.hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
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
    if (!this.callback) return;

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];
      const worldLandmarks = results.multiHandWorldLandmarks ? results.multiHandWorldLandmarks[0] : null;

      const convertedLandmarks = landmarks.map((lm) => ({
        x: (1 - lm.x) * this.canvasWidth,
        y: lm.y * this.canvasHeight
      }));

      const convertedWorldLandmarks = worldLandmarks ? worldLandmarks.map((lm) => ({
        x: -lm.x,
        y: -lm.y,
        z: lm.z
      })) : undefined;

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

      const processFrame = async () => {
        if (!this.isRunning) return;

        if (this.videoElement.readyState >= 2) {
          await this.hands.send({ image: this.videoElement });
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
