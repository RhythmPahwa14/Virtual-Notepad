import { HandTracker } from './handTracking.js';
import { GestureDetector } from './gestureDetector.js';
import { DrawingCanvas } from './drawingCanvas.js';
import { HandVisualizer } from './handVisualizer.js';
import { Scene3D } from './scene3D.js';
import { ObjectManager } from './objectManager.js';
import { Multiplayer } from './multiplayer.js';
import { GESTURE, TIMING } from './constants.js';

class AirCanvas {
  constructor() {
    const videoElement = document.getElementById('webcam');
    const sceneCanvas = document.getElementById('scene-canvas');
    const drawCanvas = document.getElementById('draw-canvas');
    const handCanvas = document.getElementById('hand-canvas');

    this.previewVideo = document.getElementById('preview-video');
    this.previewCanvas = document.getElementById('preview-canvas');
    this.previewCtx = this.previewCanvas.getContext('2d');

    this.loadingOverlay = document.getElementById('loading-overlay');
    this.statusMessage = document.getElementById('status-message');
    this.colorSwatches = document.querySelectorAll('.color-swatch');

    this.inviteModal = document.getElementById('invite-modal');
    this.roomCodeDisplay = document.getElementById('room-code');
    this.joinCodeInput = document.getElementById('join-code-input');
    this.statusDot = document.getElementById('status-dot');
    this.statusText = document.getElementById('status-text');

    this.handTracker = new HandTracker(videoElement);
    this.gestureDetector = new GestureDetector();
    this.drawingCanvas = new DrawingCanvas(drawCanvas);
    this.handVisualizer = new HandVisualizer(handCanvas);
    this.scene3D = new Scene3D(sceneCanvas);
    this.objectManager = new ObjectManager(
      this.scene3D,
      window.innerWidth,
      window.innerHeight
    );
    this.multiplayer = new Multiplayer();

    // State
    this.isDrawing = false;
    this.currentColor = '#FFB3BA';
    this.lastGestureState = null;
    this.currentLandmarks = null;
    this.palmHoldStart = 0;
    this.handDetected = false;
    this.lastFrameTime = 0;
    this.grabbedObject = null;
    this.lastPinchPosition = null;

    // Mouse state
    this.isDragging = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;
    this.selectedObject = null;

    // Camera preview drag state
    this.isPreviewDragging = false;
    this.previewDragStartX = 0;
    this.previewDragStartY = 0;
    this.previewStartLeft = 0;
    this.previewStartTop = 0;

    this.resize();
    this.setupEventListeners();
    this.setupButtonListeners();
    this.setupPreviewDrag();
    this.setupMultiplayer();
    this.init();
  }

  setupEventListeners() {
    window.addEventListener('resize', () => this.resize());

    this.colorSwatches.forEach(swatch => {
      swatch.addEventListener('click', () => {
        this.colorSwatches.forEach(s => s.classList.remove('active'));
        swatch.classList.add('active');
        this.currentColor = swatch.dataset.color || '#FFB3BA';
      });
    });

    const sceneCanvas = document.getElementById('scene-canvas');

    sceneCanvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    sceneCanvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    sceneCanvas.addEventListener('mouseup', () => this.onMouseUp());
    sceneCanvas.addEventListener('mouseleave', () => this.onMouseUp());
    sceneCanvas.addEventListener('wheel', (e) => this.onWheel(e));

    sceneCanvas.addEventListener('touchstart', (e) => this.onTouchStart(e));
    sceneCanvas.addEventListener('touchmove', (e) => this.onTouchMove(e));
    sceneCanvas.addEventListener('touchend', () => this.onMouseUp());

    sceneCanvas.addEventListener('click', (e) => this.onSceneClick(e));
  }

  setupButtonListeners() {
    const clearAllBtn = document.getElementById('clear-all-btn');
    clearAllBtn?.addEventListener('click', () => {
      this.clearAll();
      if (this.multiplayer.isConnected()) {
        this.multiplayer.broadcast({ type: 'clear_all' });
      }
    });

    const inviteBtn = document.getElementById('invite-btn');
    inviteBtn?.addEventListener('click', () => {
      this.openInviteModal();
    });

    const modalClose = document.getElementById('modal-close');
    modalClose?.addEventListener('click', () => {
      this.closeInviteModal();
    });

    this.inviteModal?.addEventListener('click', (e) => {
      if (e.target === this.inviteModal) {
        this.closeInviteModal();
      }
    });

    const copyCodeBtn = document.getElementById('copy-code-btn');
    copyCodeBtn?.addEventListener('click', () => {
      this.copyRoomCode();
    });

    const joinRoomBtn = document.getElementById('join-room-btn');
    joinRoomBtn?.addEventListener('click', () => {
      this.joinRoom();
    });

    this.joinCodeInput?.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.joinRoom();
      }
    });

    const previewExpandBtn = document.getElementById('preview-expand-btn');
    const cameraPreview = document.getElementById('camera-preview');
    previewExpandBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      cameraPreview?.classList.toggle('expanded');
      this.updatePreviewCanvasSize();
    });
  }

  setupPreviewDrag() {
    const cameraPreview = document.getElementById('camera-preview');
    const expandBtn = document.getElementById('preview-expand-btn');
    if (!cameraPreview) return;

    cameraPreview.addEventListener('mousedown', (e) => {
      if (e.target === expandBtn) return;
      this.startPreviewDrag(e.clientX, e.clientY, cameraPreview);
    });

    document.addEventListener('mousemove', (e) => {
      if (!this.isPreviewDragging) return;
      this.movePreview(e.clientX, e.clientY, cameraPreview);
    });

    document.addEventListener('mouseup', () => {
      this.endPreviewDrag(cameraPreview);
    });

    cameraPreview.addEventListener('touchstart', (e) => {
      if (e.target === expandBtn) return;
      if (e.touches.length === 1) {
        e.preventDefault();
        this.startPreviewDrag(e.touches[0].clientX, e.touches[0].clientY, cameraPreview);
      }
    }, { passive: false });

    document.addEventListener('touchmove', (e) => {
      if (!this.isPreviewDragging) return;
      if (e.touches.length === 1) {
        this.movePreview(e.touches[0].clientX, e.touches[0].clientY, cameraPreview);
      }
    }, { passive: true });

    document.addEventListener('touchend', () => {
      this.endPreviewDrag(cameraPreview);
    });

    cameraPreview.addEventListener('dblclick', (e) => {
      if (e.target === expandBtn) return;
      this.resetPreviewPosition(cameraPreview);
    });
  }

  resetPreviewPosition(preview) {
    preview.classList.remove('custom-position');
    preview.style.left = '';
    preview.style.top = '';
  }

  startPreviewDrag(clientX, clientY, preview) {
    this.isPreviewDragging = true;
    this.previewDragStartX = clientX;
    this.previewDragStartY = clientY;

    const rect = preview.getBoundingClientRect();
    this.previewStartLeft = rect.left;
    this.previewStartTop = rect.top;

    preview.classList.add('dragging');
  }

  movePreview(clientX, clientY, preview) {
    const deltaX = clientX - this.previewDragStartX;
    const deltaY = clientY - this.previewDragStartY;

    let newLeft = this.previewStartLeft + deltaX;
    let newTop = this.previewStartTop + deltaY;

    const rect = preview.getBoundingClientRect();
    const maxLeft = window.innerWidth - rect.width;
    const maxTop = window.innerHeight - rect.height;

    newLeft = Math.max(0, Math.min(newLeft, maxLeft));
    newTop = Math.max(0, Math.min(newTop, maxTop));

    preview.classList.add('custom-position');
    preview.style.left = `${newLeft}px`;
    preview.style.top = `${newTop}px`;
  }

  endPreviewDrag(preview) {
    if (this.isPreviewDragging) {
      this.isPreviewDragging = false;
      preview.classList.remove('dragging');
    }
  }

  updatePreviewCanvasSize() {
    const cameraPreview = document.getElementById('camera-preview');
    if (cameraPreview) {
      const rect = cameraPreview.getBoundingClientRect();
      this.previewCanvas.width = rect.width;
      this.previewCanvas.height = rect.height;
    }
  }

  setupMultiplayer() {
    this.multiplayer.initialize().then(() => {
      this.roomCodeDisplay.textContent = this.multiplayer.getRoomCode();
    }).catch(err => {
      console.error('Failed to initialize multiplayer:', err);
    });

    this.multiplayer.onStatusChange((status, message) => {
      this.statusDot.className = 'status-dot';
      if (status === 'connected') {
        this.statusDot.classList.add('connected');
      } else if (status === 'connecting') {
        this.statusDot.classList.add('connecting');
      }
      this.statusText.textContent = message;
    });

    this.multiplayer.onEvent((event) => {
      this.handleMultiplayerEvent(event);
    });
  }

  handleMultiplayerEvent(event) {
    switch (event.type) {
      case 'balloon_created':
        this.objectManager.createFromStroke(event.strokeData);
        break;

      case 'clear_all':
        this.drawingCanvas.clearAll();
        this.objectManager.clearAll();
        break;

      case 'peer_joined':
        this.showStatus('Friend joined!', 2000);
        break;

      case 'peer_left':
        this.showStatus('Friend left', 2000);
        break;
    }
  }

  openInviteModal() {
    this.inviteModal.classList.add('visible');
  }

  closeInviteModal() {
    this.inviteModal.classList.remove('visible');
  }

  async copyRoomCode() {
    const code = this.multiplayer.getRoomCode();
    try {
      await navigator.clipboard.writeText(code);
      const copyBtn = document.getElementById('copy-code-btn');
      if (copyBtn) {
        copyBtn.textContent = '✓';
        setTimeout(() => {
          copyBtn.textContent = '📋';
        }, 2000);
      }
    } catch {
      const textArea = document.createElement('textarea');
      textArea.value = code;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
    }
  }

  async joinRoom() {
    const code = this.joinCodeInput.value.trim().toUpperCase();
    if (code.length !== 6) {
      this.statusText.textContent = 'Please enter a 6-character code';
      return;
    }

    try {
      await this.multiplayer.joinRoom(code);
      this.showStatus('Connected!', 2000);
    } catch {
      this.statusText.textContent = 'Failed to connect';
    }
  }

  onMouseDown(e) {
    this.isDragging = true;
    this.lastMouseX = e.clientX;
    this.lastMouseY = e.clientY;

    const hitObject = this.objectManager.getObjectAtPosition(e.clientX, e.clientY);
    if (hitObject) {
      this.selectedObject = hitObject;
    }
  }

  onMouseMove(e) {
    if (!this.isDragging) return;

    const deltaX = e.clientX - this.lastMouseX;
    const deltaY = e.clientY - this.lastMouseY;

    if (this.selectedObject) {
      this.objectManager.rotateObject(this.selectedObject, deltaX * 0.01, deltaY * 0.01);
    } else {
      this.scene3D.orbitCamera(deltaX * 0.005, deltaY * 0.005);
    }

    this.lastMouseX = e.clientX;
    this.lastMouseY = e.clientY;
  }

  onMouseUp() {
    this.isDragging = false;
    this.selectedObject = null;
  }

  onWheel(e) {
    e.preventDefault();
    this.scene3D.zoomCamera(e.deltaY * 0.001);
  }

  onTouchStart(e) {
    if (e.touches.length === 1) {
      this.isDragging = true;
      this.lastMouseX = e.touches[0].clientX;
      this.lastMouseY = e.touches[0].clientY;

      const hitObject = this.objectManager.getObjectAtPosition(
        e.touches[0].clientX,
        e.touches[0].clientY
      );
      if (hitObject) {
        this.selectedObject = hitObject;
      }
    }
  }

  onTouchMove(e) {
    if (!this.isDragging || e.touches.length !== 1) return;

    const deltaX = e.touches[0].clientX - this.lastMouseX;
    const deltaY = e.touches[0].clientY - this.lastMouseY;

    if (this.selectedObject) {
      this.objectManager.rotateObject(this.selectedObject, deltaX * 0.01, deltaY * 0.01);
    } else {
      this.scene3D.orbitCamera(deltaX * 0.005, deltaY * 0.005);
    }

    this.lastMouseX = e.touches[0].clientX;
    this.lastMouseY = e.touches[0].clientY;
  }

  onSceneClick(e) {
    const hitObject = this.objectManager.getObjectAtPosition(e.clientX, e.clientY);
    if (hitObject) {
      this.objectManager.selectObject(hitObject);
    }
  }

  async init() {
    try {
      await this.handTracker.start((landmarks) => this.onHandResults(landmarks));

      this.setupCameraPreview();

      this.loadingOverlay.classList.add('hidden');

      this.animate();
    } catch (error) {
      console.error('Failed to initialize:', error);
      this.showStatus('Camera access denied. Please allow camera access and refresh.');
    }
  }

  setupCameraPreview() {
    const webcam = document.getElementById('webcam');
    if (webcam.srcObject) {
      this.previewVideo.srcObject = webcam.srcObject;
      this.previewVideo.play();
    }

    this.previewCanvas.width = 320;
    this.previewCanvas.height = 240;
  }

  resize() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    this.handTracker.setCanvasSize(width, height);
    this.drawingCanvas.resize(width, height);
    this.handVisualizer.resize(width, height);
    this.scene3D.resize(width, height);
    this.objectManager.updateSize(width, height);
  }

  onHandResults(landmarks) {
    const wasDetected = this.handDetected;
    this.handDetected = landmarks !== null;
    this.currentLandmarks = landmarks;

    if (!this.handDetected && wasDetected) {
      this.showStatus('Show your hand to begin');
    } else if (this.handDetected && !wasDetected) {
      this.hideStatus();
    }

    this.renderPreviewOverlay(landmarks);

    if (!landmarks) {
      if (this.isDrawing) {
        this.isDrawing = false;
      }
      return;
    }

    const gestureState = this.gestureDetector.detect(landmarks);

    this.handleGesture(gestureState, landmarks);

    this.lastGestureState = gestureState;
  }

  renderPreviewOverlay(landmarks) {
    const previewWidth = this.previewCanvas.width || 320;
    const previewHeight = this.previewCanvas.height || 240;
    this.previewCtx.clearRect(0, 0, previewWidth, previewHeight);

    if (!landmarks) return;

    const scale = Math.min(previewWidth / window.innerWidth, previewHeight / window.innerHeight);
    const offsetX = (previewWidth - window.innerWidth * scale) / 2;
    const offsetY = (previewHeight - window.innerHeight * scale) / 2;

    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
      [5, 9], [9, 13], [13, 17]
    ];

    const uiScale = previewWidth / 320;
    this.previewCtx.strokeStyle = '#6366f1';
    this.previewCtx.lineWidth = 2 * uiScale;

    for (const [from, to] of connections) {
      const start = landmarks.landmarks[from];
      const end = landmarks.landmarks[to];

      this.previewCtx.beginPath();
      this.previewCtx.moveTo(start.x * scale + offsetX, start.y * scale + offsetY);
      this.previewCtx.lineTo(end.x * scale + offsetX, end.y * scale + offsetY);
      this.previewCtx.stroke();
    }

    this.previewCtx.fillStyle = '#6366f1';
    for (const lm of landmarks.landmarks) {
      this.previewCtx.beginPath();
      this.previewCtx.arc(lm.x * scale + offsetX, lm.y * scale + offsetY, 3 * uiScale, 0, Math.PI * 2);
      this.previewCtx.fill();
    }
  }

  handleGesture(state, landmarks) {
    const indexTip = this.gestureDetector.getIndexTip(landmarks);

    switch (state.current) {
      case 'draw':
        this.handleDraw(indexTip);
        break;

      case 'pinch':
        this.handlePinch(landmarks);
        break;

      case 'palm':
        this.handlePalm();
        break;

      case 'swipe':
        this.handleSwipe(indexTip);
        break;

      default:
        if (this.grabbedObject) {
          this.objectManager.releaseObject(this.grabbedObject);
          this.grabbedObject = null;
          this.lastPinchPosition = null;
        }
        break;
    }

    if (this.lastGestureState && state.current !== this.lastGestureState.current) {
      this.palmHoldStart = 0;
      if (this.lastGestureState.current === 'draw') {
        this.drawingCanvas.clearLivePosition();
      }
    }
  }

  handleDraw(position) {
    this.drawingCanvas.updateLivePosition(position);

    const hitObject = this.objectManager.getObjectAtPosition(position.x, position.y);
    if (hitObject) {
      this.objectManager.pokeObject(hitObject);
      return;
    }

    if (!this.isDrawing) {
      this.isDrawing = true;
      this.drawingCanvas.startStroke(position, this.currentColor);
    } else {
      this.drawingCanvas.addPoint(position);
    }

    this.drawingCanvas.render();
  }

  handlePinch(landmarks) {
    const pinchCenter = this.gestureDetector.getPinchCenter(landmarks);

    if (this.isDrawing) {
      this.isDrawing = false;
      this.drawingCanvas.pauseStroke();
    }

    if (!this.grabbedObject) {
      const hitObject = this.objectManager.getObjectAtPosition(pinchCenter.x, pinchCenter.y);
      if (hitObject) {
        this.grabbedObject = hitObject;
        this.objectManager.grabObject(hitObject);
        this.lastPinchPosition = pinchCenter;
      }
    } else {
      if (this.lastPinchPosition) {
        const deltaX = pinchCenter.x - this.lastPinchPosition.x;
        const deltaY = pinchCenter.y - this.lastPinchPosition.y;

        this.objectManager.moveGrabbedObject(this.grabbedObject, pinchCenter.x, pinchCenter.y);
        this.objectManager.rotateObject(this.grabbedObject, deltaX * 0.02, deltaY * 0.02);
      }
      this.lastPinchPosition = pinchCenter;
    }
  }

  handlePalm() {
    if (this.grabbedObject) {
      this.objectManager.releaseObject(this.grabbedObject);
      this.grabbedObject = null;
      this.lastPinchPosition = null;
    }

    if (this.palmHoldStart === 0) {
      this.palmHoldStart = performance.now();
    }

    const holdDuration = performance.now() - this.palmHoldStart;
    if (holdDuration >= GESTURE.PALM_HOLD_TIME) {
      this.closeAndInflate();
      this.palmHoldStart = 0;
    }
  }

  handleSwipe(position) {
    const hitObject = this.objectManager.getObjectAtPosition(position.x, position.y);
    if (hitObject) {
      this.objectManager.removeObject(hitObject);
    }
  }

  async closeAndInflate() {
    const stroke = this.drawingCanvas.closeStroke();
    this.drawingCanvas.clearLivePosition();

    if (!stroke) {
      this.showStatus('Draw a larger shape', 1000);
      return;
    }

    this.isDrawing = false;

    const startTime = performance.now();
    const animate = () => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(elapsed / (TIMING.STROKE_CLOSE_PULSE * 1000), 1);

      this.drawingCanvas.renderClosingAnimation(stroke, progress);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        this.createBalloon(stroke);
      }
    };
    animate();
  }

  async createBalloon(stroke) {
    this.drawingCanvas.removeCompletedStroke(stroke);
    this.drawingCanvas.clear();

    try {
      await this.objectManager.createFromStroke(stroke);

      if (this.multiplayer.isConnected()) {
        this.multiplayer.broadcast({
          type: 'balloon_created',
          strokeData: stroke
        });
      }
    } catch (error) {
      console.error('Failed to create balloon:', error);
      this.showStatus('Failed to create shape', 2000);
    }
  }

  async clearAll() {
    this.showStatus('Clearing all...');
    this.drawingCanvas.clearAll();
    await this.objectManager.clearAll();
    this.hideStatus();
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const now = performance.now();
    const deltaTime = this.lastFrameTime > 0 ? (now - this.lastFrameTime) / 1000 : 0.016;
    this.lastFrameTime = now;

    this.objectManager.update(deltaTime, now / 1000);

    this.scene3D.render();

    this.drawingCanvas.render();

    const gestureState = this.lastGestureState || {
      current: 'none',
      previous: 'none',
      duration: 0,
      velocity: { x: 0, y: 0 },
      confidence: 0
    };
    this.handVisualizer.render(
      this.currentLandmarks,
      gestureState,
      this.currentColor,
      deltaTime
    );
  }

  showStatus(message, duration) {
    this.statusMessage.textContent = message;
    this.statusMessage.classList.add('visible');

    if (duration) {
      setTimeout(() => this.hideStatus(), duration);
    }
  }

  hideStatus() {
    this.statusMessage.classList.remove('visible');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new AirCanvas();
});
