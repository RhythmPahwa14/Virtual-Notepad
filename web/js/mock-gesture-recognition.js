class MockGestureRecognitionApp {
    constructor() {
        this.isRunning = false;
        this.debugMode = false;
        this.gestureClasses = ['fist', 'open_palm', 'peace', 'point_up', 'rock', 'thumbs_up'];
        this.currentGesture = null;
        this.currentConfidence = 0;
        this.frameCount = 0;
        this.lastTime = performance.now();
        
        // DOM elements
        this.inputVideo = document.getElementById('input_video');
        this.outputCanvas = document.getElementById('output_canvas');
        this.canvasCtx = this.outputCanvas.getContext('2d');
        this.gestureDisplay = document.getElementById('gesture_display');
        this.confidenceFill = document.getElementById('confidence_fill');
        this.confidenceText = document.getElementById('confidence_text');
        this.modelStatus = document.getElementById('model_status');
        this.cameraStatus = document.getElementById('camera_status');
        this.fpsDisplay = document.getElementById('fps_display');
        this.debugPanel = document.getElementById('debug_panel');
        this.debugText = document.getElementById('debug_text');
        
        // Buttons
        this.startBtn = document.getElementById('start_btn');
        this.stopBtn = document.getElementById('stop_btn');
        this.debugBtn = document.getElementById('debug_btn');
        
        this.initializeEventListeners();
        this.initializeMockModel();
    }
    
    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startRecognition());
        this.stopBtn.addEventListener('click', () => this.stopRecognition());
        this.debugBtn.addEventListener('click', () => this.toggleDebug());
    }
    
    initializeMockModel() {
        // Simulate model loading
        this.updateStatus('model_status', 'Loading Mock Model...', 'loading');
        
        setTimeout(() => {
            this.updateStatus('model_status', 'Mock Model Ready ✅', 'ready');
            this.startBtn.disabled = false;
            this.log('Mock gesture recognition model initialized');
        }, 1000);
    }
    
    async initializeCamera() {
        try {
            this.updateStatus('camera_status', 'Starting Camera...', 'loading');
            
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: 640,
                    height: 480,
                    facingMode: 'user'
                }
            });
            
            this.inputVideo.srcObject = stream;
            
            return new Promise((resolve) => {
                this.inputVideo.onloadedmetadata = () => {
                    this.updateStatus('camera_status', 'Camera Ready ✅', 'ready');
                    this.log('Camera initialized successfully');
                    resolve();
                };
            });
            
        } catch (error) {
            this.log(`Error initializing camera: ${error.message}`);
            this.updateStatus('camera_status', 'Camera Failed ❌', 'error');
            throw error;
        }
    }
    
    initializeMediaPipe() {
        // Initialize MediaPipe Hands
        this.hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });
        
        this.hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        this.hands.onResults((results) => this.onResults(results));
        
        // Setup camera
        this.camera = new Camera(this.inputVideo, {
            onFrame: async () => {
                if (this.hands && this.isRunning) {
                    await this.hands.send({image: this.inputVideo});
                }
            },
            width: 640,
            height: 480
        });
        
        this.log('MediaPipe Hands initialized');
    }
    
    onResults(results) {
        // Update FPS
        this.updateFPS();
        
        // Clear canvas
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, this.outputCanvas.width, this.outputCanvas.height);
        this.canvasCtx.drawImage(results.image, 0, 0, this.outputCanvas.width, this.outputCanvas.height);
        
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];
            
            // Draw hand landmarks
            this.drawLandmarks(landmarks);
            
            // Mock gesture prediction
            this.mockPredictGesture(landmarks);
            
        } else {
            // No hands detected
            this.currentGesture = null;
            this.currentConfidence = 0;
            this.updateGestureDisplay('No hand detected', 0);
        }
        
        this.canvasCtx.restore();
    }
    
    drawLandmarks(landmarks) {
        // Draw connections
        drawConnectors(this.canvasCtx, landmarks, HAND_CONNECTIONS, {
            color: '#00FF00',
            lineWidth: 2
        });
        
        // Draw landmarks
        drawLandmarks(this.canvasCtx, landmarks, {
            color: '#FF0000',
            lineWidth: 1,
            radius: 3
        });
    }
    
    mockPredictGesture(landmarks) {
        // Simple rule-based gesture recognition for demo
        const gesture = this.detectGestureByRules(landmarks);
        const confidence = Math.random() * 0.3 + 0.7; // Random confidence 70-100%
        
        this.currentGesture = gesture;
        this.currentConfidence = confidence;
        this.updateGestureDisplay(gesture, confidence);
        
        if (this.debugMode) {
            this.updateMockDebugInfo(landmarks, gesture, confidence);
        }
    }
    
    detectGestureByRules(landmarks) {
        // Simple rule-based detection using finger positions
        const fingerTips = [4, 8, 12, 16, 20]; // thumb, index, middle, ring, pinky tips
        const fingerPips = [3, 6, 10, 14, 18]; // corresponding PIP joints
        
        const fingersUp = [];
        
        // Check each finger
        for (let i = 0; i < fingerTips.length; i++) {
            const tip = landmarks[fingerTips[i]];
            const pip = landmarks[fingerPips[i]];
            
            if (i === 0) { // Thumb special case
                fingersUp.push(tip.x > pip.x); // Thumb extends sideways
            } else {
                fingersUp.push(tip.y < pip.y); // Other fingers extend upward
            }
        }
        
        // Gesture detection logic
        const upCount = fingersUp.filter(Boolean).length;
        
        if (upCount === 0) return 'fist';
        if (upCount === 5) return 'open_palm';
        if (upCount === 1 && fingersUp[0]) return 'thumbs_up';
        if (upCount === 1 && fingersUp[1]) return 'point_up';
        if (upCount === 2 && fingersUp[1] && fingersUp[2]) return 'peace';
        if (upCount === 2 && fingersUp[1] && fingersUp[4]) return 'rock';
        
        return 'unknown';
    }
    
    updateMockDebugInfo(landmarks, gesture, confidence) {
        const debugInfo = [
            `🎯 Mock Recognition Active`,
            `Current Gesture: ${gesture}`,
            `Confidence: ${(confidence * 100).toFixed(2)}%`,
            '',
            `📊 Landmarks: ${landmarks.length} detected`,
            `🎮 Rule-based detection active`,
            `⚡ Real-time processing`,
            '',
            'To use AI model:',
            '1. Fix TensorFlow installation',
            '2. Convert model to TensorFlow.js',
            '3. Replace mock with real model'
        ];
        
        this.debugText.textContent = debugInfo.join('\n');
    }
    
    updateGestureDisplay(gesture, confidence) {
        const gestureEmojis = {
            'fist': '✊',
            'open_palm': '✋',
            'peace': '✌️',
            'point_up': '☝️',
            'rock': '🤟',
            'thumbs_up': '👍',
            'unknown': '❓'
        };
        
        const emoji = gestureEmojis[gesture] || '❓';
        const displayText = gesture ? `${emoji} ${gesture.replace('_', ' ')}` : gesture;
        
        this.gestureDisplay.textContent = displayText;
        this.confidenceText.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
        this.confidenceFill.style.width = `${confidence * 100}%`;
    }
    
    updateFPS() {
        this.frameCount++;
        const now = performance.now();
        const deltaTime = now - this.lastTime;
        
        if (deltaTime >= 1000) { // Update FPS every second
            const fps = Math.round((this.frameCount * 1000) / deltaTime);
            this.fpsDisplay.textContent = `FPS: ${fps}`;
            this.frameCount = 0;
            this.lastTime = now;
        }
    }
    
    async startRecognition() {
        if (this.isRunning) return;
        
        try {
            this.log('Starting mock gesture recognition...');
            
            // Initialize MediaPipe
            this.initializeMediaPipe();
            
            // Initialize camera
            await this.initializeCamera();
            
            // Start camera
            await this.camera.start();
            
            // Set canvas size to match video
            this.outputCanvas.width = this.inputVideo.videoWidth || 640;
            this.outputCanvas.height = this.inputVideo.videoHeight || 480;
            
            this.isRunning = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            
            this.log('Mock gesture recognition started successfully');
            
        } catch (error) {
            this.log(`Error starting recognition: ${error.message}`);
            this.updateStatus('camera_status', 'Start Failed ❌', 'error');
        }
    }
    
    stopRecognition() {
        if (!this.isRunning) return;
        
        this.log('Stopping gesture recognition...');
        
        this.isRunning = false;
        
        if (this.camera) {
            this.camera.stop();
        }
        
        // Stop camera stream
        if (this.inputVideo.srcObject) {
            const tracks = this.inputVideo.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            this.inputVideo.srcObject = null;
        }
        
        // Clear canvas
        this.canvasCtx.clearRect(0, 0, this.outputCanvas.width, this.outputCanvas.height);
        
        // Reset display
        this.updateGestureDisplay('Stopped', 0);
        this.fpsDisplay.textContent = 'FPS: --';
        
        // Update buttons
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        
        // Update status
        this.updateStatus('camera_status', 'Camera Stopped', 'loading');
        
        this.log('Gesture recognition stopped');
    }
    
    toggleDebug() {
        this.debugMode = !this.debugMode;
        this.debugPanel.style.display = this.debugMode ? 'block' : 'none';
        this.debugBtn.textContent = this.debugMode ? 'Hide Debug' : 'Show Debug';
        this.log(`Debug mode ${this.debugMode ? 'enabled' : 'disabled'}`);
    }
    
    updateStatus(elementId, text, className) {
        const element = document.getElementById(elementId);
        element.textContent = text;
        element.className = `status ${className}`;
    }
    
    log(message) {
        const timestamp = new Date().toLocaleTimeString();
        console.log(`[${timestamp}] ${message}`);
        
        if (this.debugMode) {
            const currentDebug = this.debugText.textContent;
            this.debugText.textContent = `[${timestamp}] ${message}\n${currentDebug}`;
        }
    }
}

// Initialize the mock app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.gestureApp = new MockGestureRecognitionApp();
});
