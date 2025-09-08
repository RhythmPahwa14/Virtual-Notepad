class GestureRecognitionApp {
    constructor() {
        this.model = null;
        this.hands = null;
        this.camera = null;
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
        this.loadModel();
    }
    
    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startRecognition());
        this.stopBtn.addEventListener('click', () => this.stopRecognition());
        this.debugBtn.addEventListener('click', () => this.toggleDebug());
    }
    
    async loadModel() {
        try {
            this.updateStatus('model_status', 'Loading Model...', 'loading');
            this.log('Loading TensorFlow.js model...');
            
            // Try to load the model from different possible paths
            const modelPaths = [
                'models/float32/model.json',
                'models/model.json',
                './models/float32/model.json'
            ];
            
            let modelLoaded = false;
            for (const path of modelPaths) {
                try {
                    this.model = await tf.loadLayersModel(path);
                    this.log(`Model loaded successfully from: ${path}`);
                    modelLoaded = true;
                    break;
                } catch (error) {
                    this.log(`Failed to load model from ${path}: ${error.message}`);
                }
            }
            
            if (!modelLoaded) {
                throw new Error('Could not load model from any path');
            }
            
            // Verify model architecture
            const inputShape = this.model.inputs[0].shape;
            const outputShape = this.model.outputs[0].shape;
            this.log(`Model input shape: [${inputShape}]`);
            this.log(`Model output shape: [${outputShape}]`);
            
            this.updateStatus('model_status', 'Model Ready ✅', 'ready');
            this.startBtn.disabled = false;
            
        } catch (error) {
            this.log(`Error loading model: ${error.message}`);
            this.updateStatus('model_status', 'Model Load Failed ❌', 'error');
        }
    }
    
    async initializeCamera() {
        try {
            this.updateStatus('camera_status', 'Starting Camera...', 'loading');
            
            this.camera = new Camera(this.inputVideo, {
                onFrame: async () => {
                    if (this.hands && this.isRunning) {
                        await this.hands.send({image: this.inputVideo});
                    }
                },
                width: 640,
                height: 480
            });
            
            await this.camera.start();
            this.updateStatus('camera_status', 'Camera Ready ✅', 'ready');
            this.log('Camera initialized successfully');
            
        } catch (error) {
            this.log(`Error initializing camera: ${error.message}`);
            this.updateStatus('camera_status', 'Camera Failed ❌', 'error');
        }
    }
    
    initializeMediaPipe() {
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
            
            // Predict gesture
            this.predictGesture(landmarks);
            
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
    
    async predictGesture(landmarks) {
        if (!this.model) return;
        
        try {
            // Preprocess landmarks
            const processedLandmarks = this.preprocessLandmarks(landmarks);
            
            // Create tensor
            const inputTensor = tf.tensor(processedLandmarks, [1, 21, 3]);
            
            // Predict
            const prediction = await this.model.predict(inputTensor);
            const probabilities = await prediction.data();
            
            // Get predicted class and confidence
            const maxIndex = probabilities.indexOf(Math.max(...probabilities));
            const confidence = probabilities[maxIndex];
            const gestureClass = this.gestureClasses[maxIndex] || 'unknown';
            
            // Update display
            this.currentGesture = gestureClass;
            this.currentConfidence = confidence;
            this.updateGestureDisplay(gestureClass, confidence);
            
            // Debug info
            if (this.debugMode) {
                this.updateDebugInfo(probabilities, processedLandmarks);
            }
            
            // Cleanup tensors
            inputTensor.dispose();
            prediction.dispose();
            
        } catch (error) {
            this.log(`Error in prediction: ${error.message}`);
        }
    }
    
    preprocessLandmarks(landmarks) {
        // Convert MediaPipe landmarks to array
        const landmarkArray = landmarks.map(landmark => [
            landmark.x,
            landmark.y,
            landmark.z || 0
        ]);
        
        // Convert to numpy-like array
        let processedLandmarks = landmarkArray;
        
        // Normalize relative to hand center (same as training preprocessing)
        const center = this.calculateCenter(processedLandmarks);
        processedLandmarks = processedLandmarks.map(point => [
            point[0] - center[0],
            point[1] - center[1],
            point[2] - center[2]
        ]);
        
        // Scale to unit sphere
        const maxDistance = Math.max(...processedLandmarks.map(point => 
            Math.sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])
        ));
        
        if (maxDistance > 0) {
            processedLandmarks = processedLandmarks.map(point => [
                point[0] / maxDistance,
                point[1] / maxDistance,
                point[2] / maxDistance
            ]);
        }
        
        return processedLandmarks;
    }
    
    calculateCenter(landmarks) {
        const sum = landmarks.reduce((acc, point) => [
            acc[0] + point[0],
            acc[1] + point[1],
            acc[2] + point[2]
        ], [0, 0, 0]);
        
        return [
            sum[0] / landmarks.length,
            sum[1] / landmarks.length,
            sum[2] / landmarks.length
        ];
    }
    
    updateGestureDisplay(gesture, confidence) {
        const gestureEmojis = {
            'fist': '✊',
            'open_palm': '✋',
            'peace': '✌️',
            'point_up': '☝️',
            'rock': '🤟',
            'thumbs_up': '👍'
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
    
    updateDebugInfo(probabilities, landmarks) {
        const debugInfo = [
            `Current Gesture: ${this.currentGesture}`,
            `Confidence: ${(this.currentConfidence * 100).toFixed(2)}%`,
            '',
            'Class Probabilities:',
            ...this.gestureClasses.map((cls, idx) => 
                `  ${cls}: ${(probabilities[idx] * 100).toFixed(2)}%`
            ),
            '',
            `Landmark Count: ${landmarks.length}`,
            `First Landmark: [${landmarks[0].map(x => x.toFixed(3)).join(', ')}]`,
            `Model Memory: ${tf.memory().numTensors} tensors`
        ];
        
        this.debugText.textContent = debugInfo.join('\n');
    }
    
    async startRecognition() {
        if (this.isRunning) return;
        
        try {
            this.log('Starting gesture recognition...');
            
            // Initialize MediaPipe
            this.initializeMediaPipe();
            
            // Initialize camera
            await this.initializeCamera();
            
            // Set canvas size to match video
            this.outputCanvas.width = this.inputVideo.videoWidth || 640;
            this.outputCanvas.height = this.inputVideo.videoHeight || 480;
            
            this.isRunning = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            
            this.log('Gesture recognition started successfully');
            
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

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.gestureApp = new GestureRecognitionApp();
});
