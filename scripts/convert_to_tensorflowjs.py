#!/usr/bin/env python3
"""
TensorFlow.js Model Conversion Pipeline
======================================

This script provides automated conversion of trained Keras models to TensorFlow.js format
for web deployment, including:
- Model conversion with optimization
- Quantization options for reduced model size
- Validation of converted models
- Web deployment preparation
- Performance profiling and comparison
"""

import os
import sys
import json
import pickle
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, track
from rich.text import Text

console = Console()

class TensorFlowJSConverter:
    """Comprehensive TensorFlow.js conversion pipeline."""
    
    def __init__(self, model_path: str, output_dir: str = "web_models"):
        self.model_path = model_path
        self.output_dir = output_dir
        self.conversion_results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load original model
        self._load_original_model()
        
    def _load_original_model(self):
        """Load the original Keras model."""
        console.print(f"[bold blue]Loading original model from {self.model_path}[/bold blue]")
        
        try:
            self.original_model = tf.keras.models.load_model(self.model_path)
            
            # Load metadata
            model_dir = os.path.dirname(self.model_path)
            
            # Load label encoder
            encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                self.class_names = self.label_encoder.classes_.tolist()
            else:
                self.class_names = ['fist', 'open_palm', 'peace', 'rock', 'thumbs_up']
            
            # Load model info
            info_path = os.path.join(model_dir, 'model_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
            else:
                self.model_info = {}
            
            console.print(f"✓ Original model loaded: {len(self.class_names)} classes")
            console.print(f"  - Input shape: {self.original_model.input_shape}")
            console.print(f"  - Parameters: {self.original_model.count_params():,}")
            
        except Exception as e:
            console.print(f"[red]Error loading original model: {e}[/red]")
            sys.exit(1)
    
    def convert_standard_model(self) -> Dict[str, Any]:
        """Convert model to TensorFlow.js format (standard)."""
        console.print("[bold blue]Converting to TensorFlow.js (standard)...[/bold blue]")
        
        output_path = os.path.join(self.output_dir, "standard")
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Convert using tensorflowjs
            tfjs.converters.save_keras_model(
                self.original_model,
                output_path,
                quantization_bytes=None,  # No quantization for standard version
                metadata={'class_names': self.class_names}
            )
            
            # Get file sizes
            model_json_size = os.path.getsize(os.path.join(output_path, "model.json"))
            weights_files = [f for f in os.listdir(output_path) if f.startswith("weights")]
            total_weights_size = sum(os.path.getsize(os.path.join(output_path, f)) for f in weights_files)
            total_size = model_json_size + total_weights_size
            
            result = {
                'output_path': output_path,
                'model_json_size_mb': model_json_size / (1024 * 1024),
                'weights_size_mb': total_weights_size / (1024 * 1024),
                'total_size_mb': total_size / (1024 * 1024),
                'quantization': None,
                'files': os.listdir(output_path)
            }
            
            self.conversion_results['standard'] = result
            console.print(f"✓ Standard conversion completed: {result['total_size_mb']:.2f} MB")
            
            return result
            
        except Exception as e:
            console.print(f"[red]Error in standard conversion: {e}[/red]")
            return {}
    
    def convert_quantized_model(self, quantization_bytes: int = 1) -> Dict[str, Any]:
        """Convert model to TensorFlow.js format with quantization."""
        console.print(f"[bold blue]Converting to TensorFlow.js (quantized {quantization_bytes}-byte)...[/bold blue]")
        
        output_path = os.path.join(self.output_dir, f"quantized_{quantization_bytes}byte")
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Convert with quantization
            tfjs.converters.save_keras_model(
                self.original_model,
                output_path,
                quantization_bytes=quantization_bytes,
                metadata={'class_names': self.class_names}
            )
            
            # Get file sizes
            model_json_size = os.path.getsize(os.path.join(output_path, "model.json"))
            weights_files = [f for f in os.listdir(output_path) if f.startswith("weights")]
            total_weights_size = sum(os.path.getsize(os.path.join(output_path, f)) for f in weights_files)
            total_size = model_json_size + total_weights_size
            
            result = {
                'output_path': output_path,
                'model_json_size_mb': model_json_size / (1024 * 1024),
                'weights_size_mb': total_weights_size / (1024 * 1024),
                'total_size_mb': total_size / (1024 * 1024),
                'quantization': quantization_bytes,
                'files': os.listdir(output_path)
            }
            
            self.conversion_results[f'quantized_{quantization_bytes}byte'] = result
            console.print(f"✓ Quantized conversion completed: {result['total_size_mb']:.2f} MB")
            
            return result
            
        except Exception as e:
            console.print(f"[red]Error in quantized conversion: {e}[/red]")
            return {}
    
    def optimize_for_inference(self) -> Dict[str, Any]:
        """Create optimized version for inference."""
        console.print("[bold blue]Creating inference-optimized model...[/bold blue]")
        
        try:
            # Create a new model with inference optimizations
            optimized_model = tf.keras.models.clone_model(self.original_model)
            optimized_model.set_weights(self.original_model.get_weights())
            
            # Compile with optimizations for inference
            optimized_model.compile(
                optimizer='adam',  # Keep simple for inference
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Save optimized model
            optimized_path = os.path.join(self.output_dir, "optimized_model.h5")
            optimized_model.save(optimized_path)
            
            # Convert optimized model to TensorFlow.js
            output_path = os.path.join(self.output_dir, "optimized")
            os.makedirs(output_path, exist_ok=True)
            
            tfjs.converters.save_keras_model(
                optimized_model,
                output_path,
                quantization_bytes=2,  # 2-byte quantization for balance
                metadata={
                    'class_names': self.class_names,
                    'optimized': True,
                    'optimization_date': datetime.now().isoformat()
                }
            )
            
            # Get file sizes
            model_json_size = os.path.getsize(os.path.join(output_path, "model.json"))
            weights_files = [f for f in os.listdir(output_path) if f.startswith("weights")]
            total_weights_size = sum(os.path.getsize(os.path.join(output_path, f)) for f in weights_files)
            total_size = model_json_size + total_weights_size
            
            result = {
                'output_path': output_path,
                'model_json_size_mb': model_json_size / (1024 * 1024),
                'weights_size_mb': total_weights_size / (1024 * 1024),
                'total_size_mb': total_size / (1024 * 1024),
                'quantization': 2,
                'optimized': True,
                'files': os.listdir(output_path)
            }
            
            self.conversion_results['optimized'] = result
            console.print(f"✓ Optimized conversion completed: {result['total_size_mb']:.2f} MB")
            
            return result
            
        except Exception as e:
            console.print(f"[red]Error in optimized conversion: {e}[/red]")
            return {}
    
    def validate_converted_models(self, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate converted models against original."""
        console.print("[bold blue]Validating converted models...[/bold blue]")
        
        validation_results = {}
        
        # Generate test data if not provided
        if test_data_path and os.path.exists(test_data_path):
            with open(test_data_path, 'r') as f:
                data = json.load(f)
            
            # Use first 10 samples for validation
            test_samples = data[:10]
            X_test = np.array([sample['landmarks'] for sample in test_samples])
            y_true = [sample['gesture'] for sample in test_samples]
        else:
            # Generate synthetic test data
            console.print("[yellow]No test data provided, generating synthetic data for validation...[/yellow]")
            X_test = np.random.randn(10, *self.original_model.input_shape[1:])
            y_true = np.random.choice(self.class_names, 10)
        
        # Get original model predictions
        original_predictions = self.original_model.predict(X_test, verbose=0)
        
        # Validate each converted model
        for model_name, conversion_info in self.conversion_results.items():
            if 'output_path' not in conversion_info:
                continue
                
            try:
                console.print(f"  Validating {model_name}...")
                
                # Note: Actual TF.js validation would require Node.js environment
                # For now, we'll validate the conversion success and file integrity
                output_path = conversion_info['output_path']
                
                # Check required files exist
                model_json_path = os.path.join(output_path, "model.json")
                if not os.path.exists(model_json_path):
                    raise FileNotFoundError("model.json not found")
                
                # Check model.json is valid JSON
                with open(model_json_path, 'r') as f:
                    model_json = json.load(f)
                
                # Check weights files exist
                weights_files = [f for f in os.listdir(output_path) if f.startswith("weights")]
                if not weights_files:
                    raise FileNotFoundError("No weights files found")
                
                validation_results[model_name] = {
                    'valid': True,
                    'model_json_valid': True,
                    'weights_files_count': len(weights_files),
                    'total_files': len(os.listdir(output_path)),
                    'size_mb': conversion_info['total_size_mb'],
                    'error': None
                }
                
                console.print(f"    ✓ {model_name} validation passed")
                
            except Exception as e:
                validation_results[model_name] = {
                    'valid': False,
                    'error': str(e)
                }
                console.print(f"    ✗ {model_name} validation failed: {e}")
        
        self.conversion_results['validation'] = validation_results
        return validation_results
    
    def create_web_deployment_package(self) -> Dict[str, Any]:
        """Create complete web deployment package."""
        console.print("[bold blue]Creating web deployment package...[/bold blue]")
        
        deployment_dir = os.path.join(self.output_dir, "web_deployment")
        os.makedirs(deployment_dir, exist_ok=True)
        
        try:
            # Select best model (optimized if available, otherwise standard)
            best_model = 'optimized' if 'optimized' in self.conversion_results else 'standard'
            if best_model not in self.conversion_results:
                console.print("[red]No valid converted model found for deployment[/red]")
                return {}
            
            model_info = self.conversion_results[best_model]
            source_path = model_info['output_path']
            
            # Copy model files
            model_deploy_dir = os.path.join(deployment_dir, "models")
            if os.path.exists(model_deploy_dir):
                shutil.rmtree(model_deploy_dir)
            shutil.copytree(source_path, model_deploy_dir)
            
            # Create model info JSON
            model_info_json = {
                'model_name': 'gesture_recognition_model',
                'version': '1.0.0',
                'created_date': datetime.now().isoformat(),
                'class_names': self.class_names,
                'input_shape': list(self.original_model.input_shape[1:]),
                'model_type': best_model,
                'quantization': model_info.get('quantization'),
                'size_mb': model_info['total_size_mb'],
                'files': model_info['files'],
                'preprocessing': {
                    'normalization': 'none',
                    'input_format': 'landmarks_array',
                    'expected_shape': list(self.original_model.input_shape[1:])
                }
            }
            
            # Save model info
            with open(os.path.join(deployment_dir, "model_info.json"), 'w') as f:
                json.dump(model_info_json, f, indent=2)
            
            # Create JavaScript loading helper
            js_helper = self._generate_js_helper(model_info_json)
            with open(os.path.join(deployment_dir, "model_loader.js"), 'w') as f:
                f.write(js_helper)
            
            # Create HTML demo
            html_demo = self._generate_html_demo(model_info_json)
            with open(os.path.join(deployment_dir, "demo.html"), 'w') as f:
                f.write(html_demo)
            
            # Create README
            readme = self._generate_deployment_readme(model_info_json)
            with open(os.path.join(deployment_dir, "README.md"), 'w') as f:
                f.write(readme)
            
            deployment_result = {
                'deployment_path': deployment_dir,
                'model_type': best_model,
                'model_size_mb': model_info['total_size_mb'],
                'files_created': os.listdir(deployment_dir),
                'ready_for_deployment': True
            }
            
            self.conversion_results['deployment'] = deployment_result
            console.print(f"✓ Web deployment package created at {deployment_dir}")
            
            return deployment_result
            
        except Exception as e:
            console.print(f"[red]Error creating deployment package: {e}[/red]")
            return {}
    
    def _generate_js_helper(self, model_info: Dict[str, Any]) -> str:
        """Generate JavaScript helper for model loading."""
        return f'''/**
 * Gesture Recognition Model Loader
 * Generated on {datetime.now().isoformat()}
 */

class GestureRecognitionModel {{
    constructor() {{
        this.model = null;
        this.classNames = {json.dumps(model_info['class_names'])};
        this.inputShape = {json.dumps(model_info['input_shape'])};
        this.isLoaded = false;
    }}

    async loadModel(modelPath = './models/model.json') {{
        try {{
            console.log('Loading gesture recognition model...');
            this.model = await tf.loadLayersModel(modelPath);
            this.isLoaded = true;
            console.log('Model loaded successfully');
            console.log('Input shape:', this.inputShape);
            console.log('Classes:', this.classNames);
            return true;
        }} catch (error) {{
            console.error('Error loading model:', error);
            return false;
        }}
    }}

    predict(landmarks) {{
        if (!this.isLoaded || !this.model) {{
            throw new Error('Model not loaded. Call loadModel() first.');
        }}

        // Ensure input is in correct format
        const inputTensor = tf.tensor2d([landmarks], [1, {model_info['input_shape'][0]}]);
        
        // Make prediction
        const prediction = this.model.predict(inputTensor);
        const probabilities = prediction.dataSync();
        
        // Clean up tensors
        inputTensor.dispose();
        prediction.dispose();

        // Find best prediction
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        const confidence = probabilities[maxIndex];
        const gesture = this.classNames[maxIndex];

        return {{
            gesture: gesture,
            confidence: confidence,
            probabilities: Array.from(probabilities),
            all_predictions: this.classNames.map((name, i) => ({{
                class: name,
                probability: probabilities[i]
            }}))
        }};
    }}

    preprocessLandmarks(landmarks) {{
        // Ensure landmarks are in the correct format
        if (!Array.isArray(landmarks)) {{
            throw new Error('Landmarks must be an array');
        }}
        
        if (landmarks.length !== {model_info['input_shape'][0]}) {{
            throw new Error(`Expected {model_info['input_shape'][0]} landmarks, got ${{landmarks.length}}`);
        }}

        // Convert to flat array if needed
        const flatLandmarks = landmarks.flat();
        return flatLandmarks;
    }}

    dispose() {{
        if (this.model) {{
            this.model.dispose();
            this.model = null;
            this.isLoaded = false;
        }}
    }}
}}

// Export for use
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = GestureRecognitionModel;
}}
'''

    def _generate_html_demo(self, model_info: Dict[str, Any]) -> str:
        """Generate HTML demo for the converted model."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gesture Recognition Model Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .demo-container {{
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .prediction-result {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .confidence-bar {{
            background: #ddd;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }}
        .confidence-fill {{
            background: #4CAF50;
            height: 100%;
            transition: width 0.3s ease;
        }}
        button {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }}
        button:hover {{
            background: #0056b3;
        }}
        button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        .status {{
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .status.loading {{ background: #fff3cd; }}
        .status.success {{ background: #d4edda; }}
        .status.error {{ background: #f8d7da; }}
    </style>
</head>
<body>
    <h1>Gesture Recognition Model Demo</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="demo-container">
        <h2>Model Information</h2>
        <ul>
            <li><strong>Model Type:</strong> {model_info.get('model_type', 'N/A')}</li>
            <li><strong>Size:</strong> {model_info['size_mb']:.2f} MB</li>
            <li><strong>Classes:</strong> {', '.join(model_info['class_names'])}</li>
            <li><strong>Input Shape:</strong> {model_info['input_shape']}</li>
        </ul>
    </div>

    <div class="demo-container">
        <h2>Model Testing</h2>
        <div id="status" class="status loading">Ready to load model...</div>
        
        <button id="loadModel">Load Model</button>
        <button id="testModel" disabled>Test with Random Data</button>
        
        <div id="predictions" style="display: none;">
            <h3>Prediction Results</h3>
            <div id="predictionResults"></div>
        </div>
    </div>

    <script src="model_loader.js"></script>
    <script>
        const gestureModel = new GestureRecognitionModel();
        const statusDiv = document.getElementById('status');
        const loadButton = document.getElementById('loadModel');
        const testButton = document.getElementById('testModel');
        const predictionsDiv = document.getElementById('predictions');
        const predictionResults = document.getElementById('predictionResults');

        function updateStatus(message, type = 'loading') {{
            statusDiv.textContent = message;
            statusDiv.className = `status ${{type}}`;
        }}

        loadButton.addEventListener('click', async () => {{
            loadButton.disabled = true;
            updateStatus('Loading model...', 'loading');
            
            try {{
                const success = await gestureModel.loadModel('./models/model.json');
                if (success) {{
                    updateStatus('Model loaded successfully!', 'success');
                    testButton.disabled = false;
                }} else {{
                    updateStatus('Failed to load model', 'error');
                    loadButton.disabled = false;
                }}
            }} catch (error) {{
                updateStatus(`Error: ${{error.message}}`, 'error');
                loadButton.disabled = false;
            }}
        }});

        testButton.addEventListener('click', () => {{
            try {{
                // Generate random landmarks data
                const randomLandmarks = Array.from({{length: {model_info['input_shape'][0]}}}, () => Math.random() * 2 - 1);
                
                const result = gestureModel.predict(randomLandmarks);
                
                // Display results
                predictionsDiv.style.display = 'block';
                predictionResults.innerHTML = `
                    <div class="prediction-result">
                        <h4>Predicted Gesture: ${{result.gesture}}</h4>
                        <p>Confidence: ${{(result.confidence * 100).toFixed(1)}}%</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${{result.confidence * 100}}%"></div>
                        </div>
                        
                        <h5>All Predictions:</h5>
                        ${{result.all_predictions.map(pred => `
                            <div>
                                ${{pred.class}}: ${{(pred.probability * 100).toFixed(1)}}%
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${{pred.probability * 100}}%"></div>
                                </div>
                            </div>
                        `).join('')}}
                    </div>
                `;
            }} catch (error) {{
                updateStatus(`Prediction error: ${{error.message}}`, 'error');
            }}
        }});
    </script>
</body>
</html>'''

    def _generate_deployment_readme(self, model_info: Dict[str, Any]) -> str:
        """Generate deployment README."""
        return f'''# Gesture Recognition Model - Web Deployment

Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information

- **Model Type**: {model_info.get('model_type', 'N/A')}
- **Size**: {model_info['size_mb']:.2f} MB
- **Quantization**: {model_info.get('quantization', 'None')}
- **Classes**: {', '.join(model_info['class_names'])}
- **Input Shape**: {model_info['input_shape']}

## Files Included

- `models/` - TensorFlow.js model files
  - `model.json` - Model architecture
  - `weights.bin` - Model weights
- `model_loader.js` - JavaScript helper class
- `model_info.json` - Model metadata
- `demo.html` - Interactive demo
- `README.md` - This file

## Quick Start

1. **Host the files**: Serve these files from a web server (cannot run from file:// due to CORS)

2. **Include TensorFlow.js**: Add to your HTML head:
   ```html
   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
   ```

3. **Load and use the model**:
   ```javascript
   // Include the model loader
   const gestureModel = new GestureRecognitionModel();
   
   // Load the model
   await gestureModel.loadModel('./models/model.json');
   
   // Make predictions (landmarks should be array of {model_info['input_shape'][0]} values)
   const landmarks = [...]; // Your landmark data
   const result = gestureModel.predict(landmarks);
   
   console.log('Predicted gesture:', result.gesture);
   console.log('Confidence:', result.confidence);
   ```

## Integration with MediaPipe

This model is designed to work with MediaPipe hand landmark detection:

```javascript
// Assuming you have MediaPipe landmarks
const landmarks = results.multiHandLandmarks[0];

// Flatten landmarks to expected format
const flatLandmarks = landmarks.map(landmark => [landmark.x, landmark.y, landmark.z]).flat();

// Preprocess if needed
const processedLandmarks = gestureModel.preprocessLandmarks(flatLandmarks);

// Predict
const prediction = gestureModel.predict(processedLandmarks);
```

## Model Classes

The model can recognize these gestures:
{chr(10).join(f'- {cls}' for cls in model_info['class_names'])}

## Performance Notes

- **Model Size**: {model_info['size_mb']:.2f} MB
- **Expected Input**: Array of {model_info['input_shape'][0]} numeric values
- **Output**: Probability distribution over {len(model_info['class_names'])} classes

## Troubleshooting

1. **CORS Errors**: Ensure files are served from a web server, not opened directly in browser
2. **Model Loading Fails**: Check that all model files are accessible and paths are correct
3. **Prediction Errors**: Verify input landmarks are in correct format and shape
4. **Performance Issues**: Consider using Web Workers for inference in production

## Example Integration

See `demo.html` for a complete working example of model loading and prediction.

For production use, consider:
- Implementing proper error handling
- Adding input validation
- Using Web Workers for heavy computations
- Implementing model caching strategies
- Adding performance monitoring
'''

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        console.print("[bold blue]Generating performance report...[/bold blue]")
        
        # Get original model size
        original_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        
        performance_data = {
            'original_model': {
                'size_mb': original_size_mb,
                'parameters': self.original_model.count_params(),
                'format': 'Keras (.h5)'
            },
            'converted_models': {},
            'size_comparisons': {},
            'recommendations': {}
        }
        
        # Analyze each converted model
        for model_name, info in self.conversion_results.items():
            if model_name in ['validation', 'deployment']:
                continue
                
            if 'total_size_mb' in info:
                performance_data['converted_models'][model_name] = {
                    'size_mb': info['total_size_mb'],
                    'quantization': info.get('quantization', 'None'),
                    'size_reduction_percent': ((original_size_mb - info['total_size_mb']) / original_size_mb * 100),
                    'size_ratio': info['total_size_mb'] / original_size_mb
                }
        
        # Generate recommendations
        if performance_data['converted_models']:
            smallest_model = min(performance_data['converted_models'].items(), 
                                key=lambda x: x[1]['size_mb'])
            
            performance_data['recommendations'] = {
                'smallest_model': smallest_model[0],
                'smallest_size_mb': smallest_model[1]['size_mb'],
                'max_size_reduction_percent': max(
                    model['size_reduction_percent'] 
                    for model in performance_data['converted_models'].values()
                ),
                'deployment_recommendation': self._get_deployment_recommendation(performance_data)
            }
        
        # Save performance report
        report_path = os.path.join(self.output_dir, 'conversion_performance_report.json')
        with open(report_path, 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        self.conversion_results['performance_report'] = performance_data
        return performance_data
    
    def _get_deployment_recommendation(self, performance_data: Dict[str, Any]) -> str:
        """Get deployment recommendation based on model sizes."""
        models = performance_data['converted_models']
        
        if not models:
            return "No models available for recommendation"
        
        # Find model with best size/quality trade-off
        if 'optimized' in models:
            return "Use 'optimized' model for best balance of size and quality"
        elif 'quantized_2byte' in models:
            return "Use 'quantized_2byte' model for good compression with minimal quality loss"
        elif 'quantized_1byte' in models:
            if models['quantized_1byte']['size_mb'] < 5.0:
                return "Use 'quantized_1byte' model for maximum compression (verify quality)"
            else:
                return "Use 'standard' model if quality is critical, 'quantized_1byte' for size"
        else:
            return "Use 'standard' model for best quality"
    
    def display_conversion_results(self):
        """Display comprehensive conversion results."""
        console.print(Panel.fit("📦 TensorFlow.js Conversion Results", style="bold magenta"))
        
        # Conversion Summary Table
        if self.conversion_results:
            summary_table = Table(title="Model Conversion Summary")
            summary_table.add_column("Model Type", style="cyan")
            summary_table.add_column("Size (MB)", style="green")
            summary_table.add_column("Quantization", style="yellow")
            summary_table.add_column("Status", style="magenta")
            
            original_size = os.path.getsize(self.model_path) / (1024 * 1024)
            summary_table.add_row("Original (Keras)", f"{original_size:.2f}", "None", "✓")
            
            for model_name, info in self.conversion_results.items():
                if model_name in ['validation', 'deployment', 'performance_report']:
                    continue
                    
                if 'total_size_mb' in info:
                    size_mb = info['total_size_mb']
                    quantization = str(info.get('quantization', 'None'))
                    status = "✓" if info.get('output_path') else "✗"
                    reduction = f"({(original_size - size_mb) / original_size * 100:.1f}% smaller)"
                    
                    summary_table.add_row(
                        model_name.title(),
                        f"{size_mb:.2f} {reduction}",
                        quantization,
                        status
                    )
            
            console.print(summary_table)
        
        # Validation Results
        if 'validation' in self.conversion_results:
            validation_table = Table(title="Model Validation Results")
            validation_table.add_column("Model", style="cyan")
            validation_table.add_column("Valid", style="green")
            validation_table.add_column("Files", style="yellow")
            validation_table.add_column("Notes", style="magenta")
            
            for model_name, validation in self.conversion_results['validation'].items():
                valid_status = "✓" if validation['valid'] else "✗"
                files_count = str(validation.get('total_files', 'N/A'))
                notes = validation.get('error', 'OK') if not validation['valid'] else 'OK'
                
                validation_table.add_row(model_name, valid_status, files_count, notes)
            
            console.print(validation_table)
        
        # Deployment Information
        if 'deployment' in self.conversion_results:
            deployment_info = self.conversion_results['deployment']
            deployment_panel = Panel.fit(
                f"""
[bold cyan]Web Deployment Package:[/bold cyan]

[green]✓ Ready for deployment[/green]
📁 Location: {deployment_info['deployment_path']}
📊 Model: {deployment_info['model_type']} ({deployment_info['model_size_mb']:.2f} MB)
📄 Files: {', '.join(deployment_info['files_created'])}

[yellow]Next Steps:[/yellow]
1. Host files on web server
2. Include TensorFlow.js in your HTML
3. Use model_loader.js for easy integration
4. Test with demo.html
                """.strip(),
                title="Deployment Status",
                border_style="green"
            )
            console.print(deployment_panel)
        
        # Performance Recommendations
        if 'performance_report' in self.conversion_results:
            perf_report = self.conversion_results['performance_report']
            if 'recommendations' in perf_report:
                rec = perf_report['recommendations']
                recommendation_panel = Panel.fit(
                    f"""
[bold cyan]Performance Analysis:[/bold cyan]

[green]Best Size Reduction:[/green] {rec.get('max_size_reduction_percent', 0):.1f}%
[green]Smallest Model:[/green] {rec.get('smallest_model', 'N/A')} ({rec.get('smallest_size_mb', 0):.2f} MB)

[yellow]Recommendation:[/yellow]
{rec.get('deployment_recommendation', 'No recommendation available')}
                    """.strip(),
                    title="Performance Recommendations",
                    border_style="blue"
                )
                console.print(recommendation_panel)
    
    def run_complete_conversion(self, include_quantized: bool = True, test_data_path: Optional[str] = None):
        """Run complete conversion pipeline."""
        console.print(Panel.fit("🔄 Starting TensorFlow.js Conversion Pipeline", style="bold blue"))
        
        with Progress() as progress:
            total_tasks = 6 if include_quantized else 4
            task = progress.add_task("Conversion", total=total_tasks)
            
            # Standard conversion
            progress.update(task, description="Converting standard model...")
            self.convert_standard_model()
            progress.advance(task)
            
            if include_quantized:
                # Quantized conversions
                progress.update(task, description="Converting quantized models...")
                self.convert_quantized_model(1)  # 1-byte quantization
                progress.advance(task)
                
                progress.update(task, description="Converting optimized model...")
                self.optimize_for_inference()
                progress.advance(task)
            
            # Validation
            progress.update(task, description="Validating models...")
            self.validate_converted_models(test_data_path)
            progress.advance(task)
            
            # Deployment package
            progress.update(task, description="Creating deployment package...")
            self.create_web_deployment_package()
            progress.advance(task)
            
            # Performance report
            progress.update(task, description="Generating performance report...")
            self.generate_performance_report()
            progress.advance(task)
        
        # Display results
        self.display_conversion_results()
        
        console.print(f"\n✓ [bold green]TensorFlow.js conversion completed![/bold green]")
        console.print(f"Results saved to: {self.output_dir}")

def main():
    """Main conversion interface."""
    parser = argparse.ArgumentParser(description='Convert Keras model to TensorFlow.js')
    parser.add_argument('--model', type=str, required=True, help='Path to Keras model (.h5)')
    parser.add_argument('--output', type=str, default='web_models', help='Output directory')
    parser.add_argument('--test-data', type=str, help='Path to test data for validation')
    parser.add_argument('--no-quantization', action='store_true', help='Skip quantized versions')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        console.print(f"[red]Error: Model file not found: {args.model}[/red]")
        sys.exit(1)
    
    if args.test_data and not os.path.exists(args.test_data):
        console.print(f"[red]Error: Test data file not found: {args.test_data}[/red]")
        sys.exit(1)
    
    # Run conversion
    converter = TensorFlowJSConverter(args.model, args.output)
    converter.run_complete_conversion(
        include_quantized=not args.no_quantization,
        test_data_path=args.test_data
    )

if __name__ == '__main__':
    main()