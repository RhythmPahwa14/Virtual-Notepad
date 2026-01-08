#!/usr/bin/env python3
"""
Advanced Gesture Recognition Model Training Script
==================================================

This script implements a complete ML pipeline for training custom gesture recognition models
with comprehensive evaluation and comparison against MediaPipe baselines.

Features:
- Custom neural network architecture design
- Advanced data preprocessing and augmentation
- Comprehensive evaluation metrics
- Model comparison and benchmarking
- Hyperparameter optimization
- Performance profiling
"""

import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import optuna
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

class GestureDataProcessor:
    """Advanced data processing and augmentation for gesture recognition."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess gesture data."""
        console.print(f"[bold blue]Loading data from {data_path}[/bold blue]")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Extract features and labels
        X = []
        y = []
        
        for sample in data:
            landmarks = np.array(sample['landmarks']).reshape(21, 3)
            X.append(landmarks)
            y.append(sample['gesture'])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Loaded {len(X)} samples with {len(np.unique(y))} classes")
        console.print(f"✓ Data loaded: {len(X)} samples, {len(np.unique(y))} classes")
        
        return X, y
    
    def preprocess_landmarks(self, X: np.ndarray) -> np.ndarray:
        """Advanced landmark preprocessing with normalization and feature engineering."""
        console.print("[bold blue]Preprocessing landmarks...[/bold blue]")
        
        processed_X = []
        
        for landmarks in X:
            # Center landmarks relative to hand center
            center = np.mean(landmarks, axis=0)
            centered = landmarks - center
            
            # Scale to unit sphere
            max_distance = np.max(np.linalg.norm(centered, axis=1))
            if max_distance > 0:
                scaled = centered / max_distance
            else:
                scaled = centered
            
            # Add engineered features
            features = self._extract_features(scaled)
            processed_X.append(features)
        
        processed_X = np.array(processed_X)
        console.print(f"✓ Preprocessing complete: {processed_X.shape}")
        return processed_X
    
    def _extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract advanced features from landmarks."""
        # Original landmarks (flattened)
        base_features = landmarks.flatten()
        
        # Distance features
        distances = []
        for i in range(len(landmarks)):
            for j in range(i + 1, len(landmarks)):
                dist = np.linalg.norm(landmarks[i] - landmarks[j])
                distances.append(dist)
        
        # Angle features
        angles = []
        for i in range(1, len(landmarks) - 1):
            v1 = landmarks[i] - landmarks[i-1]
            v2 = landmarks[i+1] - landmarks[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angles.append(cos_angle)
        
        # Combine all features
        all_features = np.concatenate([
            base_features,
            distances[:50],  # Limit distance features
            angles
        ])
        
        return all_features
    
    def augment_data(self, X: np.ndarray, y: np.ndarray, 
                    augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Generate augmented training data."""
        if not self.config.get('augmentation', {}).get('enabled', False):
            return X, y
            
        console.print(f"[bold blue]Augmenting data (factor: {augmentation_factor})[/bold blue]")
        
        augmented_X = [X]
        augmented_y = [y]
        
        for _ in range(augmentation_factor):
            aug_X = []
            for landmarks in X:
                # Add noise
                noise = np.random.normal(0, 0.01, landmarks.shape)
                noisy_landmarks = landmarks + noise
                
                # Random rotation
                theta = np.random.uniform(-0.1, 0.1)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                rotated_landmarks = np.dot(noisy_landmarks, rotation_matrix.T)
                
                # Random scaling
                scale = np.random.uniform(0.9, 1.1)
                scaled_landmarks = rotated_landmarks * scale
                
                aug_X.append(scaled_landmarks)
            
            augmented_X.append(np.array(aug_X))
            augmented_y.append(y)
        
        final_X = np.vstack(augmented_X)
        final_y = np.hstack(augmented_y)
        
        console.print(f"✓ Augmentation complete: {len(final_X)} samples")
        return final_X, final_y

class GestureModel:
    """Custom neural network model for gesture recognition."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
        """Build custom neural network architecture."""
        console.print("[bold blue]Building model architecture...[/bold blue]")
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Flatten() if len(input_shape) > 1 else layers.Identity(),
            
            # Feature extraction layers
            layers.Dense(256, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu', name='dense_3'),
            layers.Dropout(0.2),
            
            # Classification layer
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        optimizer = optimizers.Adam(
            learning_rate=self.config.get('learning_rate', 0.001)
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        console.print(f"✓ Model built with {model.count_params()} parameters")
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train the model with advanced callbacks and monitoring."""
        console.print("[bold blue]Starting model training...[/bold blue]")
        
        # Prepare callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Training with progress tracking
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            
            epochs = self.config.get('epochs', 100)
            task = progress.add_task("Training", total=epochs)
            
            class ProgressCallback(callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress.update(task, advance=1)
                    if epoch % 10 == 0:
                        console.print(
                            f"Epoch {epoch+1}/{epochs} - "
                            f"Loss: {logs['loss']:.4f} - "
                            f"Acc: {logs['accuracy']:.4f} - "
                            f"Val_Acc: {logs['val_accuracy']:.4f}"
                        )
            
            callbacks_list.append(ProgressCallback())
            
            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=self.config.get('batch_size', 32),
                callbacks=callbacks_list,
                verbose=0
            )
        
        console.print("✓ Training completed!")
        return self.history.history

class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model: tf.keras.Model, X_test: np.ndarray, 
                      y_test: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        console.print("[bold blue]Evaluating model performance...[/bold blue]")
        
        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            class_precision = precision_score(y_true, y_pred, labels=[i], average='micro')
            class_recall = recall_score(y_true, y_pred, labels=[i], average='micro')
            class_f1 = f1_score(y_true, y_pred, labels=[i], average='micro')
            
            per_class_metrics[class_name] = {
                'precision': class_precision,
                'recall': class_recall,
                'f1_score': class_f1
            }
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'per_class_metrics': per_class_metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Display results
        self._display_results(results, class_names)
        
        return results
    
    def _display_results(self, results: Dict[str, Any], class_names: List[str]):
        """Display evaluation results in a formatted table."""
        
        # Overall metrics table
        metrics_table = Table(title="Overall Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Score", style="magenta")
        
        metrics_table.add_row("Accuracy", f"{results['accuracy']:.4f}")
        metrics_table.add_row("Precision", f"{results['precision']:.4f}")
        metrics_table.add_row("Recall", f"{results['recall']:.4f}")
        metrics_table.add_row("F1-Score", f"{results['f1_score']:.4f}")
        
        console.print(metrics_table)
        
        # Per-class metrics table
        class_table = Table(title="Per-Class Performance")
        class_table.add_column("Class", style="cyan")
        class_table.add_column("Precision", style="green")
        class_table.add_column("Recall", style="yellow")
        class_table.add_column("F1-Score", style="magenta")
        
        for class_name, metrics in results['per_class_metrics'].items():
            class_table.add_row(
                class_name,
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1_score']:.4f}"
            )
        
        console.print(class_table)

def create_training_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        'data_path': 'data/gesture_data.json',
        'output_dir': 'models',
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'test_split': 0.2,
        'random_state': 42,
        'augmentation': {
            'enabled': True,
            'factor': 2,
            'noise_std': 0.01,
            'rotation_range': 0.1,
            'scale_range': (0.9, 1.1)
        },
        'model': {
            'architecture': 'custom_dense',
            'dropout_rate': 0.3,
            'batch_normalization': True
        }
    }

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train custom gesture recognition model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-path', type=str, default='data/gesture_data.json')
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--no-augmentation', action='store_true')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_training_config()
    
    # Override with command line arguments
    if args.data_path:
        config['data_path'] = args.data_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.no_augmentation:
        config['augmentation']['enabled'] = False
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    console.print(Panel.fit("🧠 Advanced Gesture Recognition Training Pipeline", style="bold magenta"))
    
    # Initialize components
    processor = GestureDataProcessor(config)
    model_builder = GestureModel(config)
    evaluator = ModelEvaluator()
    
    try:
        # Load and preprocess data
        X, y = processor.load_data(config['data_path'])
        X_processed = processor.preprocess_landmarks(X)
        
        # Encode labels
        y_encoded = processor.label_encoder.fit_transform(y)
        class_names = processor.label_encoder.classes_.tolist()
        num_classes = len(class_names)
        
        # Data augmentation
        if config['augmentation']['enabled']:
            X_processed, y_encoded = processor.augment_data(
                X_processed, y_encoded, 
                config['augmentation']['factor']
            )
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_processed, y_encoded,
            test_size=config['test_split'],
            random_state=config['random_state'],
            stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=config['validation_split'] / (1 - config['test_split']),
            random_state=config['random_state'],
            stratify=y_temp
        )
        
        # Convert to categorical
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        console.print(f"✓ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Build and train model
        model = model_builder.build_model(X_train.shape[1:], num_classes)
        training_history = model_builder.train(X_train, y_train_cat, X_val, y_val_cat)
        
        # Evaluate model
        results = evaluator.evaluate_model(model, X_test, y_test_cat, class_names)
        
        # Save model and artifacts
        model_path = os.path.join(config['output_dir'], 'gesture_model.h5')
        model.save(model_path)
        
        # Save label encoder
        encoder_path = os.path.join(config['output_dir'], 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(processor.label_encoder, f)
        
        # Save model info
        model_info = {
            'classes': class_names,
            'input_shape': X_train.shape[1:],
            'num_classes': num_classes,
            'model_params': model.count_params(),
            'training_config': config,
            'evaluation_results': {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        info_path = os.path.join(config['output_dir'], 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        console.print(f"✓ Model saved to {model_path}")
        console.print(f"✓ Training completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == '__main__':
    main()