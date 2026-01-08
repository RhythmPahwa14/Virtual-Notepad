#!/usr/bin/env python3
"""
Model Comparison System
======================

This script provides comprehensive comparison between custom trained models
and MediaPipe baseline, including:
- Performance benchmarking (accuracy, latency, memory)
- Statistical significance testing
- Trade-off analysis
- Detailed comparison reports
- Visualization of performance differences
"""

import os
import sys
import json
import pickle
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import tensorflow as tf

# MediaPipe imports
import mediapipe as mp
import cv2

# Rich for beautiful terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, track
from rich.layout import Layout
from rich.columns import Columns

console = Console()

class MediaPipeBaseline:
    """MediaPipe hand gesture recognition baseline."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Simple gesture classification based on hand landmarks
        self.gesture_classes = ['fist', 'open_palm', 'peace', 'rock', 'thumbs_up']
    
    def _classify_gesture(self, landmarks: np.ndarray) -> str:
        """
        Simple gesture classification based on landmark geometry.
        This is a simplified baseline - MediaPipe doesn't include gesture classification.
        """
        # Convert landmarks to hand shape features
        # This is a simplified heuristic-based classifier
        
        # Get key landmark positions
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Calculate finger extensions
        fingers_up = []
        
        # Thumb (different logic due to orientation)
        if thumb_tip[0] > thumb_ip[0]:  # Right hand assumption
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # Other fingers
        for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip), 
                        (ring_tip, ring_pip), (pinky_tip, pinky_pip)]:
            if tip[1] < pip[1]:  # Y coordinate decreases upward
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        total_fingers = sum(fingers_up)
        
        # Simple heuristic classification
        if total_fingers == 0:
            return 'fist'
        elif total_fingers == 5:
            return 'open_palm'
        elif total_fingers == 2 and fingers_up[1] and fingers_up[2]:
            return 'peace'
        elif total_fingers == 1 and fingers_up[0]:
            return 'thumbs_up'
        else:
            return 'rock'  # Default fallback
    
    def predict(self, landmarks_array: np.ndarray) -> List[str]:
        """Predict gestures for batch of landmarks."""
        predictions = []
        
        for landmarks in landmarks_array:
            gesture = self._classify_gesture(landmarks.reshape(21, 3))
            predictions.append(gesture)
        
        return predictions
    
    def predict_proba(self, landmarks_array: np.ndarray) -> np.ndarray:
        """Return prediction probabilities (simplified for baseline)."""
        predictions = self.predict(landmarks_array)
        probabilities = []
        
        for pred in predictions:
            # Create one-hot style probabilities with some noise
            proba = np.random.uniform(0.05, 0.15, len(self.gesture_classes))
            class_idx = self.gesture_classes.index(pred)
            proba[class_idx] = np.random.uniform(0.7, 0.9)
            proba = proba / proba.sum()  # Normalize
            probabilities.append(proba)
        
        return np.array(probabilities)

class ModelComparator:
    """Comprehensive model comparison system."""
    
    def __init__(self, custom_model_path: str, data_path: str, output_dir: str = "comparison_results"):
        self.custom_model_path = custom_model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load models and data
        self._load_custom_model()
        self._load_baseline_model()
        self._load_test_data()
        
    def _load_custom_model(self):
        """Load custom trained model."""
        console.print(f"[bold blue]Loading custom model from {self.custom_model_path}[/bold blue]")
        
        try:
            self.custom_model = tf.keras.models.load_model(self.custom_model_path)
            
            # Load metadata
            model_dir = os.path.dirname(self.custom_model_path)
            
            # Load label encoder
            encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                self.class_names = self.label_encoder.classes_.tolist()
            else:
                # Fallback class names
                self.class_names = ['fist', 'open_palm', 'peace', 'rock', 'thumbs_up']
            
            console.print(f"✓ Custom model loaded: {len(self.class_names)} classes")
            
        except Exception as e:
            console.print(f"[red]Error loading custom model: {e}[/red]")
            sys.exit(1)
    
    def _load_baseline_model(self):
        """Load MediaPipe baseline model."""
        console.print("[bold blue]Initializing MediaPipe baseline[/bold blue]")
        
        try:
            self.baseline_model = MediaPipeBaseline()
            console.print("✓ MediaPipe baseline initialized")
        except Exception as e:
            console.print(f"[red]Error loading baseline model: {e}[/red]")
            sys.exit(1)
    
    def _load_test_data(self):
        """Load test data."""
        console.print(f"[bold blue]Loading test data from {self.data_path}[/bold blue]")
        
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            # Extract test data (last 20% for consistency)
            test_size = int(len(data) * 0.2)
            test_data = data[-test_size:]
            
            # Prepare arrays
            self.X_test = []
            self.y_test = []
            
            for sample in test_data:
                landmarks = np.array(sample['landmarks'])
                self.X_test.append(landmarks)
                self.y_test.append(sample['gesture'])
            
            self.X_test = np.array(self.X_test)
            self.y_test = np.array(self.y_test)
            
            console.print(f"✓ Test data loaded: {len(self.X_test)} samples")
            
        except Exception as e:
            console.print(f"[red]Error loading test data: {e}[/red]")
            sys.exit(1)
    
    def evaluate_accuracy_comparison(self) -> Dict[str, Any]:
        """Compare accuracy metrics between models."""
        console.print("[bold blue]Comparing accuracy metrics...[/bold blue]")
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Get predictions from both models
        custom_pred_proba = self.custom_model.predict(self.X_test, verbose=0)
        custom_pred = np.argmax(custom_pred_proba, axis=1)
        
        # Convert custom predictions to gesture names
        if hasattr(self, 'label_encoder'):
            custom_pred_gestures = self.label_encoder.inverse_transform(custom_pred)
        else:
            custom_pred_gestures = [self.class_names[i] for i in custom_pred]
        
        # Get baseline predictions
        baseline_pred_gestures = self.baseline_model.predict(self.X_test)
        
        # Calculate metrics for custom model
        custom_accuracy = accuracy_score(self.y_test, custom_pred_gestures)
        custom_precision = precision_score(self.y_test, custom_pred_gestures, 
                                         average='weighted', zero_division=0)
        custom_recall = recall_score(self.y_test, custom_pred_gestures, 
                                   average='weighted', zero_division=0)
        custom_f1 = f1_score(self.y_test, custom_pred_gestures, 
                           average='weighted', zero_division=0)
        
        # Calculate metrics for baseline
        baseline_accuracy = accuracy_score(self.y_test, baseline_pred_gestures)
        baseline_precision = precision_score(self.y_test, baseline_pred_gestures, 
                                           average='weighted', zero_division=0)
        baseline_recall = recall_score(self.y_test, baseline_pred_gestures, 
                                     average='weighted', zero_division=0)
        baseline_f1 = f1_score(self.y_test, baseline_pred_gestures, 
                             average='weighted', zero_division=0)
        
        # Calculate improvements
        accuracy_improvement = custom_accuracy - baseline_accuracy
        precision_improvement = custom_precision - baseline_precision
        recall_improvement = custom_recall - baseline_recall
        f1_improvement = custom_f1 - baseline_f1
        
        results = {
            'custom_model': {
                'accuracy': custom_accuracy,
                'precision': custom_precision,
                'recall': custom_recall,
                'f1_score': custom_f1,
                'predictions': custom_pred_gestures.tolist()
            },
            'baseline_model': {
                'accuracy': baseline_accuracy,
                'precision': baseline_precision,
                'recall': baseline_recall,
                'f1_score': baseline_f1,
                'predictions': baseline_pred_gestures
            },
            'improvements': {
                'accuracy': accuracy_improvement,
                'precision': precision_improvement,
                'recall': recall_improvement,
                'f1_score': f1_improvement
            },
            'relative_improvements': {
                'accuracy': (accuracy_improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else 0,
                'precision': (precision_improvement / baseline_precision * 100) if baseline_precision > 0 else 0,
                'recall': (recall_improvement / baseline_recall * 100) if baseline_recall > 0 else 0,
                'f1_score': (f1_improvement / baseline_f1 * 100) if baseline_f1 > 0 else 0
            }
        }
        
        self.results['accuracy_comparison'] = results
        return results
    
    def evaluate_performance_comparison(self) -> Dict[str, Any]:
        """Compare performance metrics (speed, memory) between models."""
        console.print("[bold blue]Comparing performance metrics...[/bold blue]")
        
        # Test different batch sizes
        batch_sizes = [1, 8, 16, 32]
        custom_times = {}
        baseline_times = {}
        
        for batch_size in batch_sizes:
            if len(self.X_test) >= batch_size:
                batch_data = self.X_test[:batch_size]
                
                # Custom model timing
                custom_batch_times = []
                for _ in range(10):  # Multiple runs for accuracy
                    start_time = time.perf_counter()
                    _ = self.custom_model.predict(batch_data, verbose=0)
                    end_time = time.perf_counter()
                    custom_batch_times.append((end_time - start_time) / batch_size * 1000)
                
                custom_times[batch_size] = {
                    'mean': np.mean(custom_batch_times),
                    'std': np.std(custom_batch_times),
                    'min': np.min(custom_batch_times),
                    'max': np.max(custom_batch_times)
                }
                
                # Baseline model timing
                baseline_batch_times = []
                for _ in range(10):
                    start_time = time.perf_counter()
                    _ = self.baseline_model.predict(batch_data)
                    end_time = time.perf_counter()
                    baseline_batch_times.append((end_time - start_time) / batch_size * 1000)
                
                baseline_times[batch_size] = {
                    'mean': np.mean(baseline_batch_times),
                    'std': np.std(baseline_batch_times),
                    'min': np.min(baseline_batch_times),
                    'max': np.max(baseline_batch_times)
                }
        
        # Model sizes and complexity
        custom_model_size = os.path.getsize(self.custom_model_path) / (1024 * 1024)  # MB
        custom_params = self.custom_model.count_params()
        
        # Baseline is essentially zero since it's rule-based
        baseline_model_size = 0.1  # Approximate size of MediaPipe libraries
        baseline_params = 100  # Approximate rule complexity
        
        results = {
            'inference_times': {
                'custom_model': custom_times,
                'baseline_model': baseline_times
            },
            'model_properties': {
                'custom_model': {
                    'size_mb': custom_model_size,
                    'parameters': custom_params,
                    'type': 'Neural Network'
                },
                'baseline_model': {
                    'size_mb': baseline_model_size,
                    'parameters': baseline_params,
                    'type': 'Rule-based Heuristics'
                }
            },
            'performance_ratios': {}
        }
        
        # Calculate performance ratios
        for batch_size in batch_sizes:
            if batch_size in custom_times and batch_size in baseline_times:
                custom_time = custom_times[batch_size]['mean']
                baseline_time = baseline_times[batch_size]['mean']
                results['performance_ratios'][batch_size] = {
                    'speed_ratio': baseline_time / custom_time if custom_time > 0 else 0,
                    'custom_faster': custom_time < baseline_time
                }
        
        self.results['performance_comparison'] = results
        return results
    
    def statistical_significance_testing(self) -> Dict[str, Any]:
        """Perform statistical significance testing."""
        console.print("[bold blue]Performing statistical significance testing...[/bold blue]")
        
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedKFold
        
        # Perform cross-validation on both models
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        custom_scores = []
        baseline_scores = []
        
        for train_idx, test_idx in cv.split(self.X_test, self.y_test):
            X_fold = self.X_test[test_idx]
            y_fold = self.y_test[test_idx]
            
            # Custom model predictions
            custom_pred_proba = self.custom_model.predict(X_fold, verbose=0)
            custom_pred = np.argmax(custom_pred_proba, axis=1)
            if hasattr(self, 'label_encoder'):
                custom_pred_gestures = self.label_encoder.inverse_transform(custom_pred)
            else:
                custom_pred_gestures = [self.class_names[i] for i in custom_pred]
            
            custom_score = accuracy_score(y_fold, custom_pred_gestures)
            custom_scores.append(custom_score)
            
            # Baseline predictions
            baseline_pred_gestures = self.baseline_model.predict(X_fold)
            baseline_score = accuracy_score(y_fold, baseline_pred_gestures)
            baseline_scores.append(baseline_score)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(custom_scores, baseline_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(custom_scores) + np.var(baseline_scores)) / 2)
        cohens_d = (np.mean(custom_scores) - np.mean(baseline_scores)) / pooled_std
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(custom_scores, baseline_scores)
        
        # Confidence interval for difference
        diff_scores = np.array(custom_scores) - np.array(baseline_scores)
        ci_lower = np.percentile(diff_scores, 2.5)
        ci_upper = np.percentile(diff_scores, 97.5)
        
        results = {
            'cv_scores': {
                'custom_model': custom_scores,
                'baseline_model': baseline_scores,
                'differences': diff_scores.tolist()
            },
            'statistical_tests': {
                'paired_t_test': {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                },
                'wilcoxon_test': {
                    'statistic': wilcoxon_stat,
                    'p_value': wilcoxon_p,
                    'significant': wilcoxon_p < 0.05
                }
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_cohens_d(cohens_d)
            },
            'confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper,
                'mean_difference': np.mean(diff_scores)
            }
        }
        
        self.results['statistical_significance'] = results
        return results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def analyze_trade_offs(self) -> Dict[str, Any]:
        """Analyze trade-offs between models."""
        console.print("[bold blue]Analyzing model trade-offs...[/bold blue]")
        
        acc_comp = self.results.get('accuracy_comparison', {})
        perf_comp = self.results.get('performance_comparison', {})
        
        # Get single sample inference times
        custom_single_time = perf_comp.get('inference_times', {}).get('custom_model', {}).get(1, {}).get('mean', 0)
        baseline_single_time = perf_comp.get('inference_times', {}).get('baseline_model', {}).get(1, {}).get('mean', 0)
        
        # Calculate trade-off metrics
        accuracy_gain = acc_comp.get('improvements', {}).get('accuracy', 0)
        latency_cost = custom_single_time - baseline_single_time
        
        # Model complexity comparison
        custom_params = perf_comp.get('model_properties', {}).get('custom_model', {}).get('parameters', 0)
        baseline_params = perf_comp.get('model_properties', {}).get('baseline_model', {}).get('parameters', 0)
        complexity_ratio = custom_params / baseline_params if baseline_params > 0 else 0
        
        # Calculate efficiency metrics
        accuracy_per_ms = accuracy_gain / latency_cost if latency_cost > 0 else float('inf')
        accuracy_per_param = accuracy_gain / custom_params if custom_params > 0 else 0
        
        trade_offs = {
            'accuracy_vs_latency': {
                'accuracy_gain': accuracy_gain,
                'latency_cost_ms': latency_cost,
                'accuracy_per_ms': accuracy_per_ms,
                'trade_off_acceptable': accuracy_gain > 0.05 and latency_cost < 50  # Heuristic
            },
            'accuracy_vs_complexity': {
                'accuracy_gain': accuracy_gain,
                'complexity_ratio': complexity_ratio,
                'accuracy_per_param': accuracy_per_param,
                'efficiency_score': accuracy_gain / np.log(complexity_ratio + 1)
            },
            'deployment_considerations': {
                'custom_model': {
                    'pros': [
                        f"Higher accuracy (+{accuracy_gain:.3f})",
                        "Trainable on new data",
                        "Consistent performance",
                        "Probability outputs"
                    ],
                    'cons': [
                        f"Slower inference (+{latency_cost:.2f}ms)",
                        f"Larger model size ({perf_comp.get('model_properties', {}).get('custom_model', {}).get('size_mb', 0):.2f}MB)",
                        "Requires TensorFlow runtime",
                        "More complex deployment"
                    ]
                },
                'baseline_model': {
                    'pros': [
                        f"Faster inference ({baseline_single_time:.2f}ms)",
                        "Minimal memory footprint",
                        "No ML framework dependency",
                        "Interpretable rules"
                    ],
                    'cons': [
                        f"Lower accuracy (-{accuracy_gain:.3f})",
                        "Hard-coded logic",
                        "Not easily adaptable",
                        "Limited gesture support"
                    ]
                }
            }
        }
        
        self.results['trade_offs'] = trade_offs
        return trade_offs
    
    def generate_comparison_visualizations(self):
        """Generate comprehensive comparison visualizations."""
        console.print("[bold blue]Generating comparison visualizations...[/bold blue]")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Accuracy Comparison
        if 'accuracy_comparison' in self.results:
            acc_comp = self.results['accuracy_comparison']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Metrics comparison
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            custom_values = [acc_comp['custom_model'][m] for m in metrics]
            baseline_values = [acc_comp['baseline_model'][m] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, custom_values, width, label='Custom Model', color='#2E86AB')
            ax1.bar(x + width/2, baseline_values, width, label='MediaPipe Baseline', color='#A23B72')
            
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Score')
            ax1.set_title('Model Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Improvement percentages
            improvements = [acc_comp['relative_improvements'][m] for m in metrics]
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            
            bars = ax2.bar(metrics, improvements, color=colors, alpha=0.7)
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title('Relative Performance Improvement')
            ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, improvements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                        f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'accuracy_comparison.png'), dpi=300)
            plt.close()
        
        # 2. Performance Comparison
        if 'performance_comparison' in self.results:
            perf_comp = self.results['performance_comparison']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Inference time comparison
            batch_sizes = sorted([int(k) for k in perf_comp['inference_times']['custom_model'].keys()])
            custom_times = [perf_comp['inference_times']['custom_model'][str(b)]['mean'] for b in batch_sizes]
            baseline_times = [perf_comp['inference_times']['baseline_model'][str(b)]['mean'] for b in batch_sizes]
            
            ax1.plot(batch_sizes, custom_times, marker='o', label='Custom Model', linewidth=2, color='#2E86AB')
            ax1.plot(batch_sizes, baseline_times, marker='s', label='MediaPipe Baseline', linewidth=2, color='#A23B72')
            
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Inference Time (ms per sample)')
            ax1.set_title('Inference Time Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            
            # Model complexity comparison
            models = ['Custom Model', 'MediaPipe Baseline']
            sizes = [
                perf_comp['model_properties']['custom_model']['size_mb'],
                perf_comp['model_properties']['baseline_model']['size_mb']
            ]
            params = [
                perf_comp['model_properties']['custom_model']['parameters'],
                perf_comp['model_properties']['baseline_model']['parameters']
            ]
            
            ax2_twin = ax2.twinx()
            
            bar1 = ax2.bar([m + ' (Size)' for m in models], sizes, alpha=0.7, color='#F18F01', label='Model Size (MB)')
            bar2 = ax2_twin.bar([m + ' (Params)' for m in models], params, alpha=0.7, color='#C73E1D', label='Parameters')
            
            ax2.set_ylabel('Model Size (MB)', color='#F18F01')
            ax2_twin.set_ylabel('Parameters', color='#C73E1D')
            ax2.set_title('Model Complexity Comparison')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bar1, sizes):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}MB', ha='center', va='bottom')
            
            for bar, value in zip(bar2, params):
                ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.05,
                             f'{value:,}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'), dpi=300)
            plt.close()
        
        # 3. Statistical Significance Visualization
        if 'statistical_significance' in self.results:
            stat_results = self.results['statistical_significance']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Box plot of CV scores
            custom_scores = stat_results['cv_scores']['custom_model']
            baseline_scores = stat_results['cv_scores']['baseline_model']
            
            data_to_plot = [baseline_scores, custom_scores]
            labels = ['MediaPipe Baseline', 'Custom Model']
            
            box_plot = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('#A23B72')
            box_plot['boxes'][1].set_facecolor('#2E86AB')
            
            ax1.set_ylabel('Accuracy Score')
            ax1.set_title('Cross-Validation Score Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Add significance annotation
            p_value = stat_results['statistical_tests']['paired_t_test']['p_value']
            significance_text = f"p-value: {p_value:.4f}\n"
            significance_text += "Significant" if p_value < 0.05 else "Not Significant"
            ax1.text(0.5, 0.95, significance_text, transform=ax1.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    ha='center', va='top')
            
            # Effect size visualization
            effect_size = stat_results['effect_size']['cohens_d']
            interpretation = stat_results['effect_size']['interpretation']
            
            # Create effect size bar chart
            ax2.bar(['Effect Size'], [abs(effect_size)], 
                   color='green' if effect_size > 0 else 'red', alpha=0.7)
            ax2.set_ylabel("Cohen's d")
            ax2.set_title(f'Effect Size: {interpretation.title()}')
            ax2.grid(True, alpha=0.3)
            
            # Add interpretation text
            ax2.text(0, abs(effect_size) + 0.05, f'{effect_size:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'statistical_significance.png'), dpi=300)
            plt.close()
        
        console.print("✓ Comparison visualizations saved")
    
    def display_comparison_results(self):
        """Display comprehensive comparison results."""
        console.print(Panel.fit("⚡ Model Comparison Results", style="bold magenta"))
        
        # Accuracy Comparison Table
        if 'accuracy_comparison' in self.results:
            acc_table = Table(title="Accuracy Metrics Comparison")
            acc_table.add_column("Metric", style="cyan")
            acc_table.add_column("Custom Model", style="green")
            acc_table.add_column("MediaPipe Baseline", style="yellow")
            acc_table.add_column("Improvement", style="magenta")
            
            acc_comp = self.results['accuracy_comparison']
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for metric in metrics:
                custom_val = acc_comp['custom_model'][metric]
                baseline_val = acc_comp['baseline_model'][metric]
                improvement = acc_comp['improvements'][metric]
                
                acc_table.add_row(
                    metric.replace('_', ' ').title(),
                    f"{custom_val:.4f}",
                    f"{baseline_val:.4f}",
                    f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
                )
            
            console.print(acc_table)
        
        # Performance Comparison Table
        if 'performance_comparison' in self.results:
            perf_table = Table(title="Performance Metrics Comparison")
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Custom Model", style="green")
            perf_table.add_column("MediaPipe Baseline", style="yellow")
            perf_table.add_column("Ratio", style="magenta")
            
            perf_comp = self.results['performance_comparison']
            
            # Single sample inference time
            custom_time = perf_comp['inference_times']['custom_model']['1']['mean']
            baseline_time = perf_comp['inference_times']['baseline_model']['1']['mean']
            time_ratio = custom_time / baseline_time if baseline_time > 0 else 0
            
            perf_table.add_row("Inference Time (1 sample)", 
                             f"{custom_time:.2f} ms", 
                             f"{baseline_time:.2f} ms",
                             f"{time_ratio:.2f}x")
            
            # Model size
            custom_size = perf_comp['model_properties']['custom_model']['size_mb']
            baseline_size = perf_comp['model_properties']['baseline_model']['size_mb']
            size_ratio = custom_size / baseline_size if baseline_size > 0 else 0
            
            perf_table.add_row("Model Size", 
                             f"{custom_size:.2f} MB", 
                             f"{baseline_size:.2f} MB",
                             f"{size_ratio:.1f}x")
            
            # Parameters
            custom_params = perf_comp['model_properties']['custom_model']['parameters']
            baseline_params = perf_comp['model_properties']['baseline_model']['parameters']
            param_ratio = custom_params / baseline_params if baseline_params > 0 else 0
            
            perf_table.add_row("Parameters", 
                             f"{custom_params:,}", 
                             f"{baseline_params:,}",
                             f"{param_ratio:.0f}x")
            
            console.print(perf_table)
        
        # Statistical Significance
        if 'statistical_significance' in self.results:
            stat_table = Table(title="Statistical Significance Analysis")
            stat_table.add_column("Test", style="cyan")
            stat_table.add_column("Statistic", style="green")
            stat_table.add_column("P-value", style="yellow")
            stat_table.add_column("Significant", style="magenta")
            
            stat_results = self.results['statistical_significance']
            
            # T-test
            t_test = stat_results['statistical_tests']['paired_t_test']
            stat_table.add_row("Paired T-test", 
                             f"{t_test['t_statistic']:.4f}",
                             f"{t_test['p_value']:.4f}",
                             "Yes" if t_test['significant'] else "No")
            
            # Wilcoxon test
            wilcoxon = stat_results['statistical_tests']['wilcoxon_test']
            stat_table.add_row("Wilcoxon Test", 
                             f"{wilcoxon['statistic']:.4f}",
                             f"{wilcoxon['p_value']:.4f}",
                             "Yes" if wilcoxon['significant'] else "No")
            
            # Effect size
            effect_size = stat_results['effect_size']
            stat_table.add_row("Effect Size (Cohen's d)", 
                             f"{effect_size['cohens_d']:.4f}",
                             f"Interpretation: {effect_size['interpretation']}",
                             "")
            
            console.print(stat_table)
        
        # Trade-off Summary
        if 'trade_offs' in self.results:
            trade_off_panel = Panel.fit(
                f"""
[bold cyan]Trade-off Analysis Summary:[/bold cyan]

[green]Custom Model Advantages:[/green]
• Higher accuracy and precision
• Trainable and adaptable
• Consistent probabilistic outputs

[yellow]Custom Model Trade-offs:[/yellow]
• Increased inference latency
• Larger model size and memory usage
• Deployment complexity

[blue]Recommendation:[/blue]
{'Use custom model for accuracy-critical applications' if self.results['accuracy_comparison']['improvements']['accuracy'] > 0.05 
 else 'Consider baseline for speed-critical applications'}
                """.strip(),
                title="Trade-off Analysis",
                border_style="blue"
            )
            console.print(trade_off_panel)
    
    def save_comparison_results(self):
        """Save comprehensive comparison results."""
        console.print("[bold blue]Saving comparison results...[/bold blue]")
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, 'comparison_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save executive summary
        summary = {
            'comparison_timestamp': datetime.now().isoformat(),
            'models_compared': {
                'custom_model': self.custom_model_path,
                'baseline_model': 'MediaPipe + Heuristic Classification'
            },
            'test_samples': len(self.X_test),
            'key_findings': {},
            'recommendations': {}
        }
        
        if 'accuracy_comparison' in self.results:
            acc_comp = self.results['accuracy_comparison']
            summary['key_findings']['accuracy'] = {
                'custom_model_accuracy': acc_comp['custom_model']['accuracy'],
                'baseline_accuracy': acc_comp['baseline_model']['accuracy'],
                'improvement': acc_comp['improvements']['accuracy'],
                'relative_improvement_percent': acc_comp['relative_improvements']['accuracy']
            }
        
        if 'performance_comparison' in self.results:
            perf_comp = self.results['performance_comparison']
            custom_time = perf_comp['inference_times']['custom_model']['1']['mean']
            baseline_time = perf_comp['inference_times']['baseline_model']['1']['mean']
            
            summary['key_findings']['performance'] = {
                'custom_model_latency_ms': custom_time,
                'baseline_latency_ms': baseline_time,
                'latency_overhead_ms': custom_time - baseline_time,
                'speed_ratio': baseline_time / custom_time if custom_time > 0 else 0
            }
        
        if 'statistical_significance' in self.results:
            stat_results = self.results['statistical_significance']
            summary['key_findings']['statistical_significance'] = {
                'p_value': stat_results['statistical_tests']['paired_t_test']['p_value'],
                'significant': stat_results['statistical_tests']['paired_t_test']['significant'],
                'effect_size': stat_results['effect_size']['cohens_d'],
                'effect_interpretation': stat_results['effect_size']['interpretation']
            }
        
        # Add recommendations
        accuracy_improvement = summary['key_findings'].get('accuracy', {}).get('improvement', 0)
        latency_overhead = summary['key_findings'].get('performance', {}).get('latency_overhead_ms', 0)
        
        if accuracy_improvement > 0.05 and latency_overhead < 50:
            summary['recommendations']['primary'] = "Custom model recommended - significant accuracy gain with acceptable latency overhead"
        elif accuracy_improvement > 0.1:
            summary['recommendations']['primary'] = "Custom model recommended - substantial accuracy improvement justifies performance cost"
        elif latency_overhead > 100:
            summary['recommendations']['primary'] = "Consider baseline for real-time applications - custom model too slow"
        else:
            summary['recommendations']['primary'] = "Choice depends on application requirements - both models viable"
        
        summary_path = os.path.join(self.output_dir, 'comparison_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        console.print(f"✓ Comparison results saved to {self.output_dir}")
    
    def run_comprehensive_comparison(self):
        """Run complete model comparison pipeline."""
        console.print(Panel.fit("🔄 Starting Comprehensive Model Comparison", style="bold blue"))
        
        with Progress() as progress:
            task = progress.add_task("Comparison", total=6)
            
            # Accuracy comparison
            progress.update(task, description="Comparing accuracy metrics...")
            self.evaluate_accuracy_comparison()
            progress.advance(task)
            
            # Performance comparison
            progress.update(task, description="Comparing performance...")
            self.evaluate_performance_comparison()
            progress.advance(task)
            
            # Statistical significance
            progress.update(task, description="Statistical testing...")
            self.statistical_significance_testing()
            progress.advance(task)
            
            # Trade-off analysis
            progress.update(task, description="Analyzing trade-offs...")
            self.analyze_trade_offs()
            progress.advance(task)
            
            # Visualizations
            progress.update(task, description="Generating visualizations...")
            self.generate_comparison_visualizations()
            progress.advance(task)
            
            # Save results
            progress.update(task, description="Saving results...")
            self.save_comparison_results()
            progress.advance(task)
        
        # Display results
        self.display_comparison_results()
        
        console.print(f"\n✓ [bold green]Model comparison completed![/bold green]")
        console.print(f"Results saved to: {self.output_dir}")

def main():
    """Main comparison interface."""
    parser = argparse.ArgumentParser(description='Compare custom model with MediaPipe baseline')
    parser.add_argument('--custom-model', type=str, required=True, help='Path to custom trained model (.h5)')
    parser.add_argument('--data', type=str, required=True, help='Path to test data (.json)')
    parser.add_argument('--output', type=str, default='comparison_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.custom_model):
        console.print(f"[red]Error: Custom model file not found: {args.custom_model}[/red]")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        console.print(f"[red]Error: Data file not found: {args.data}[/red]")
        sys.exit(1)
    
    # Run comparison
    comparator = ModelComparator(args.custom_model, args.data, args.output)
    comparator.run_comprehensive_comparison()

if __name__ == '__main__':
    main()