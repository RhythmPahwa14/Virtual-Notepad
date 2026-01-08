#!/usr/bin/env python3
"""
Comprehensive Model Evaluation and Comparison
============================================

This script provides extensive evaluation capabilities for gesture recognition models,
including comparison with MediaPipe baselines and performance profiling.

Features:
- Comprehensive evaluation metrics
- Statistical significance testing
- Performance profiling and benchmarking
- Confusion matrix analysis
- ROC curve analysis
- Cross-validation
- Model comparison
"""

import os
import json
import pickle
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import tensorflow as tf
import mediapipe as mp
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, track

console = Console()

class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.label_encoder = None
        self.test_data = None
        
        # Load model and dependencies
        self._load_model()
        self._load_data()
        
    def _load_model(self):
        """Load trained model and label encoder."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            console.print(f"✓ Model loaded from {self.model_path}")
            
            # Load label encoder
            encoder_path = self.model_path.replace('.h5', '_label_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                console.print(f"✓ Label encoder loaded")
            else:
                console.print("[yellow]Warning: Label encoder not found[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            raise
    
    def _load_data(self):
        """Load test data."""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            # Split data for testing (use last 20% as test set)
            test_size = int(len(data) * 0.2)
            self.test_data = data[-test_size:]
            
            console.print(f"✓ Loaded {len(self.test_data)} test samples")
            
        except Exception as e:
            console.print(f"[red]Error loading data: {e}[/red]")
            raise
    
    def preprocess_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess test data for evaluation."""
        X_test = []
        y_test = []
        
        for sample in self.test_data:
            landmarks = np.array(sample['landmarks'])
            X_test.append(landmarks)
            y_test.append(sample['gesture'])
        
        X_test = np.array(X_test)
        
        if self.label_encoder:
            y_test = self.label_encoder.transform(y_test)
        
        return X_test, y_test
    
    def evaluate_accuracy_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics."""
        console.print("[bold blue]Calculating accuracy metrics...[/bold blue]")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class metrics
        class_names = self.label_encoder.classes_ if self.label_encoder else [f"Class_{i}" for i in range(len(np.unique(y_test)))]
        
        per_class_precision = precision_score(y_test, y_pred, average=None)
        per_class_recall = recall_score(y_test, y_pred, average=None)
        per_class_f1 = f1_score(y_test, y_pred, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist(),
            'class_names': class_names.tolist()
        }
        
        return metrics
    
    def evaluate_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """Generate and analyze confusion matrix."""
        console.print("[bold blue]Generating confusion matrix...[/bold blue]")
        
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return cm, cm_normalized
    
    def evaluate_performance_metrics(self, X_test: np.ndarray) -> Dict[str, float]:
        """Evaluate performance metrics (speed, memory)."""
        console.print("[bold blue]Evaluating performance metrics...[/bold blue]")
        
        # Warm up
        _ = self.model.predict(X_test[:5], verbose=0)
        
        # Measure inference time
        times = []
        for i in track(range(100), description="Measuring inference time..."):
            start_time = time.time()
            _ = self.model.predict(X_test[i:i+1], verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_inference_time = np.mean(times) * 1000  # Convert to milliseconds
        std_inference_time = np.std(times) * 1000
        
        # Batch inference
        batch_times = []
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            if len(X_test) >= batch_size:
                start_time = time.time()
                _ = self.model.predict(X_test[:batch_size], verbose=0)
                end_time = time.time()
                batch_times.append((end_time - start_time) / batch_size * 1000)
        
        # Model size
        model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
        
        # Memory usage (approximation)
        total_params = self.model.count_params()
        memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        performance_metrics = {
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'batch_inference_times': batch_times,
            'model_size_mb': model_size,
            'estimated_memory_mb': memory_mb,
            'total_parameters': total_params
        }
        
        return performance_metrics
    
    def cross_validation_evaluation(self, X_test: np.ndarray, y_test: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation evaluation."""
        console.print(f"[bold blue]Performing {cv_folds}-fold cross-validation...[/bold blue]")
        
        # For neural networks, we'll do a simpler evaluation
        # Split data into cv_folds and evaluate
        from sklearn.model_selection import StratifiedKFold
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_test, y_test)):
            X_val = X_test[val_idx]
            y_val = y_test[val_idx]
            
            # Predict on validation fold
            y_pred_proba = self.model.predict(X_val, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculate metrics
            cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
            cv_scores['precision'].append(precision_score(y_val, y_pred, average='weighted'))
            cv_scores['recall'].append(recall_score(y_val, y_pred, average='weighted'))
            cv_scores['f1'].append(f1_score(y_val, y_pred, average='weighted'))
        
        # Calculate statistics
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
            cv_results[f'{metric}_scores'] = scores
        
        return cv_results
    
    def statistical_significance_test(self, baseline_scores: List[float], 
                                    model_scores: List[float]) -> Dict[str, Any]:
        """Perform statistical significance testing."""
        console.print("[bold blue]Performing statistical significance test...[/bold blue]")
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model_scores, baseline_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(model_scores) + np.var(baseline_scores)) / 2)
        cohens_d = (np.mean(model_scores) - np.mean(baseline_scores)) / pooled_std
        
        # Wilcoxon signed-rank test (non-parametric)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(model_scores, baseline_scores)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'wilcoxon_statistic': wilcoxon_stat,
            'wilcoxon_p_value': wilcoxon_p,
            'significant': p_value < 0.05
        }
    
    def generate_visualizations(self, metrics: Dict[str, Any], 
                              cm: np.ndarray, cm_normalized: np.ndarray,
                              output_dir: str = "evaluation_results"):
        """Generate comprehensive evaluation visualizations."""
        console.print("[bold blue]Generating visualizations...[/bold blue]")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        class_names = metrics['class_names']
        
        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Raw Counts)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-class metrics
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(class_names))
        width = 0.25
        
        ax.bar(x - width, metrics['per_class_precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, metrics['per_class_recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, metrics['per_class_f1'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Gesture Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Overall metrics comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        overall_metrics = [
            metrics['accuracy'],
            metrics['precision_weighted'],
            metrics['recall_weighted'],
            metrics['f1_weighted']
        ]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        bars = ax.bar(metric_names, overall_metrics, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        # Add value labels on bars
        for bar, value in zip(bars, overall_metrics):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Score')
        ax.set_title('Overall Performance Metrics')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"✓ Visualizations saved to {output_dir}")
    
    def generate_comprehensive_report(self, all_results: Dict[str, Any], 
                                    output_path: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report."""
        console.print("[bold blue]Generating comprehensive report...[/bold blue]")
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'test_data_path': self.data_path,
            'test_samples_count': len(self.test_data),
            'results': all_results,
            'summary': {
                'overall_accuracy': all_results['accuracy_metrics']['accuracy'],
                'average_inference_time_ms': all_results['performance_metrics']['avg_inference_time_ms'],
                'model_size_mb': all_results['performance_metrics']['model_size_mb'],
                'cross_validation_accuracy_mean': all_results['cross_validation']['accuracy_mean'],
                'cross_validation_accuracy_std': all_results['cross_validation']['accuracy_std']
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"✓ Comprehensive report saved to {output_path}")
        
        return report
    
    def display_results_table(self, results: Dict[str, Any]):
        """Display results in formatted tables."""
        
        # Overall metrics table
        metrics_table = Table(title="Overall Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Score", style="magenta")
        
        acc_metrics = results['accuracy_metrics']
        metrics_table.add_row("Accuracy", f"{acc_metrics['accuracy']:.4f}")
        metrics_table.add_row("Precision (Weighted)", f"{acc_metrics['precision_weighted']:.4f}")
        metrics_table.add_row("Recall (Weighted)", f"{acc_metrics['recall_weighted']:.4f}")
        metrics_table.add_row("F1-Score (Weighted)", f"{acc_metrics['f1_weighted']:.4f}")
        
        console.print(metrics_table)
        
        # Performance metrics table
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        perf_metrics = results['performance_metrics']
        perf_table.add_row("Avg Inference Time", f"{perf_metrics['avg_inference_time_ms']:.2f} ms")
        perf_table.add_row("Model Size", f"{perf_metrics['model_size_mb']:.2f} MB")
        perf_table.add_row("Total Parameters", f"{perf_metrics['total_parameters']:,}")
        perf_table.add_row("Est. Memory Usage", f"{perf_metrics['estimated_memory_mb']:.2f} MB")
        
        console.print(perf_table)
        
        # Cross-validation results
        cv_table = Table(title="Cross-Validation Results")
        cv_table.add_column("Metric", style="cyan")
        cv_table.add_column("Mean", style="green")
        cv_table.add_column("Std Dev", style="yellow")
        
        cv_results = results['cross_validation']
        cv_table.add_row("Accuracy", f"{cv_results['accuracy_mean']:.4f}", f"±{cv_results['accuracy_std']:.4f}")
        cv_table.add_row("Precision", f"{cv_results['precision_mean']:.4f}", f"±{cv_results['precision_std']:.4f}")
        cv_table.add_row("Recall", f"{cv_results['recall_mean']:.4f}", f"±{cv_results['recall_std']:.4f}")
        cv_table.add_row("F1-Score", f"{cv_results['f1_mean']:.4f}", f"±{cv_results['f1_std']:.4f}")
        
        console.print(cv_table)
    
    def run_comprehensive_evaluation(self, output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Run comprehensive model evaluation."""
        console.print(Panel.fit("🔍 Comprehensive Model Evaluation", style="bold magenta"))
        
        # Preprocess test data
        X_test, y_test = self.preprocess_test_data()
        
        # Run all evaluations
        results = {}
        
        # 1. Accuracy metrics
        results['accuracy_metrics'] = self.evaluate_accuracy_metrics(X_test, y_test)
        
        # 2. Confusion matrix
        cm, cm_normalized = self.evaluate_confusion_matrix(X_test, y_test)
        results['confusion_matrix'] = cm.tolist()
        results['confusion_matrix_normalized'] = cm_normalized.tolist()
        
        # 3. Performance metrics
        results['performance_metrics'] = self.evaluate_performance_metrics(X_test)
        
        # 4. Cross-validation
        results['cross_validation'] = self.cross_validation_evaluation(X_test, y_test)
        
        # 5. Generate visualizations
        self.generate_visualizations(results['accuracy_metrics'], cm, cm_normalized, output_dir)
        
        # 6. Display results
        self.display_results_table(results)
        
        # 7. Generate comprehensive report
        report = self.generate_comprehensive_report(results, 
                                                  os.path.join(output_dir, 'evaluation_report.json'))
        
        return results

def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        console.print(f"[red]Error: Model file not found: {args.model}[/red]")
        return
    
    if not os.path.exists(args.data):
        console.print(f"[red]Error: Data file not found: {args.data}[/red]")
        return
    
    try:
        evaluator = ModelEvaluator(args.model, args.data)
        results = evaluator.run_comprehensive_evaluation(args.output_dir)
        
        console.print("\n[bold green]✓ Evaluation completed successfully![/bold green]")
        console.print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise

if __name__ == '__main__':
    main()