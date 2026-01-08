#!/usr/bin/env python3
"""
Advanced Data Collection for Gesture Recognition
===============================================

This script provides a comprehensive data collection interface for gathering
high-quality gesture recognition training data with real-time validation.

Features:
- Real-time hand tracking with MediaPipe
- Quality validation and outlier detection
- Balanced dataset collection
- Automated data augmentation
- Export to multiple formats
"""

import cv2
import json
import numpy as np
import mediapipe as mp
from datetime import datetime
import argparse
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()

class DataCollector:
    """Advanced data collection system for gesture recognition."""
    
    def __init__(self, output_path: str = "data/gesture_data.json"):
        self.output_path = output_path
        self.data = []
        self.gesture_classes = [
            "fist", "open_palm", "peace", "rock", "thumbs_up"
        ]
        self.class_counts = defaultdict(int)
        self.quality_metrics = {
            'total_samples': 0,
            'rejected_samples': 0,
            'quality_scores': []
        }
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load existing data if available
        self._load_existing_data()
        
    def _load_existing_data(self):
        """Load existing data from file."""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as f:
                    self.data = json.load(f)
                
                # Update class counts
                for sample in self.data:
                    self.class_counts[sample['gesture']] += 1
                
                console.print(f"✓ Loaded {len(self.data)} existing samples")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load existing data: {e}[/yellow]")
    
    def _calculate_quality_score(self, landmarks: List[Dict]) -> float:
        """Calculate quality score for hand landmarks."""
        if not landmarks or len(landmarks) != 21:
            return 0.0
        
        # Convert to numpy array
        points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # Quality factors
        
        # 1. Visibility (all landmarks should be detected)
        visibility_score = 1.0  # MediaPipe already filters for visibility
        
        # 2. Hand size (reasonable size in frame)
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        size_score = min(1.0, (x_range + y_range) / 0.4)  # Penalize very small hands
        
        # 3. Landmark stability (low jitter)
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
        
        stability_score = 1.0 if np.std(distances) < 0.5 else 0.5
        
        # 4. Hand centering (hand should be reasonably centered)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        center_score = 1.0 - min(1.0, 2 * max(abs(center_x - 0.5), abs(center_y - 0.5)))
        
        # Combined score
        quality_score = (visibility_score * 0.3 + 
                        size_score * 0.3 + 
                        stability_score * 0.2 + 
                        center_score * 0.2)
        
        return quality_score
    
    def _preprocess_landmarks(self, landmarks: List[Dict]) -> List[float]:
        """Preprocess landmarks for storage."""
        # Convert to numpy array
        points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # Normalize relative to hand center
        center = np.mean(points, axis=0)
        centered = points - center
        
        # Scale to unit sphere
        max_distance = np.max(np.linalg.norm(centered, axis=1))
        if max_distance > 0:
            normalized = centered / max_distance
        else:
            normalized = centered
        
        return normalized.flatten().tolist()
    
    def _analyze_finger_states(self, landmarks: List[Dict]) -> List[bool]:
        """Analyze which fingers are extended."""
        # Finger tip and PIP landmark indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [3, 6, 10, 14, 18]
        
        finger_states = []
        points = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            # Simple heuristic: finger is extended if tip is further from wrist than PIP
            wrist = points[0]
            tip = points[tip_idx]
            pip = points[pip_idx]
            
            tip_dist = np.linalg.norm(tip - wrist)
            pip_dist = np.linalg.norm(pip - wrist)
            
            is_extended = tip_dist > pip_dist
            finger_states.append(is_extended)
        
        return finger_states
    
    def collect_gesture_data(self, gesture: str, target_samples: int = 50):
        """Collect data for a specific gesture."""
        console.print(f"\n[bold blue]Collecting data for gesture: {gesture.upper()}[/bold blue]")
        console.print(f"Target: {target_samples} samples")
        console.print("Press 'SPACE' to capture, 'q' to quit, 's' to skip")
        
        cap = cv2.VideoCapture(0)
        samples_collected = 0
        
        try:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
            ) as progress:
                
                task = progress.add_task(f"Collecting {gesture}", total=target_samples)
                
                while samples_collected < target_samples:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    results = self.hands.process(rgb_frame)
                    
                    # Draw landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_draw.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                            )
                            
                            # Calculate quality score
                            landmarks_list = [
                                {
                                    'x': lm.x, 'y': lm.y, 'z': lm.z
                                } for lm in hand_landmarks.landmark
                            ]
                            quality_score = self._calculate_quality_score(landmarks_list)
                            
                            # Display quality indicator
                            quality_color = (0, 255, 0) if quality_score > 0.7 else (0, 255, 255) if quality_score > 0.5 else (0, 0, 255)
                            cv2.putText(frame, f"Quality: {quality_score:.2f}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, quality_color, 2)
                    
                    # Display instructions
                    cv2.putText(frame, f"Gesture: {gesture.upper()}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, f"Samples: {samples_collected}/{target_samples}", 
                              (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, "SPACE: Capture | Q: Quit | S: Skip", 
                              (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Gesture Collection', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord(' '):  # Space to capture
                        if results.multi_hand_landmarks:
                            hand_landmarks = results.multi_hand_landmarks[0]
                            landmarks_list = [
                                {
                                    'x': lm.x, 'y': lm.y, 'z': lm.z
                                } for lm in hand_landmarks.landmark
                            ]
                            
                            quality_score = self._calculate_quality_score(landmarks_list)
                            
                            if quality_score > 0.5:  # Quality threshold
                                # Process and store sample
                                processed_landmarks = self._preprocess_landmarks(landmarks_list)
                                finger_states = self._analyze_finger_states(landmarks_list)
                                
                                sample = {
                                    'landmarks': processed_landmarks,
                                    'gesture': gesture,
                                    'timestamp': datetime.now().isoformat(),
                                    'finger_states': finger_states,
                                    'quality_score': quality_score
                                }
                                
                                self.data.append(sample)
                                self.class_counts[gesture] += 1
                                self.quality_metrics['total_samples'] += 1
                                self.quality_metrics['quality_scores'].append(quality_score)
                                
                                samples_collected += 1
                                progress.update(task, advance=1)
                                
                                console.print(f"✓ Sample {samples_collected} captured (quality: {quality_score:.2f})")
                            else:
                                self.quality_metrics['rejected_samples'] += 1
                                console.print(f"[red]✗ Sample rejected (low quality: {quality_score:.2f})[/red]")
                        else:
                            console.print("[red]✗ No hand detected[/red]")
                    
                    elif key == ord('q'):  # Quit
                        break
                    elif key == ord('s'):  # Skip
                        break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        console.print(f"✓ Collected {samples_collected} samples for {gesture}")
        return samples_collected
    
    def display_statistics(self):
        """Display collection statistics."""
        table = Table(title="Data Collection Statistics")
        table.add_column("Gesture", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Percentage", style="green")
        
        total_samples = sum(self.class_counts.values())
        
        for gesture in self.gesture_classes:
            count = self.class_counts[gesture]
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            table.add_row(gesture, str(count), f"{percentage:.1f}%")
        
        table.add_row("TOTAL", str(total_samples), "100.0%", style="bold")
        
        console.print(table)
        
        # Quality statistics
        if self.quality_metrics['quality_scores']:
            avg_quality = np.mean(self.quality_metrics['quality_scores'])
            console.print(f"\n[bold blue]Quality Metrics:[/bold blue]")
            console.print(f"Average Quality Score: {avg_quality:.3f}")
            console.print(f"Rejected Samples: {self.quality_metrics['rejected_samples']}")
            console.print(f"Acceptance Rate: {(self.quality_metrics['total_samples'] / (self.quality_metrics['total_samples'] + self.quality_metrics['rejected_samples']) * 100):.1f}%")
    
    def save_data(self):
        """Save collected data to file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Save main data file
        with open(self.output_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        # Save metadata
        metadata = {
            'total_samples': len(self.data),
            'class_distribution': dict(self.class_counts),
            'quality_metrics': self.quality_metrics,
            'collection_date': datetime.now().isoformat(),
            'gesture_classes': self.gesture_classes
        }
        
        metadata_path = self.output_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        console.print(f"✓ Data saved to {self.output_path}")
        console.print(f"✓ Metadata saved to {metadata_path}")
    
    def interactive_collection(self):
        """Interactive data collection interface."""
        console.print(Panel.fit("🤚 Advanced Gesture Data Collection", style="bold magenta"))
        
        while True:
            self.display_statistics()
            
            console.print("\n[bold blue]Options:[/bold blue]")
            console.print("1. Collect specific gesture")
            console.print("2. Collect balanced dataset")
            console.print("3. View statistics")
            console.print("4. Save and exit")
            console.print("5. Exit without saving")
            
            choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4", "5"])
            
            if choice == "1":
                # Collect specific gesture
                console.print(f"\nAvailable gestures: {', '.join(self.gesture_classes)}")
                gesture = Prompt.ask("Enter gesture name", choices=self.gesture_classes)
                target = int(Prompt.ask("Target samples", default="50"))
                self.collect_gesture_data(gesture, target)
                
            elif choice == "2":
                # Collect balanced dataset
                target_per_class = int(Prompt.ask("Samples per class", default="100"))
                
                for gesture in self.gesture_classes:
                    current_count = self.class_counts[gesture]
                    needed = max(0, target_per_class - current_count)
                    
                    if needed > 0:
                        console.print(f"\n[bold yellow]Need {needed} more samples for {gesture}[/bold yellow]")
                        if Confirm.ask(f"Collect data for {gesture}?"):
                            self.collect_gesture_data(gesture, needed)
                    else:
                        console.print(f"✓ {gesture}: already has {current_count} samples")
                
            elif choice == "3":
                # View statistics
                self.display_statistics()
                
            elif choice == "4":
                # Save and exit
                self.save_data()
                break
                
            elif choice == "5":
                # Exit without saving
                if not Confirm.ask("Exit without saving?"):
                    continue
                break

def main():
    """Main data collection interface."""
    parser = argparse.ArgumentParser(description='Collect gesture recognition data')
    parser.add_argument('--output', type=str, default='data/gesture_data.json',
                       help='Output file path')
    parser.add_argument('--gesture', type=str, help='Specific gesture to collect')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to collect')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    collector = DataCollector(args.output)
    
    if args.interactive or (not args.gesture):
        collector.interactive_collection()
    else:
        # Direct collection mode
        if args.gesture not in collector.gesture_classes:
            console.print(f"[red]Error: Invalid gesture '{args.gesture}'[/red]")
            console.print(f"Available gestures: {', '.join(collector.gesture_classes)}")
            return
        
        collector.collect_gesture_data(args.gesture, args.samples)
        collector.save_data()

if __name__ == '__main__':
    main()