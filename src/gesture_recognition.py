import cv2
import numpy as np
import time
import json
import os
from datetime import datetime

class GestureRecognizer:
    def __init__(self):
        self.last_gesture = None
        self.gesture_start_time = None
        self.gesture_hold_duration = 1.0  # seconds
        self.gesture_data = []
        self.data_file = "data/gesture_data.json"
        self._debug_counter = 0
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Load existing data
        self.load_gesture_data()
    
    def calculate_finger_angles(self, landmarks):
        """Calculate finger positions using distance-based method"""
        try:
            # Extract landmark positions
            lm = landmarks.landmark
            
            # Finger tip and joint positions (MediaPipe hand landmarks)
            thumb_tip = lm[4]     # Thumb tip
            thumb_ip = lm[3]      # Thumb IP joint
            
            index_tip = lm[8]     # Index tip
            index_pip = lm[6]     # Index PIP joint
            
            middle_tip = lm[12]   # Middle tip
            middle_pip = lm[10]   # Middle PIP joint
            
            ring_tip = lm[16]     # Ring tip
            ring_pip = lm[14]     # Ring PIP joint
            
            pinky_tip = lm[20]    # Pinky tip
            pinky_pip = lm[18]    # Pinky PIP joint
            
            # Calculate if fingers are extended using distance method
            def is_finger_extended(tip, pip, reference_point):
                tip_distance = ((tip.x - reference_point.x)**2 + 
                              (tip.y - reference_point.y)**2)**0.5
                pip_distance = ((pip.x - reference_point.x)**2 + 
                              (pip.y - reference_point.y)**2)**0.5
                return tip_distance > pip_distance
            
            # Use wrist as reference point
            wrist = lm[0]
            
            # Check each finger
            thumb_up = is_finger_extended(thumb_tip, thumb_ip, wrist)
            index_up = is_finger_extended(index_tip, index_pip, wrist)
            middle_up = is_finger_extended(middle_tip, middle_pip, wrist)
            ring_up = is_finger_extended(ring_tip, ring_pip, wrist)
            pinky_up = is_finger_extended(pinky_tip, pinky_pip, wrist)
            
            return [thumb_up, index_up, middle_up, ring_up, pinky_up]
            
        except Exception as e:
            print(f"Error calculating finger angles: {e}")
            return None
    
    def recognize_gesture(self, landmarks):
        """Recognize gesture from hand landmarks"""
        if not landmarks:
            return None
            
        fingers = self.calculate_finger_angles(landmarks)
        if not fingers:
            return None
        
        # Store for debug display
        self._last_fingers = fingers
            
        # Debug: print finger states occasionally
        self._debug_counter += 1
        if self._debug_counter % 30 == 0:  # Print every 30th detection
            print(f"Fingers [T,I,M,R,P]: {fingers}")
        
        # Define gesture patterns with improved detection
        gestures = {
            'thumbs_up': [True, False, False, False, False], # 👍 Thumb only
            'open_palm': [True, True, True, True, True],     # ✋ All fingers up
            'peace': [False, True, True, False, False],      # ✌️ Index + Middle
            'point_up': [False, True, False, False, False],  # ☝️ Index only
            'fist': [False, False, False, False, False],     # ✊ All fingers down
            'rock': [False, True, False, False, True],       # 🤟 Index + Pinky
        }
        
        # Find matching gesture with exact pattern match
        for gesture_name, pattern in gestures.items():
            if fingers == pattern:
                if gesture_name != 'thumbs_up' or self._debug_counter % 50 == 0:
                    print(f"🎯 EXACT: {gesture_name} detected!")
                return gesture_name
        
        # Fallback: Try approximate matching for difficult gestures
        if self._approximate_match(fingers, [False, True, True, False, False]):
            if self._debug_counter % 50 == 0:
                print("🎯 APPROXIMATE: Peace gesture detected!")
            return 'peace'
        elif self._approximate_match(fingers, [True, True, False, False, True]):
            if self._debug_counter % 50 == 0:
                print("🎯 APPROXIMATE: Rock gesture detected!")
            return 'rock'
        elif self._approximate_match(fingers, [False, False, False, False, False]):
            if self._debug_counter % 50 == 0:
                print("🎯 APPROXIMATE: Fist gesture detected!")
            return 'fist'
        elif self._approximate_match(fingers, [True, True, True, True, True]):
            if self._debug_counter % 50 == 0:
                print("🎯 APPROXIMATE: Open palm gesture detected!")
            return 'open_palm'
        
        # No gesture detected
        return None
    
    def _approximate_match(self, fingers, pattern):
        """Check if fingers approximately match pattern (allows 1 finger difference)"""
        if len(fingers) != len(pattern):
            return False
        
        differences = sum(1 for i in range(len(fingers)) if fingers[i] != pattern[i])
        return differences <= 1  # Allow 1 finger to be different
    
    def process_gesture(self, landmarks):
        """Process gesture with timing and data collection"""
        current_gesture = self.recognize_gesture(landmarks)
        current_time = time.time()
        
        # Check if gesture changed
        if current_gesture != self.last_gesture:
            self.last_gesture = current_gesture
            self.gesture_start_time = current_time if current_gesture else None
            return None, False
        
        # Check if gesture held long enough
        if (current_gesture and self.gesture_start_time and 
            current_time - self.gesture_start_time >= self.gesture_hold_duration):
            
            # Collect data for ML training
            self.collect_gesture_data(landmarks, current_gesture)
            
            # Reset timer to avoid repeated triggers
            self.gesture_start_time = current_time + 2.0  # 2-second cooldown
            
            return current_gesture, True
            
        return current_gesture, False
    
    def collect_gesture_data(self, landmarks, gesture_name):
        """Collect gesture data for ML training"""
        if len(self.gesture_data) >= 100:  # Limit data collection
            return
            
        # Extract landmark coordinates
        landmark_data = []
        for lm in landmarks.landmark:
            landmark_data.extend([lm.x, lm.y, lm.z])
        
        # Create data entry
        data_entry = {
            'landmarks': landmark_data,
            'gesture': gesture_name,
            'timestamp': datetime.now().isoformat(),
            'finger_states': self._last_fingers if hasattr(self, '_last_fingers') else None
        }
        
        self.gesture_data.append(data_entry)
        
        # Save data periodically
        if len(self.gesture_data) % 20 == 0:
            self.save_gesture_data()
            print(f"💾 Gesture data saved ({len(self.gesture_data)} samples)")
    
    def save_gesture_data(self):
        """Save collected gesture data to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.gesture_data, f, indent=2)
        except Exception as e:
            print(f"Error saving gesture data: {e}")
    
    def load_gesture_data(self):
        """Load existing gesture data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.gesture_data = json.load(f)
        except Exception as e:
            print(f"Error loading gesture data: {e}")
            self.gesture_data = []
    
    def get_gesture_feedback(self, gesture_name):
        """Get visual feedback for current gesture"""
        if not gesture_name:
            return "No gesture detected", (100, 100, 100)
        
        feedback_map = {
            'peace': ("Peace Sign", (0, 255, 0)),
            'thumbs_up': ("Thumbs Up", (0, 255, 255)),
            'fist': ("Fist", (0, 0, 255)),
            'rock': ("Rock Sign", (255, 0, 255)),
            'point_up': ("Point Up", (255, 255, 0)),
            'open_palm': ("Open Palm", (255, 150, 0))
        }
        
        return feedback_map.get(gesture_name, ("Unknown gesture", (128, 128, 128)))
    
    def draw_gesture_feedback(self, image, gesture_name, landmarks=None):
        """Draw gesture feedback on image"""
        height, width = image.shape[:2]
        
        # Get gesture feedback
        text, color = self.get_gesture_feedback(gesture_name)
        
        # Draw gesture text - moved higher to avoid control text overlap
        cv2.putText(image, text, (10, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw finger states for debugging (only occasionally)
        if (hasattr(self, '_last_fingers') and self._last_fingers and 
            hasattr(self, '_debug_counter') and self._debug_counter % 100 == 0):
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            for i, (name, state) in enumerate(zip(finger_names, self._last_fingers)):
                status = "UP" if state else "DOWN"
                finger_color = (0, 255, 0) if state else (0, 0, 255)
                cv2.putText(image, f"{name}: {status}", (10, height - 40 + i*15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, finger_color, 1)
