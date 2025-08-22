import cv2
import numpy as np
import time
from .gesture_recognition import GestureRecognizer

class DrawingBoard:
    def __init__(self, tracker):
        self.tracker = tracker
        self.gesture_recognizer = GestureRecognizer()  # Add ML gesture recognition
        self.canvas = None
        self.prev_x, self.prev_y = None, None
        self.drawing_enabled = True  # Toggle drawing on/off
        self.eraser_size = 30
        self.brush_size = 5
        self.drawing_history = []  # Store drawing states for undo
        self.max_history = 10  # Maximum undo steps
        
        # Color options
        self.colors = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255),
            'orange': (0, 165, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        self.color_names = list(self.colors.keys())
        self.current_color_index = 0  # Start with red
        self.current_color = self.colors[self.color_names[self.current_color_index]]
        
        # UI settings
        self.fullscreen = False
        self.show_ui = True

    def execute_gesture_command(self, gesture):
        """Execute commands based on recognized gestures"""
        if gesture == 'peace':  # Peace Save drawing
            result = self.save_drawing()
            print(f"Peace Sign - {result}")
            return f"Saved Drawing"
            
        elif gesture == 'thumbs_up':  # Thumbs Change color
            self.change_color()
            color_name = self.color_names[self.current_color_index]
            print(f"Thumbs Up - Color changed to {color_name}")
            return f"Color: {color_name}"
            
        elif gesture == 'fist':  # Fist Clear canvas
            self.add_to_history()
            self.clear_canvas()
            print("Fist - Canvas cleared")
            return "Canvas Cleared"
            
        elif gesture == 'rock':  # Rock Undo
            success = self.undo_last_action()
            message = "Undo successful" if success else "Nothing to undo"
            print(f"Rock Sign - {message}")
            return message
            
        elif gesture == 'point_up':  # Point Toggle drawing
            self.toggle_drawing()
            status = "Drawing ON" if self.drawing_enabled else "Drawing OFF"
            print(f"Point Up - {status}")
            return status
            
        elif gesture == 'open_palm':  # Palm Change brush style
            style = self.change_brush_style()
            print(f"Open Palm - Brush style: {style}")
            return f"Brush: {style}"
            
        return None

    def detect_erasing_gesture(self, landmarks):
        """Detect if user is making erasing gesture (pinching thumb and index finger)"""
        if landmarks:
            # Get thumb tip and index finger tip
            thumb_tip = landmarks.landmark[self.tracker.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[self.tracker.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate distance between thumb and index finger
            distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            
            # If distance is small, user is pinching (erasing mode)
            return distance < 0.05
        return False

    def change_color(self):
        """Cycle through available colors"""
        self.current_color_index = (self.current_color_index + 1) % len(self.color_names)
        self.current_color = self.colors[self.color_names[self.current_color_index]]
        color_name = self.color_names[self.current_color_index]
        print(f"Color changed to: {color_name.upper()}")

    def toggle_drawing(self):
        """Toggle drawing on/off"""
        self.drawing_enabled = not self.drawing_enabled
        status = "ENABLED" if self.drawing_enabled else "DISABLED"
        print(f"Drawing {status}")

    def save_drawing(self):
        """Save the current drawing to a file"""
        if self.canvas is None:
            return "No drawing to save"
        
        try:
            # Create filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drawing_{timestamp}.png"
            
            # Save the canvas
            cv2.imwrite(filename, self.canvas)
            print(f"Drawing saved as: {filename}")
            return f"Saved as {filename}"
        except Exception as e:
            print(f"Error saving drawing: {e}")
            return "Save failed"

    def add_to_history(self):
        """Add current canvas state to history for undo functionality"""
        if self.canvas is not None:
            # Add current state to history
            self.drawing_history.append(self.canvas.copy())
            
            # Keep only the last max_history states
            if len(self.drawing_history) > self.max_history:
                self.drawing_history.pop(0)

    def undo_last_action(self):
        """Undo the last drawing action"""
        if len(self.drawing_history) > 0:
            # Restore previous state
            self.canvas = self.drawing_history.pop().copy()
            print("Undo: Restored previous drawing state")
            return True
        else:
            print("No more actions to undo")
            return False

    def add_shape(self, shape_type, center_x, center_y):
        """Add predefined shapes to the drawing"""
        if self.canvas is None:
            return
        
        size = self.brush_size * 10  # Scale shape size with brush size
        
        if shape_type == "circle":
            cv2.circle(self.canvas, (center_x, center_y), size, self.current_color, self.brush_size)
        elif shape_type == "rectangle":
            pt1 = (center_x - size, center_y - size)
            pt2 = (center_x + size, center_y + size)
            cv2.rectangle(self.canvas, pt1, pt2, self.current_color, self.brush_size)
        elif shape_type == "line":
            pt1 = (center_x - size, center_y)
            pt2 = (center_x + size, center_y)
            cv2.line(self.canvas, pt1, pt2, self.current_color, self.brush_size)
        
        print(f"🔷 Added {shape_type} at ({center_x}, {center_y})")

    def change_brush_style(self):
        """Change brush drawing style"""
        # Cycle through different brush styles
        if not hasattr(self, 'brush_style'):
            self.brush_style = 0
        
        styles = ["Normal", "Dotted", "Dashed"]
        self.brush_style = (self.brush_style + 1) % len(styles)
        print(f"Brush style: {styles[self.brush_style]}")
        return styles[self.brush_style]

    def draw_dotted_line(self, x1, y1, x2, y2):
        """Draw a dotted line between two points"""
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if distance < 1:
            return
        
        num_dots = int(distance / (self.brush_size * 2))
        for i in range(num_dots):
            t = i / max(num_dots - 1, 1)
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            cv2.circle(self.canvas, (x, y), self.brush_size // 2, self.current_color, -1)

    def draw_dashed_line(self, x1, y1, x2, y2):
        """Draw a dashed line between two points"""
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if distance < 1:
            return
        
        dash_length = self.brush_size * 3
        gap_length = self.brush_size * 2
        total_length = dash_length + gap_length
        
        num_segments = int(distance / total_length)
        for i in range(num_segments):
            # Draw dash
            t1 = (i * total_length) / distance
            t2 = min(((i * total_length) + dash_length) / distance, 1.0)
            
            start_x = int(x1 + (x2 - x1) * t1)
            start_y = int(y1 + (y2 - y1) * t1)
            end_x = int(x1 + (x2 - x1) * t2)
            end_y = int(y1 + (y2 - y1) * t2)
            
            cv2.line(self.canvas, (start_x, start_y), (end_x, end_y), self.current_color, self.brush_size)

    def draw_ui_elements(self, frame):
        """Draw UI elements on the frame"""
        h, w, _ = frame.shape
        
        if not self.show_ui:
            return
        
        # Draw color indicator
        color_name = self.color_names[self.current_color_index]
        cv2.rectangle(frame, (w - 150, 10), (w - 10, 50), self.current_color, -1)
        cv2.rectangle(frame, (w - 150, 10), (w - 10, 50), (255, 255, 255), 2)
        cv2.putText(frame, color_name.upper(), (w - 140, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show additional status info in top-right
        cv2.putText(frame, f"Colors: {len(self.color_names)}", (w - 140, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Brush: {self.brush_size}px", (w - 140, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show gesture recognition status
        gesture_status = "Gestures: ON" if hasattr(self, 'gesture_recognizer') else "Gestures: OFF"
        cv2.putText(frame, gesture_status, (w - 140, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw drawing status
        drawing_status = "DRAWING ON" if self.drawing_enabled else "DRAWING OFF"
        status_color = (0, 255, 0) if self.drawing_enabled else (0, 0, 255)
        cv2.putText(frame, drawing_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 2)
        
        # Draw brush size indicator
        cv2.circle(frame, (50, 70), self.brush_size, self.current_color, 2)
        cv2.putText(frame, f"Brush: {self.brush_size}", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show brush style if available
        if hasattr(self, 'brush_style'):
            styles = ["Normal", "Dotted", "Dashed"]
            style_name = styles[self.brush_style] if self.brush_style < len(styles) else "Normal"
            cv2.putText(frame, f"Style: {style_name}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show undo history count
        history_count = len(self.drawing_history) if hasattr(self, 'drawing_history') else 0
        cv2.putText(frame, f"Undo: {history_count}/{self.max_history}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show current mode (Drawing/Eraser/Shape)
        if hasattr(self, 'eraser_mode') and self.eraser_mode:
            cv2.putText(frame, "Mode: ERASER", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        elif hasattr(self, 'shape_mode') and self.shape_mode:
            shapes = ["Circle", "Rectangle", "Line"]
            shape_name = shapes[getattr(self, 'current_shape', 0)]
            cv2.putText(frame, f"Mode: SHAPE ({shape_name})", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        else:
            cv2.putText(frame, "Mode: DRAWING", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw ML gesture info
        cv2.putText(frame, f"ML Samples: {len(self.gesture_recognizer.gesture_data)}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show UI visibility status
        ui_status = "UI: ON" if self.show_ui else "UI: OFF"
        cv2.putText(frame, ui_status, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw controls help - Complete feature list
        controls = [
            "ML GESTURES (Hold 1.5s): Peace=Save | Thumbs=Color | Fist=Clear | Rock=Undo | Point=Toggle | Palm=Style",
            "KEYBOARD: SPACE=Toggle | S=Save | K=Color | U=Undo | C=Clear | B=Style | F=Fullscreen | H=UI | ESC=Exit",
            "ARROWS: UP/DOWN=Brush Size | SHAPES: 1=Circle | 2=Rectangle | 3=Line | DRAWING: Point finger | ERASE: Pinch"
        ]
        
        # Position controls text much higher to avoid overlap with gesture feedback (3 lines now)
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (10, h - 140 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def clear_canvas(self):
        """Clear the entire canvas"""
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)

    def run(self):
        # Try different camera indices if default fails
        cap = None
        for camera_index in [0, 1, 2]:
            print(f"Trying camera index {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Camera {camera_index} opened successfully!")
                break
            else:
                cap.release()
        
        if not cap or not cap.isOpened():
            print("Error: Could not open any camera!")
            print("Troubleshooting:")
            print("   1. Make sure no other apps are using the camera")
            print("   2. Check camera permissions")
            print("   3. Try closing other camera apps (Zoom, Teams, etc.)")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Wait for camera to initialize
        import time
        time.sleep(1)
        
        # Test frame capture
        for i in range(5):
            ret, test_frame = cap.read()
            if ret:
                print(f"Camera frame capture working!")
                break
            else:
                print(f"Retrying frame capture... ({i+1}/5)")
                time.sleep(0.5)
        
        if not ret:
            print("Camera opened but cannot capture frames!")
            print("Try:")
            print("   1. Close all other camera applications")
            print("   2. Restart the application")
            print("   3. Check Windows camera privacy settings")
            cap.release()
            return
        
        print("Enhanced Virtual Notepad - Smart Drawing Mode + ML Gestures")
        print("=" * 70)
        print("FEATURES:")
        print("- Smart Drawing: Point with index finger to draw")
        print("- Eraser: Pinch thumb and index finger together")
        print("- ML Gesture Commands: Use hand gestures for controls")
        print("- Multiple Colors: 8 different colors available")
        print("- Stop/Start Drawing: Toggle drawing on/off")
        print("- Save Drawings: Save your artwork as PNG files")
        print("- Undo Function: Undo up to 10 recent actions")
        print("- Shape Tools: Add circles, rectangles, and lines")
        print("- Brush Styles: Normal, dotted, and dashed lines")
        print("=" * 70)
        print("ML GESTURE COMMANDS (Hold for 1.5 seconds):")
        print("- Peace Sign: Save drawing")
        print("- Thumbs Up: Change color")
        print("- Fist: Clear canvas")
        print("- Rock Sign: Undo action")
        print("- Point Up: Toggle drawing on/off")
        print("- Open Palm: Change brush style")
        print("=" * 70)
        print("KEYBOARD CONTROLS:")
        print("- SPACEBAR: Toggle Drawing ON/OFF")
        print("- S: Save current drawing")
        print("- K: Change drawing color")
        print("- UP/DOWN Arrow: Change brush size")
        print("- U: Undo last action")
        print("- B: Change brush style")
        print("- 1/2/3: Add Circle/Rectangle/Line at center")
        print("- C: Clear canvas")
        print("- F: Toggle fullscreen")
        print("- H: Toggle UI visibility")
        print("- ESC: Exit")
        print("=" * 70)
        
        window_name = "Smart Virtual Notepad + ML Gestures"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        if self.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        gesture_command_result = None
        gesture_command_time = None

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("❌ Failed to capture frame from camera.")
                break

            frame = cv2.flip(frame, 1)  # Mirror effect
            h, w, _ = frame.shape
            
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

            landmarks_list = self.tracker.get_landmarks(frame)

            if landmarks_list:
                for landmarks in landmarks_list:
                    # ===== ML GESTURE RECOGNITION =====
                    current_gesture, is_triggered = self.gesture_recognizer.process_gesture(landmarks)
                    
                    # Execute gesture command
                    if is_triggered and current_gesture:
                        result = self.execute_gesture_command(current_gesture)
                        if result:
                            gesture_command_result = result
                            gesture_command_time = time.time()
                    
                    # Draw gesture feedback
                    self.gesture_recognizer.draw_gesture_feedback(frame, current_gesture, is_triggered)
                    
                    # Draw hand landmarks
                    self.tracker.mp_draw.draw_landmarks(
                        frame, landmarks, self.tracker.hand_connections,
                        self.tracker.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1),
                        self.tracker.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1)
                    )

                    # Get index finger tip position
                    index_finger = landmarks.landmark[self.tracker.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(index_finger.x * w), int(index_finger.y * h)

                    # Detect erasing gesture
                    is_erasing = self.detect_erasing_gesture(landmarks)
                    
                    if is_erasing:
                        # ===== ERASING MODE =====
                        cv2.circle(self.canvas, (x, y), self.eraser_size, (0, 0, 0), -1)
                        
                        # Visual feedback for eraser
                        cv2.circle(frame, (x, y), self.eraser_size, (255, 255, 0), 3)
                        cv2.putText(frame, "🧹 ERASING", (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                    elif self.drawing_enabled and not is_triggered:  # Don't draw during gesture commands
                        # ===== DRAWING MODE =====
                        if self.prev_x is not None and self.prev_y is not None:
                            # Add current state to history before drawing
                            if not hasattr(self, 'last_history_frame') or time.time() - self.last_history_frame > 1.0:
                                self.add_to_history()
                                self.last_history_frame = time.time()
                            
                            # Draw based on brush style
                            if hasattr(self, 'brush_style'):
                                if self.brush_style == 1:  # Dotted
                                    self.draw_dotted_line(self.prev_x, self.prev_y, x, y)
                                elif self.brush_style == 2:  # Dashed
                                    self.draw_dashed_line(self.prev_x, self.prev_y, x, y)
                                else:  # Normal
                                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), self.current_color, self.brush_size)
                            else:
                                cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), self.current_color, self.brush_size)
                        
                        # Visual feedback for drawing
                        cv2.circle(frame, (x, y), self.brush_size + 2, self.current_color, 2)
                        color_name = self.color_names[self.current_color_index]
                        cv2.putText(frame, f"✏️ {color_name.upper()}", (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.current_color, 2)
                    
                    elif not self.drawing_enabled:
                        # ===== DRAWING DISABLED =====
                        cv2.circle(frame, (x, y), 10, (128, 128, 128), 2)
                        cv2.putText(frame, "⏸️ PAUSED", (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

                    # Update previous position only when not doing gestures
                    if not is_triggered:
                        self.prev_x, self.prev_y = x, y
            else:
                self.prev_x, self.prev_y = None, None

            # Show gesture command result
            if gesture_command_result and gesture_command_time:
                if time.time() - gesture_command_time < 3.0:  # Show for 3 seconds
                    cv2.putText(frame, f"🤖 {gesture_command_result}", (w//2 - 100, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    gesture_command_result = None

            # Combine canvas with live camera feed
            combined = cv2.addWeighted(frame, 0.6, self.canvas, 0.4, 0)
            
            # Draw UI elements
            self.draw_ui_elements(combined)

            cv2.imshow(window_name, combined)
            
            # ===== KEYBOARD CONTROLS =====
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                # Save gesture data before exit
                self.gesture_recognizer.save_gesture_data()
                break
            elif key == 32:  # SPACEBAR - Toggle drawing
                self.toggle_drawing()
            elif key == ord('s') or key == ord('S'):  # Save drawing
                result = self.save_drawing()
                print(f"{result}")
            elif key == ord('u') or key == ord('U'):  # Undo
                self.undo_last_action()
            elif key == ord('b') or key == ord('B'):  # Change brush style
                self.change_brush_style()
            elif key == ord('1'):  # Add circle
                h, w, _ = frame.shape
                self.add_to_history()
                self.add_shape("circle", w // 2, h // 2)
            elif key == ord('2'):  # Add rectangle
                h, w, _ = frame.shape
                self.add_to_history()
                self.add_shape("rectangle", w // 2, h // 2)
            elif key == ord('3'):  # Add line
                h, w, _ = frame.shape
                self.add_to_history()
                self.add_shape("line", w // 2, h // 2)
            elif key == ord('k') or key == ord('K'):  # Change color
                self.change_color()
            elif key == ord('c') or key == ord('C'):  # Clear canvas
                self.add_to_history()
                self.clear_canvas()
                print("🧹 Canvas cleared!")
            elif key == ord('f') or key == ord('F'):  # Toggle fullscreen
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print(f"🔳 Fullscreen: {'ON' if self.fullscreen else 'OFF'}")
            elif key == ord('h') or key == ord('H'):  # Toggle UI
                self.show_ui = not self.show_ui
                print(f"📺 UI: {'VISIBLE' if self.show_ui else 'HIDDEN'}")
            elif key == 0:  # UP arrow - Increase brush size
                self.brush_size = min(self.brush_size + 1, 20)
                print(f"🖌️ Brush size: {self.brush_size}")
            elif key == 1:  # DOWN arrow - Decrease brush size
                self.brush_size = max(self.brush_size - 1, 1)
                print(f"🖌️ Brush size: {self.brush_size}")

        cap.release()
        cv2.destroyAllWindows()
