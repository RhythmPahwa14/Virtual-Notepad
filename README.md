# Virtual Notepad + ML Gestures

**An AI-powered hand tracking drawing application with machine learning gesture recognition**

## Features

- **Hand Tracking**: Draw using your index finger
- **ML Gesture Commands**: Control app with hand gestures (6 different gestures)
- **Smart Eraser**: Pinch thumb and index finger to erase
- **8 Colors**: Red, Blue, Green, Yellow, Purple, Orange, White, Black
- **3 Brush Styles**: Normal, Dotted, Dashed
- **Save Drawings**: Export as PNG files with timestamps
- **Undo System**: Undo up to 10 recent actions
- **Shape Tools**: Add circles, rectangles, and lines
- **Drawing Toggle**: Start/stop drawing mode
- **Data Collection**: Automatically collects gesture data for ML training
- **Intuitive Controls**: Gesture + keyboard shortcuts for all features

## Quick Start

### Prerequisites
- Python 3.8+
- Webcam/Camera

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/VirtualNotepad.git
   cd VirtualNotepad
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## ML Gesture Commands

**Hold gestures for 1.5 seconds to trigger:**

| Gesture | Command | Function |
|---------|---------|----------|
| Peace Sign | Save Drawing | Saves current artwork as PNG |
| Thumbs Up | Change Color | Cycles through 8 available colors |
| Fist | Clear Canvas | Clears the entire drawing area |
| Rock Sign | Undo Action | Undoes the last drawing action |
| Point Up | Toggle Drawing | Enables/disables drawing mode |
| Open Palm | Brush Style | Changes brush style (Normal/Dotted/Dashed) |

## Keyboard Controls

| Key | Function |
|-----|----------|
| **SPACEBAR** | Toggle Drawing ON/OFF |
| **S** | Save current drawing |
| **K** | Change drawing color |
| **UP/DOWN** | Change brush size |
| **U** | Undo last action |
| **B** | Change brush style |
| **1/2/3** | Add Circle/Rectangle/Line |
| **C** | Clear canvas |
| **F** | Toggle fullscreen |
| **H** | Toggle UI visibility |
| **ESC** | Exit application |

## Team

- **Rhythm** - Lead Developer
- **Chaitnya Dhar Dwivedi** - Co-Developer

*Collaborative project built with passion for computer vision and interactive applications.*

## Machine Learning Features

### Hybrid ML Approach
- **Phase 1**: Uses pre-trained MediaPipe models for hand tracking
- **Phase 2**: Collects user gesture data automatically during usage
- **Phase 3**: Future custom ML model training from collected data

### Data Collection
- Automatically saves gesture patterns to `gesture_data.json`
- Tracks gesture accuracy and user preferences
- No personal data collected - only hand landmark coordinates
- Data used for improving gesture recognition accuracy

### ML Training Pipeline (Future Enhancement)
- Custom CNN model for gesture classification
- Personalized gesture recognition for individual users
- Adaptive learning from user interactions

## Technical Stack

- **Computer Vision**: OpenCV
- **Hand Tracking**: MediaPipe  
- **Machine Learning**: TensorFlow (MediaPipe backend)
- **Gesture Recognition**: Custom ML pipeline (hybrid approach)
- **Numerical Computing**: NumPy
- **Language**: Python 3.10+

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Acknowledgments

- MediaPipe team for excellent hand tracking models
- OpenCV community for computer vision tools

---

**Star this repository if you found it helpful!**
