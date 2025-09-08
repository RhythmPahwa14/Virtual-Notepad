
# Virtual Notepad - AI-Powered Hand Gesture Recognition


**Real-time hand gesture recognition drawing application with TensorFlow.js and MediaPipe. Professional web-based virtual notepad that lets you draw and control with hand gestures.**


## Features

### Core Functionality
- **Real-time Hand Tracking**: Advanced MediaPipe integration for precise hand detection
- **Gesture-Based Drawing**: Draw using index finger with real-time hand tracking
- **Smart Drawing Controls**: Point up to draw, fist to stop, open palm to erase
- **Professional UI**: Modern design with Poppins typography and smooth animations
- **Multi-Color Support**: 6 color options with visual color picker
- **Canvas Controls**: Undo, clear, and save functionality
- **Responsive Design**: Works on desktop and mobile devices

### Advanced Gesture Recognition
- **6 Gesture Classes**: Fist, Open Palm, Peace, Point Up, Rock, Thumbs Up

- **Gesture Stabilization**: Intelligent gesture filtering to prevent false triggers
- **Confidence Scoring**: Real-time confidence levels for gesture accuracy
- **Rule-Based Recognition**: Optimized gesture detection without heavy ML models
- **Fallback System**: Works even when AI models are unavailable

### Visual Design
- **Modern Animations**: Smooth transitions and hover effects throughout the interface
- **Glass Morphism Effects**: Contemporary design with backdrop blur and transparency
- **Professional Typography**: Google Fonts Poppins integration for clean text rendering
- **Loading Experience**: Animated loading overlay with progress indicators
- **Interactive Elements**: Animated buttons, color selectors, and gesture displays

### Technical Features
- **Browser-Based**: No installation required, runs in any modern web browser
- **TensorFlow.js Ready**: Compatible with AI model integration
- **MediaPipe Integration**: Industry-standard hand tracking technology
- **Real-time Processing**: 30+ FPS gesture recognition and drawing
- **Error Handling**: Robust error management with user-friendly messages
- **Cross-Platform**: Compatible with Windows, macOS, and Linux
=======
- **Demo Mode**: Works even without trained models

## Quick Deploy

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

### Web Version (Recommended)
1. **Clone the repository**
   ```bash
   git clone https://github.com/RhythmPahwa14/Virtual-Notepad.git
   cd Virtual-Notepad
   ```

2. **Start local server**
   ```bash
   cd web
   python -m http.server 8000
   ```

3. **Open in browser**
   Navigate to `http://localhost:8000/working-notepad.html`

### Vercel Deployment
Deploy instantly to Vercel for production use:

```bash
npm install -g vercel
vercel --prod
```

## Gesture Controls

| Gesture | Function | Description |
|---------|----------|-------------|
| **Point Up** | Draw/Write | Use index finger to draw on canvas |
| **Fist** | Stop Drawing | Pause drawing mode |
| **Open Palm** | Erase | Erase area around hand position |
| **Peace** | Navigation | Navigate interface elements |
| **Rock** | Special Action | Context-specific actions |
| **Thumbs Up** | Confirm | Confirm actions and selections |

## Drawing Features

- **Natural Drawing**: Smooth line rendering with adjustable brush size
- **Color Palette**: Blue, Red, Green, Orange, Purple, Black color options
- **Undo System**: Step-by-step undo with history management
- **Save Functionality**: Export drawings as PNG files
- **Clear Canvas**: One-click canvas reset
- **Real-time Feedback**: Visual gesture recognition feedback

## Technical Architecture

### Frontend Stack
- **HTML5 Canvas**: High-performance drawing surface
- **CSS3 Animations**: Modern animations and transitions
- **Vanilla JavaScript**: Lightweight, no framework dependencies
- **MediaPipe**: Google's hand tracking solution
- **TensorFlow.js**: Machine learning in the browser

### Performance Optimizations
- **Gesture Stabilization**: Multi-frame gesture validation
- **Efficient Canvas Rendering**: Optimized drawing operations
- **Memory Management**: Proper cleanup and resource management
- **FPS Monitoring**: Real-time performance tracking
- **Error Recovery**: Graceful degradation on hardware limitations

### Browser Compatibility
- **Modern Browsers**: Chrome 80+, Firefox 75+, Safari 14+, Edge 80+
- **Camera Access**: Requires HTTPS for camera permissions
- **WebGL Support**: For TensorFlow.js acceleration
- **Canvas API**: HTML5 canvas support required

## Project Structure

```
Virtual-Notepad/
├── web/
│   ├── working-notepad.html    # Main application
│   ├── models/                 # AI model files
│   │   ├── model.json         # TensorFlow.js model
│   │   ├── weights.bin        # Model weights
│   │   └── model_info.json    # Model metadata
│   └── vercel.json            # Vercel deployment config
├── data/
│   └── gesture_data.json      # Training data
├── models/
│   ├── gesture_model.h5       # Python model
│   ├── label_encoder.pkl      # Label encoder
│   └── model_info.json       # Model information
└── deploy-vercel.ps1          # Deployment script
```

## Development Team

- **Rhythm Pahwa** - Lead Developer | [LinkedIn](https://www.linkedin.com/in/pahwa-rhythm/)
- **Chaitnya Dhar Dwivedi** - Co-Developer| [LinkedIn](https://www.linkedin.com/in/chaitnya-dhar-dwivedi-65333a255/)

*Built with passion for computer vision, machine learning, and creating intuitive user experiences.*

## Contributing

We welcome contributions to improve the Virtual Notepad experience:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow existing code style and conventions
- Test thoroughly on multiple browsers
- Update documentation for new features
- Ensure responsive design compatibility

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Star this repository if you found it helpful and follow us for more innovative projects!**
