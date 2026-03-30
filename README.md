# Virtual Notepad

A browser-based gesture drawing experience where you draw shapes in the air using hand gestures via your webcam. Completed drawings inflate into soft, 3D balloon-like objects that float in a shared scene - no installation, just open and create.

## Recent Updates

- **Draggable Camera Preview** - Move the camera preview anywhere on screen so it doesn't block your creations. Double-click to reset position.
- **Improved Line Smoothing** - Jitter filter eliminates hand vibration for smoother strokes.
- **Better Hand Detection** - Higher accuracy model for more reliable tracking.
- **Clear All Button** - Dedicated button for easy clearing.
- **Expandable Preview** - Click the expand button on the camera preview to enlarge it.
- **Blue UI Theme** - Consistent blue accent colors matching the landing page.

## Features

- **Gesture-Based Drawing** - Point your index finger to draw in the air.
- **3D Balloon Inflation** - Completed shapes transform into puffy, floating 3D objects.
- **Real-Time Hand Tracking** - Powered by MediaPipe for responsive hand detection.
- **Color Palette** - Choose from multiple colors for your creations.
- **Interactive Objects** - Poke, grab, and rotate your balloon creations.
- **Draggable Camera Preview** - See your hand tracking skeleton and move it anywhere.
- **Multiplayer Support** - Share your canvas with others via PeerJS WebRTC.
- **Mouse/Touch Controls** - Orbit and zoom the 3D scene.

## How It Works

1. **Draw** - Extend your index finger (keep other fingers curled) to draw.
2. **Complete Shape** - Hold an open palm for 0.5 seconds to close and inflate your drawing.
3. **Interact** - Pinch to grab and move objects, poke with your finger to squish them.
4. **Clear** - Click the "Clear All" button to remove all objects.

## Gesture Controls

| Gesture | Action |
|---------|--------|
| Point (index finger) | Draw in the air |
| Open Palm (hold 0.5s) | Close shape and inflate to 3D |
| Pinch | Grab and move objects |
| Swipe | Remove individual object |

## UI Controls

| Control | Action |
|---------|--------|
| Clear All button | Remove all objects |
| Color swatches | Change drawing color |
| Camera preview | Drag to move, double-click to reset |
| Expand button (on preview) | Toggle larger preview |

## Deployment

### Vercel (Production)

This project is configured for zero-config Vercel deployment. Just connect the repository and Vercel will serve the `web/` directory automatically.

```bash
npm install -g vercel
vercel --prod
```

### Local Development

```bash
git clone https://github.com/RhythmPahwa14/Virtual-Notepad.git
cd Virtual-Notepad
npm install
npm run dev
```

Opens automatically at `http://localhost:3000`.

> **Note**: Camera access requires HTTPS in production. Vercel provides HTTPS automatically.

## Requirements

- Modern browser with WebGL support (Chrome, Firefox, Edge, Safari)
- Webcam access
- Good lighting for hand tracking
- HTTPS connection (required for camera access when hosted)

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| **JavaScript (ES Modules)** | Core application logic |
| **Three.js** | 3D rendering and scene management |
| **MediaPipe Hands** | Real-time hand tracking |
| **GSAP** | Smooth animations |
| **PeerJS** | WebRTC multiplayer support |
| **Vercel** | Production deployment |

## Project Structure

```
web/
 index.html              # Landing page
 air-canvas.html         # Main canvas application
 js/
     air-canvas/
         main.js             # Application entry point
         handTracking.js     # MediaPipe hand detection
         gestureDetector.js  # Gesture recognition logic
         drawingCanvas.js    # 2D stroke rendering
         scene3D.js          # Three.js scene setup
         objectManager.js    # 3D balloon creation and physics
         balloonInflator.js  # 3D mesh generation from strokes
         handVisualizer.js   # Hand skeleton overlay
         multiplayer.js      # PeerJS multiplayer support
         constants.js        # Configuration values
```

## Tips for Best Results

- Use good lighting so your hand is clearly visible.
- Keep your hand about 1-2 feet from the camera.
- Point with just your index finger extended for drawing.
- Draw slowly and steadily for smoother lines.

## Mouse/Touch Controls

- **Click + Drag** on empty space to orbit the camera.
- **Click + Drag** on an object to rotate it.
- **Scroll wheel** to zoom in/out.
- **Touch** gestures supported on mobile.

## Feedback & Contributions

Found a bug? Have an idea?

- Open an issue on GitHub
- Fork and submit a pull request
- Share your own version built on this project


### Open Source Libraries

| Library | Description | License |
|---------|-------------|---------|
| [Three.js](https://threejs.org/) | 3D graphics library for WebGL rendering | MIT |
| [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) | Google's real-time hand tracking | Apache 2.0 |
| [GSAP](https://greensock.com/gsap/) | Professional-grade animation library | GreenSock License |
| [PeerJS](https://peerjs.com/) | WebRTC peer-to-peer connections | MIT |

## License
MIT

## Contributing

We welcome contributions to improve the Virtual Notepad experience:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

*Built with passion for computer vision and creating intuitive user experiences.*
**Star this repository if you found it helpful and follow us for more innovative projects!**
