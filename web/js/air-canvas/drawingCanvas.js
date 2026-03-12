import { STROKE, GESTURE } from './constants.js';

// Exponential Moving Average alpha.
// 0.65 = ~0.54 frame lag at 30fps (~18ms) – responsive with natural jitter suppression.
// Lower = smoother but laggier. Higher = faster but jittery.
const EMA_ALPHA = 0.65;

export class DrawingCanvas {
  constructor(canvas) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) throw new Error('Could not get 2D context');
    this.ctx = ctx;

    this.currentStroke = null;
    this.completedStrokes = [];
    this.livePosition = null;
    this.emaPosition = null; // replaces filteredPosition + recentPoints

    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
  }

  startStroke(point, color) {
    this.currentStroke = {
      points: [point],
      color,
      width: STROKE.WIDTH,
      closed: false
    };
    this.livePosition = point;
    this.emaPosition = point; // seed EMA so first move isn't jarring
  }

  addPoint(point) {
    if (!this.currentStroke) return;

    const smoothed = this.applyEMA(point);
    this.livePosition = smoothed;

    const lastPoint = this.currentStroke.points[this.currentStroke.points.length - 1];
    if (this.distance(smoothed, lastPoint) >= STROKE.MIN_POINT_DISTANCE) {
      this.currentStroke.points.push({ ...smoothed });
    }
  }

  // Single-pole EMA: output = alpha*input + (1-alpha)*previous
  // Effective lag = (1-alpha)/alpha frames ≈ 1 frame at alpha=0.5
  applyEMA(point) {
    if (!this.emaPosition) {
      this.emaPosition = { ...point };
      return this.emaPosition;
    }
    this.emaPosition = {
      x: EMA_ALPHA * point.x + (1 - EMA_ALPHA) * this.emaPosition.x,
      y: EMA_ALPHA * point.y + (1 - EMA_ALPHA) * this.emaPosition.y
    };
    return this.emaPosition;
  }

  updateLivePosition(point) {
    this.livePosition = this.applyEMA(point);
  }

  clearLivePosition() {
    this.livePosition = null;
    this.emaPosition = null;
  }

  distance(p1, p2) {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  pauseStroke() {
    // Stroke remains but we stop adding points
  }

  closeStroke() {
    if (!this.currentStroke) return null;

    const length = this.calculateStrokeLength();
    if (length < GESTURE.MIN_STROKE_LENGTH) {
      this.discardStroke();
      return null;
    }

    if (this.currentStroke.points.length > 2) {
      this.currentStroke.closed = true;
      const closedStroke = { ...this.currentStroke };
      this.completedStrokes.push(closedStroke);
      this.currentStroke = null;
      return closedStroke;
    }

    this.discardStroke();
    return null;
  }

  discardStroke() {
    this.currentStroke = null;
  }

  calculateStrokeLength() {
    if (!this.currentStroke || this.currentStroke.points.length < 2) return 0;

    let length = 0;
    for (let i = 1; i < this.currentStroke.points.length; i++) {
      length += this.distance(this.currentStroke.points[i - 1], this.currentStroke.points[i]);
    }
    return length;
  }

  getCurrentStroke() {
    return this.currentStroke;
  }

  clearAll() {
    this.currentStroke = null;
    this.completedStrokes = [];
    this.clear();
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  render() {
    this.clear();

    for (const stroke of this.completedStrokes) {
      this.renderStroke(stroke, 0.3);
    }

    if (this.currentStroke && this.currentStroke.points.length >= 1) {
      this.renderStrokeWithLiveExtension(this.currentStroke, 1.0);
    }
  }

  renderStrokeWithLiveExtension(stroke, alpha) {
    if (stroke.points.length === 0) return;

    this.ctx.save();
    this.ctx.globalAlpha = alpha;
    this.ctx.fillStyle = stroke.color;
    this.ctx.strokeStyle = stroke.color;
    this.ctx.lineWidth = stroke.width;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    let points = [...stroke.points];
    if (this.livePosition) {
      points.push(this.livePosition);
    }

    if (points.length === 1) {
      this.ctx.beginPath();
      this.ctx.arc(points[0].x, points[0].y, stroke.width / 2, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.restore();
      return;
    }

    this.drawSmoothCurve(points);
    this.ctx.stroke();
    this.ctx.restore();
  }

  drawSmoothCurve(points) {
    if (points.length < 2) return;

    this.ctx.beginPath();
    this.ctx.moveTo(points[0].x, points[0].y);

    if (points.length === 2) {
      this.ctx.lineTo(points[1].x, points[1].y);
      return;
    }

    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[i];
      const p1 = points[i + 1];

      if (i === 0) {
        const midX = (p0.x + p1.x) / 2;
        const midY = (p0.y + p1.y) / 2;
        this.ctx.lineTo(midX, midY);
      } else if (i === points.length - 2) {
        this.ctx.quadraticCurveTo(p0.x, p0.y, p1.x, p1.y);
      } else {
        const midX = (p0.x + p1.x) / 2;
        const midY = (p0.y + p1.y) / 2;
        this.ctx.quadraticCurveTo(p0.x, p0.y, midX, midY);
      }
    }
  }

  renderStroke(stroke, alpha) {
    if (stroke.points.length === 0) return;

    this.ctx.save();
    this.ctx.globalAlpha = alpha;
    this.ctx.fillStyle = stroke.color;
    this.ctx.strokeStyle = stroke.color;
    this.ctx.lineWidth = stroke.width;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';

    if (stroke.points.length === 1) {
      this.ctx.beginPath();
      this.ctx.arc(stroke.points[0].x, stroke.points[0].y, stroke.width / 2, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.restore();
      return;
    }

    const points = stroke.closed ? [...stroke.points, stroke.points[0]] : [...stroke.points];
    this.drawSmoothCurve(points);
    this.ctx.stroke();
    this.ctx.restore();
  }

  renderClosingAnimation(stroke, progress) {
    if (stroke.points.length < 2) return;

    this.clear();

    const pulseScale = 1 + Math.sin(progress * Math.PI) * 0.1;
    const pulseAlpha = 0.5 + Math.sin(progress * Math.PI * 2) * 0.5;

    this.ctx.save();

    this.ctx.globalAlpha = pulseAlpha * 0.3;
    this.ctx.strokeStyle = stroke.color;
    this.ctx.lineWidth = stroke.width * pulseScale * 2;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.filter = 'blur(8px)';
    this.drawStrokePath(stroke);
    this.ctx.stroke();

    this.ctx.filter = 'none';
    this.ctx.globalAlpha = 1;
    this.ctx.lineWidth = stroke.width * pulseScale;
    this.drawStrokePath(stroke);
    this.ctx.stroke();

    this.ctx.restore();
  }

  drawStrokePath(stroke) {
    const points = stroke.closed ? [...stroke.points, stroke.points[0]] : [...stroke.points];
    this.drawSmoothCurve(points);
  }

  removeCompletedStroke(stroke) {
    const index = this.completedStrokes.indexOf(stroke);
    if (index > -1) {
      this.completedStrokes.splice(index, 1);
    }
  }
}
