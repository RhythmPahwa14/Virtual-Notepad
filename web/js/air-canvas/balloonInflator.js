import * as THREE from 'three';
import { SCENE } from './constants.js';

export class BalloonInflator {
  constructor(camera, canvasWidth, canvasHeight) {
    this.camera = camera;
    this.canvasWidth = canvasWidth;
    this.canvasHeight = canvasHeight;
  }

  updateSize(width, height) {
    this.canvasWidth = width;
    this.canvasHeight = height;
  }

  createBalloonMesh(stroke) {
    const simplifiedPoints = this.simplifyPoints(stroke.points, 20);
    const shape = this.createShape(simplifiedPoints);

    const extrudeSettings = {
      depth: SCENE.OBJECT_DEPTH,
      bevelEnabled: true,
      bevelThickness: 0.3,
      bevelSize: 0.3,
      bevelOffset: 0,
      bevelSegments: 8,
      curveSegments: 24
    };

    let geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
    geometry.center();
    geometry = this.inflateGeometry(geometry);

    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(stroke.color),
      roughness: 0.35,
      metalness: 0.0,
      side: THREE.DoubleSide
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    const center = this.getStrokeCenter(stroke.points);
    const worldPos = this.screenToWorld(center.x, center.y);
    mesh.position.copy(worldPos);

    const strokeSize = this.getStrokeSize(stroke.points);
    const scale = Math.max(strokeSize.width, strokeSize.height) / 600;
    mesh.scale.set(scale, scale, scale);

    return mesh;
  }

  simplifyPoints(points, targetCount) {
    if (points.length <= targetCount) return points;

    const step = points.length / targetCount;
    const simplified = [];
    for (let i = 0; i < targetCount; i++) {
      simplified.push(points[Math.min(Math.floor(i * step), points.length - 1)]);
    }
    return simplified;
  }

  createShape(points) {
    if (points.length < 3) {
      const shape = new THREE.Shape();
      shape.absarc(0, 0, 1, 0, Math.PI * 2, false);
      return shape;
    }

    const center = this.getStrokeCenter(points);
    const normalizedPoints = points.map(p => ({
      x: (p.x - center.x) / 50,
      y: -(p.y - center.y) / 50
    }));

    const curve = new THREE.CatmullRomCurve3(
      normalizedPoints.map(p => new THREE.Vector3(p.x, p.y, 0)),
      true,
      'catmullrom',
      0.5
    );

    const sampledPoints = curve.getPoints(64);
    const shape = new THREE.Shape();
    shape.moveTo(sampledPoints[0].x, sampledPoints[0].y);
    for (let i = 1; i < sampledPoints.length; i++) {
      shape.lineTo(sampledPoints[i].x, sampledPoints[i].y);
    }
    shape.closePath();

    return shape;
  }

  inflateGeometry(geometry) {
    const position = geometry.getAttribute('position');
    const normal = geometry.getAttribute('normal');
    if (!position || !normal) return geometry;

    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);

    const inflatedPositions = new Float32Array(position.count * 3);
    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i);
      const y = position.getY(i);
      const z = position.getZ(i);
      const dx = x - center.x;
      const dy = y - center.y;
      const dz = z - center.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (dist > 0.001) {
        const inflateAmount = SCENE.INFLATION_AMOUNT * 0.1;
        inflatedPositions[i * 3]     = x + (dx / dist) * inflateAmount;
        inflatedPositions[i * 3 + 1] = y + (dy / dist) * inflateAmount;
        inflatedPositions[i * 3 + 2] = z + (dz / dist) * inflateAmount;
      } else {
        inflatedPositions[i * 3]     = x;
        inflatedPositions[i * 3 + 1] = y;
        inflatedPositions[i * 3 + 2] = z;
      }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(inflatedPositions, 3));
    geometry.computeVertexNormals();
    return geometry;
  }

  getStrokeCenter(points) {
    const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
    return { x: sum.x / points.length, y: sum.y / points.length };
  }

  getStrokeSize(points) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of points) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    return { width: maxX - minX, height: maxY - minY };
  }

  screenToWorld(screenX, screenY) {
    const vector = new THREE.Vector3(
      (screenX / this.canvasWidth) * 2 - 1,
      -(screenY / this.canvasHeight) * 2 + 1,
      0.5
    );
    vector.unproject(this.camera);

    const dir = vector.sub(this.camera.position).normalize();
    const distance = -this.camera.position.z / dir.z;
    return this.camera.position.clone().add(dir.multiplyScalar(distance));
  }
}
