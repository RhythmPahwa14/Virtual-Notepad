// three is resolved via the importmap in air-canvas.html
import * as THREE from 'three';
import { SCENE } from './constants.js';

export class Scene3D {
  constructor(canvas) {
    this.scene = new THREE.Scene();

    this.cameraDistance = SCENE.CAMERA_Z;
    this.cameraTheta = 0;
    this.cameraPhi = Math.PI / 2;
    this.cameraTarget = new THREE.Vector3(0, 0, 0);

    this.camera = new THREE.PerspectiveCamera(
      SCENE.CAMERA_FOV,
      window.innerWidth / window.innerHeight,
      SCENE.CAMERA_NEAR,
      SCENE.CAMERA_FAR
    );
    this.updateCameraPosition();

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;

    this.setupLighting();

    this.clock = new THREE.Clock();
  }

  setupLighting() {
    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(this.ambientLight);

    this.directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    this.directionalLight.position.set(5, 10, 7);
    this.directionalLight.castShadow = true;
    this.directionalLight.shadow.mapSize.width = 2048;
    this.directionalLight.shadow.mapSize.height = 2048;
    this.directionalLight.shadow.camera.near = 0.1;
    this.directionalLight.shadow.camera.far = 50;
    this.directionalLight.shadow.camera.left = -10;
    this.directionalLight.shadow.camera.right = 10;
    this.directionalLight.shadow.camera.top = 10;
    this.directionalLight.shadow.camera.bottom = -10;
    this.directionalLight.shadow.bias = -0.0001;
    this.scene.add(this.directionalLight);

    const fillLight = new THREE.DirectionalLight(0xc9b8ff, 0.3);
    fillLight.position.set(-3, -5, 3);
    this.scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xffd4e5, 0.4);
    rimLight.position.set(-5, 3, -5);
    this.scene.add(rimLight);
  }

  resize(width, height) {
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  add(object) {
    this.scene.add(object);
  }

  remove(object) {
    this.scene.remove(object);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  getDeltaTime() {
    return this.clock.getDelta();
  }

  getElapsedTime() {
    return this.clock.getElapsedTime();
  }

  getCamera() {
    return this.camera;
  }

  getScene() {
    return this.scene;
  }

  screenToWorld(screenX, screenY, z = 0) {
    const vector = new THREE.Vector3(
      (screenX / window.innerWidth) * 2 - 1,
      -(screenY / window.innerHeight) * 2 + 1,
      0.5
    );
    vector.unproject(this.camera);

    const dir = vector.sub(this.camera.position).normalize();
    const distance = (z - this.camera.position.z) / dir.z;
    return this.camera.position.clone().add(dir.multiplyScalar(distance));
  }

  raycastObjects(screenX, screenY, objects) {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(
      (screenX / window.innerWidth) * 2 - 1,
      -(screenY / window.innerHeight) * 2 + 1
    );
    raycaster.setFromCamera(mouse, this.camera);
    return raycaster.intersectObjects(objects, true);
  }

  updateCameraPosition() {
    const x = this.cameraDistance * Math.sin(this.cameraPhi) * Math.sin(this.cameraTheta);
    const y = this.cameraDistance * Math.cos(this.cameraPhi);
    const z = this.cameraDistance * Math.sin(this.cameraPhi) * Math.cos(this.cameraTheta);

    this.camera.position.set(
      this.cameraTarget.x + x,
      this.cameraTarget.y + y,
      this.cameraTarget.z + z
    );
    this.camera.lookAt(this.cameraTarget);
  }

  orbitCamera(deltaTheta, deltaPhi) {
    this.cameraTheta += deltaTheta;
    this.cameraPhi = Math.max(0.1, Math.min(Math.PI - 0.1, this.cameraPhi + deltaPhi));
    this.updateCameraPosition();
  }

  zoomCamera(delta) {
    this.cameraDistance = Math.max(3, Math.min(30, this.cameraDistance + delta * 5));
    this.updateCameraPosition();
  }

  resetCamera() {
    this.cameraDistance = SCENE.CAMERA_Z;
    this.cameraTheta = 0;
    this.cameraPhi = Math.PI / 2;
    this.cameraTarget.set(0, 0, 0);
    this.updateCameraPosition();
  }
}
