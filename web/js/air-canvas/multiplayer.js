// Multiplayer uses PeerJS loaded from CDN as global window.Peer

export class Multiplayer {
  constructor() {
    this.peer = null;
    this.connections = new Map();
    this.roomCode = this.generateRoomCode();
    this.statusCallback = null;
    this.eventCallback = null;
    this.isHost = false;
  }

  generateRoomCode() {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    let code = '';
    for (let i = 0; i < 6; i++) {
      code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
  }

  getRoomCode() {
    return this.roomCode;
  }

  onStatusChange(callback) {
    this.statusCallback = callback;
  }

  onEvent(callback) {
    this.eventCallback = callback;
  }

  updateStatus(status, message) {
    if (this.statusCallback) {
      this.statusCallback(status, message);
    }
  }

  emitEvent(event) {
    if (this.eventCallback) {
      this.eventCallback(event);
    }
  }

  async initialize() {
    return new Promise((resolve, reject) => {
      const peerId = `aircanvas-${this.roomCode}`;

      this.peer = new window.Peer(peerId, {
        debug: 0
      });

      this.peer.on('open', (id) => {
        console.log('Connected to PeerJS with ID:', id);
        this.isHost = true;
        this.updateStatus('disconnected', 'Ready to connect');
        resolve();
      });

      this.peer.on('connection', (conn) => {
        this.handleConnection(conn);
      });

      this.peer.on('error', (err) => {
        console.error('PeerJS error:', err);
        if (err.type === 'unavailable-id') {
          this.roomCode = this.generateRoomCode();
          this.peer.destroy();
          this.initialize().then(resolve).catch(reject);
        } else if (err.type === 'peer-unavailable') {
          this.updateStatus('disconnected', 'Room not found');
        } else {
          this.updateStatus('disconnected', 'Connection error');
          reject(err);
        }
      });

      this.peer.on('disconnected', () => {
        this.updateStatus('disconnected', 'Disconnected from server');
      });
    });
  }

  handleConnection(conn) {
    const peerId = conn.peer;

    conn.on('open', () => {
      this.connections.set(peerId, conn);
      this.updateStatus('connected', `Connected (${this.connections.size} peer${this.connections.size > 1 ? 's' : ''})`);
      this.emitEvent({ type: 'peer_joined', peerId });
    });

    conn.on('data', (data) => {
      const event = data;
      this.emitEvent(event);

      if (this.isHost) {
        this.broadcastExcept(event, peerId);
      }
    });

    conn.on('close', () => {
      this.connections.delete(peerId);
      if (this.connections.size === 0) {
        this.updateStatus('disconnected', 'Not connected');
      } else {
        this.updateStatus('connected', `Connected (${this.connections.size} peer${this.connections.size > 1 ? 's' : ''})`);
      }
      this.emitEvent({ type: 'peer_left', peerId });
    });

    conn.on('error', (err) => {
      console.error('Connection error:', err);
      this.connections.delete(peerId);
    });
  }

  async joinRoom(code) {
    if (!this.peer) {
      throw new Error('Peer not initialized');
    }

    const targetPeerId = `aircanvas-${code.toUpperCase()}`;

    this.updateStatus('connecting', 'Connecting...');

    return new Promise((resolve, reject) => {
      const conn = this.peer.connect(targetPeerId, {
        reliable: true
      });

      const timeout = setTimeout(() => {
        conn.close();
        this.updateStatus('disconnected', 'Connection timeout');
        reject(new Error('Connection timeout'));
      }, 10000);

      conn.on('open', () => {
        clearTimeout(timeout);
        this.handleConnection(conn);
        this.isHost = false;
        resolve();
      });

      conn.on('error', (err) => {
        clearTimeout(timeout);
        console.error('Failed to connect:', err);
        this.updateStatus('disconnected', 'Failed to connect');
        reject(err);
      });
    });
  }

  broadcast(event) {
    this.connections.forEach((conn) => {
      if (conn.open) {
        conn.send(event);
      }
    });
  }

  broadcastExcept(event, excludePeerId) {
    this.connections.forEach((conn, peerId) => {
      if (conn.open && peerId !== excludePeerId) {
        conn.send(event);
      }
    });
  }

  isConnected() {
    return this.connections.size > 0;
  }

  getConnectionCount() {
    return this.connections.size;
  }

  disconnect() {
    this.connections.forEach((conn) => {
      conn.close();
    });
    this.connections.clear();
    this.updateStatus('disconnected', 'Not connected');
  }

  destroy() {
    this.disconnect();
    if (this.peer) {
      this.peer.destroy();
      this.peer = null;
    }
  }
}
