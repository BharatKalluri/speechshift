#!/usr/bin/env python3
"""
Simplified SpeechShift Daemon

A single-file daemon implementation that handles:
1. Start daemon and keep running in background
2. Toggle recording: start recording OR stop recording + transcribe + paste text
3. Simple state management: IDLE -> RECORDING -> TRANSCRIBING -> IDLE

Communication via simple Unix socket with text commands (no JSON).
Singleton behavior via socket file existence + ping test.
"""

import os
import signal
import socket
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from speechshift.core.logger import logger


class DaemonState(Enum):
    """Simple daemon state machine"""
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    TRANSCRIBING = "TRANSCRIBING"


class SimpleDaemon:
    """Simplified SpeechShift daemon - all functionality in one class"""
    
    def __init__(self):
        # Import here to avoid circular imports
        from speechshift.main import CONFIG, AUDIO_AVAILABLE, WHISPER_AVAILABLE
        
        self.socket_path = CONFIG["daemon_socket"]
        self.socket = None
        self.running = False
        self.state = DaemonState.IDLE
        
        # Initialize components
        self.audio_recorder = None
        self.transcriber = None
        
        if AUDIO_AVAILABLE:
            try:
                from speechshift.core.audio_recorder import AudioRecorder
                self.audio_recorder = AudioRecorder()
            except ImportError:
                logger.error("Audio recording not available")
                
        if WHISPER_AVAILABLE:
            from speechshift.core.whisper_transcriber import WhisperTranscriber
            self.transcriber = WhisperTranscriber()
    
    def is_daemon_running(self) -> bool:
        """Check if daemon is already running (socket file + ping test)"""
        if not self.socket_path.exists():
            return False
            
        # Try to connect and ping
        try:
            test_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            test_socket.settimeout(2.0)
            test_socket.connect(str(self.socket_path))
            
            # Send ping
            test_socket.sendall(b"ping\n")
            response = test_socket.recv(1024).decode().strip()
            test_socket.close()
            
            return response == "pong"
        except (socket.error, FileNotFoundError, ConnectionRefusedError):
            # Socket file exists but not responsive, clean it up
            try:
                self.socket_path.unlink()
            except:
                pass
            return False
    
    def start_daemon(self) -> bool:
        """Start the daemon process"""
        logger.info("Starting simplified SpeechShift daemon...")
        
        # Check if already running
        if self.is_daemon_running():
            logger.error("Daemon is already running")
            return False
            
        # Validate components
        if not self.audio_recorder:
            logger.error("Audio recorder not available. Install dependencies manually.")
            return False

        if not self.transcriber:
            logger.error("Whisper transcriber not available. Install dependencies manually.")
            return False
        
        # Create socket
        if not self._create_socket():
            return False
            
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.running = True
        self.state = DaemonState.IDLE
        
        logger.info("Daemon started successfully")
        
        try:
            # Main event loop
            while self.running:
                try:
                    # Accept client connection
                    if self.socket:
                        self.socket.settimeout(1.0)
                        client_socket, _ = self.socket.accept()
                    else:
                        break
                    
                    # Handle request
                    self._handle_client(client_socket)
                    
                except socket.timeout:
                    # Timeout is expected, continue loop
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting client: {e}")
                        
        except KeyboardInterrupt:
            logger.info("Daemon interrupted")
        finally:
            self._cleanup()
            
        return True
    
    def _create_socket(self) -> bool:
        """Create Unix domain socket"""
        try:
            # Remove existing socket file if present
            if self.socket_path.exists():
                self.socket_path.unlink()
                
            # Create socket
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.bind(str(self.socket_path))
            self.socket.listen(5)
            
            # Set permissions
            os.chmod(str(self.socket_path), 0o600)
            
            logger.info(f"Daemon socket created: {self.socket_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create socket: {e}")
            return False
    
    def _handle_client(self, client_socket: socket.socket):
        """Handle client request with simple text protocol"""
        try:
            # Receive command
            data = client_socket.recv(1024).decode().strip()
            if not data:
                return
                
            logger.info(f"Received command: {data}")
            
            # Process command
            if data == "ping":
                response = "pong"
            elif data == "toggle":
                response = self._handle_toggle()
            elif data == "status":
                response = self.state.value
            elif data == "shutdown":
                response = "OK shutting down"
                self.running = False
            else:
                response = f"ERROR unknown command: {data}"
            
            # Send response
            client_socket.sendall(f"{response}\n".encode())
            
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            try:
                client_socket.sendall(f"ERROR internal error: {e}\n".encode())
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Client disconnected - this is normal, don't log as error
                logger.debug(f"Client disconnected during error response: {e}")
            except:
                pass
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def _handle_toggle(self) -> str:
        """Handle toggle command - core daemon logic"""
        if self.state == DaemonState.TRANSCRIBING:
            return "ERROR busy transcribing"
        
        if self.state == DaemonState.IDLE:
            # Start recording
            if self._start_recording():
                self.state = DaemonState.RECORDING
                return "OK started recording"
            else:
                return "ERROR failed to start recording"
                
        elif self.state == DaemonState.RECORDING:
            # Stop recording and transcribe
            return self._stop_and_transcribe()
        
        return "ERROR invalid state"
    
    def _start_recording(self) -> bool:
        """Start audio recording"""
        if not self.audio_recorder:
            return False
            
        return self.audio_recorder.start_recording()
    
    def _stop_and_transcribe(self) -> str:
        """Stop recording, transcribe, and paste text"""
        if not self.audio_recorder:
            return "ERROR no audio recorder"
            
        self.state = DaemonState.TRANSCRIBING
        
        try:
            # Stop recording
            result = self.audio_recorder.stop_recording()
            if result is None:
                self.state = DaemonState.IDLE
                return "ERROR failed to stop recording"
            
            # Recording stopped successfully, transcription and pasting handled by AudioRecorder
            self.state = DaemonState.IDLE
            return "OK stopped recording and transcribed"
            
        except Exception as e:
            self.state = DaemonState.IDLE
            logger.error(f"Stop and transcribe failed: {e}")
            return f"ERROR transcription failed: {e}"
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _cleanup(self):
        """Clean up daemon resources"""
        logger.info("Cleaning up daemon...")
        
        # Stop any ongoing recording
        if self.state == DaemonState.RECORDING and self.audio_recorder:
            try:
                self.audio_recorder.stop_recording()
            except Exception as e:
                logger.warning(f"Error stopping recording during cleanup: {e}")
        
        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
                
        # Remove socket file
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except:
            pass
            
        logger.info("Daemon cleanup completed")


class SimpleDaemonClient:
    """Simple client for communicating with daemon"""
    
    def __init__(self):
        from speechshift.main import CONFIG
        self.socket_path = CONFIG["daemon_socket"]
    
    def is_daemon_running(self) -> bool:
        """Check if daemon is running"""
        daemon = SimpleDaemon()
        return daemon.is_daemon_running()
    
    def start_daemon(self) -> bool:
        """Start daemon process"""
        if self.is_daemon_running():
            logger.info("Daemon already running")
            return True
            
        logger.info("Starting daemon process...")
        
        # Start daemon in background
        import subprocess
        try:
            subprocess.Popen([
                sys.executable, __file__, "--daemon"
            ], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for daemon to be ready
            from speechshift.main import CONFIG
            start_time = time.time()
            timeout = CONFIG["daemon_startup_timeout"]
            
            while time.time() - start_time < timeout:
                time.sleep(0.2)
                if self.is_daemon_running():
                    logger.info("Daemon started successfully")
                    return True
                    
            logger.error("Daemon failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            return False
    
    def send_command(self, command: str) -> Optional[str]:
        """Send command to daemon"""
        # Ensure daemon is running
        if not self.is_daemon_running():
            if not self.start_daemon():
                return "ERROR failed to start daemon"
        
        try:
            # Connect to daemon
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.settimeout(5.0)
            client_socket.connect(str(self.socket_path))
            
            # Send command
            client_socket.sendall(f"{command}\n".encode())
            
            # Receive response
            response = client_socket.recv(1024).decode().strip()
            client_socket.close()
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return f"ERROR communication failed: {e}"


# CLI entry point for daemon mode
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--daemon":
        daemon = SimpleDaemon()
        sys.exit(0 if daemon.start_daemon() else 1)
    else:
        print("Usage: python speechshift_daemon.py --daemon")
        sys.exit(1)