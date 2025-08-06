#!/usr/bin/env python3
"""
Script to ensure the server is using the latest code
"""

import os
import sys
import time
import signal
import psutil
import subprocess

def kill_existing_servers():
    """Kill any existing uvicorn processes"""
    print("Checking for existing server processes...")
    killed = False
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's a Python process running uvicorn or main.py
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('uvicorn' in str(arg) or 'main.py' in str(arg) for arg in cmdline):
                print(f"Found existing server process: PID {proc.info['pid']}")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
                print(f"Killed process {proc.info['pid']}")
                killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed:
        print("Waiting for ports to be released...")
        time.sleep(3)
    else:
        print("No existing server processes found")
    
    return killed

def check_port_available(port=8000):
    """Check if port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result != 0

def main():
    print("="*60)
    print("SERVER RESTART AND VERIFICATION")
    print("="*60)
    
    # Kill existing servers
    kill_existing_servers()
    
    # Check if port is available
    if not check_port_available(8000):
        print("\n⚠️  Port 8000 is still in use!")
        print("Please manually stop the server and try again.")
        print("\nYou can stop the server by:")
        print("1. Going to the terminal running the server")
        print("2. Pressing Ctrl+C")
        print("\nOr close the terminal window running the server.")
        return False
    
    print("\n✓ Port 8000 is available")
    print("\n" + "="*60)
    print("IMPORTANT: Next Steps")
    print("="*60)
    print("\n1. Start the server with the latest code:")
    print("   cd backend")
    print("   python app/main.py")
    print("\n2. Once the server is running, test with:")
    print("   python test_endpoints.py")
    print("\n" + "="*60)
    
    return True

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("Installing psutil...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    success = main()
    sys.exit(0 if success else 1)
