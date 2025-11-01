"""
Quick setup script for the Gesture Recognition Demo
Checks if required packages are installed and installs them if needed.
"""

import subprocess
import sys
import shutil

def check_python_version():
    """Check if Python is installed and meets requirements."""
    print("Checking Python installation...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"Python {version.major}.{version.minor}.{version.micro} is installed")
        return True
    else:
        print(f"ERROR: Python 3.7+ required. Found Python {version.major}.{version.minor}")
        print("Please install Python 3.7 or higher from https://www.python.org/downloads/")
        return False

def check_pip():
    """Check if pip is installed."""
    print("Checking pip installation...")
    if shutil.which("pip") is not None:
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            print(f"pip is installed: {result.stdout.strip()}")
            return True
        except Exception as e:
            print(f"Warning: Could not verify pip: {e}")
    else:
        print("pip not found in PATH, trying python -m pip...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            print(f"pip is available: {result.stdout.strip()}")
            return True
        except Exception:
            print("ERROR: pip is not installed")
            print("Please install pip or reinstall Python with pip included")
            return False
    return False

def upgrade_pip():
    """Upgrade pip to latest version."""
    try:
        print("Upgrading pip to latest version...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("pip upgraded successfully")
        return True
    except Exception as e:
        print(f"Warning: Could not upgrade pip: {e}")
        print("Continuing anyway...")
        return False

def check_and_install_package(package_name, import_name=None):
    """Check if a package is installed, install if not."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"{package_name} is already installed")
        return True
    except ImportError:
        print(f"{package_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"{package_name} installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to install {package_name}")
            return False

def main():
    print("=" * 60)
    print("Gesture Recognition Demo - Setup")
    print("=" * 60)
    print()
    
    # Check Python
    if not check_python_version():
        print()
        print("=" * 60)
        print("Setup failed: Python 3.7+ required")
        print("=" * 60)
        sys.exit(1)
    print()
    
    # Check pip
    if not check_pip():
        print()
        print("=" * 60)
        print("Setup failed: pip required")
        print("=" * 60)
        sys.exit(1)
    print()
    
    # Upgrade pip
    upgrade_pip()
    print()
    
    required_packages = [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy")
    ]
    
    all_installed = True
    for package, import_name in required_packages:
        if not check_and_install_package(package, import_name):
            all_installed = False
        print()
    
    if all_installed:
        print("=" * 60)
        print("Setup complete! All packages are installed.")
        print()
        print("To run the demo:")
        print("  python demo_gesture_detection.py")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Setup incomplete. Please install missing packages manually:")
        print("  pip install opencv-python mediapipe numpy")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()

