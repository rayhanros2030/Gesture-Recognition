"""
Quick setup script for the Gesture Recognition Demo
Checks if required packages are installed and installs them if needed.
"""

import subprocess
import sys

def check_and_install_package(package_name, import_name=None):
    """Check if a package is installed, install if not."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"⚠️  {package_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def main():
    print("=" * 50)
    print("🎯 Gesture Recognition Demo - Setup")
    print("=" * 50)
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
        print("=" * 50)
        print("🎉 Setup complete! All packages are installed.")
        print()
        print("To run the demo:")
        print("  python demo_gesture_detection.py")
        print("=" * 50)
    else:
        print("=" * 50)
        print("⚠️  Setup incomplete. Please install missing packages manually.")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()

