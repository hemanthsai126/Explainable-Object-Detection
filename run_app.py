#!/usr/bin/env python3
"""
Quick launcher script for the Streamlit app
Run: python run_app.py
"""

import subprocess
import sys
import os

def main():
    # Change to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 60)
    print("🚀 Launching Explainable Object Detection Web App")
    print("=" * 60)
    print("\nStarting Streamlit server...")
    print("The app will open in your browser automatically.\n")
    print("If it doesn't open, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.")
    print("=" * 60 + "\n")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main()

