#!/usr/bin/env python3
"""
Complete Threat Hunting ML Pipeline Runner
This script runs the entire pipeline: data generation, preprocessing, model training, and threat detection.
"""

import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print('='*50)

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Run the complete threat hunting pipeline."""
    print("Threat Hunting ML Pipeline")
    print("==========================")

    # Change to the project directory
    os.chdir('demo/threat_hunting_ml')

    # Step 1: Generate data
    if not run_command("python data_generator.py", "Data Generation"):
        sys.exit(1)

    # Step 2: Train models
    if not run_command("python model.py", "Model Training"):
        sys.exit(1)

    # Step 3: Run threat detection
    if not run_command("python detect.py", "Threat Detection"):
        sys.exit(1)

    print("\n" + "="*50)
    print("✓ Pipeline completed successfully!")
    print("="*50)
    print("\nNext steps:")
    print("1. Run 'python app.py' to start the web dashboard")
    print("2. Open http://localhost:5000 in your browser")
    print("3. Explore the threat detection results and visualizations")

if __name__ == "__main__":
    main()
