"""Utility scripts for XAI project."""
import os
import subprocess


def clean():
    """Clean build artifacts and cache files from the project."""
    # Replicate the clean command from setup.py
    clean_command = 'rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info ./**/__pycache__ ./__pycache__ ./.eggs ./.cache'

    try:
        # Use subprocess for better error handling than os.system
        result = subprocess.run(clean_command, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode == 0:
            print("Clean completed successfully")
        else:
            print(f"Clean completed with return code: {result.returncode}")
    except Exception as e:
        print(f"Error during clean: {e}")