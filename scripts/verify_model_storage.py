#!/usr/bin/env python3
"""
Script to verify model storage location and accessibility
"""

import os
import sys
import argparse
from pathlib import Path

def check_model_storage(model_dir: str = None):
    """Check if models are stored correctly and are accessible"""
    
    if model_dir is None:
        model_dir = os.getenv("MODEL_DIR", "./ai_service/models")
    
    print(f"🔍 Checking model storage at: {model_dir}")
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory does not exist: {model_dir}")
        return False
    
    print(f"✅ Model directory exists: {model_dir}")
    
    # List model versions
    try:
        model_versions = [d for d in os.listdir(model_dir) 
                         if os.path.isdir(os.path.join(model_dir, d)) 
                         and d.startswith('v')]
        
        if not model_versions:
            print("⚠️  No model versions found")
            return True
        
        print(f"📦 Found {len(model_versions)} model versions:")
        for version in sorted(model_versions, reverse=True)[:5]:  # Show latest 5
            version_path = os.path.join(model_dir, version)
            files = os.listdir(version_path)
            print(f"  - {version}: {len(files)} files")
            
            # Check for essential files
            essential_files = ['model.pkl', 'metadata.json']
            for file in essential_files:
                if file in files:
                    file_path = os.path.join(version_path, file)
                    size = os.path.getsize(file_path)
                    print(f"    ✅ {file} ({size} bytes)")
                else:
                    print(f"    ❌ {file} (missing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking model storage: {e}")
        return False

def check_volume_mount():
    """Check if running in Docker and volume is mounted correctly"""
    if os.path.exists('/.dockerenv'):
        print("🐳 Running inside Docker container")
        
        # Check if models directory is writable
        model_dir = os.getenv("MODEL_DIR", "/app/models")
        test_file = os.path.join(model_dir, ".test_write")
        
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✅ Model directory is writable: {model_dir}")
            return True
        except Exception as e:
            print(f"❌ Model directory is not writable: {e}")
            return False
    else:
        print("🖥️  Running on host machine")
        return True

def main():
    parser = argparse.ArgumentParser(description="Verify model storage")
    parser.add_argument("--model-dir", help="Model directory path")
    args = parser.parse_args()
    
    print("=== Model Storage Verification ===\n")
    
    # Check volume mount
    volume_ok = check_volume_mount()
    print()
    
    # Check model storage
    storage_ok = check_model_storage(args.model_dir)
    print()
    
    if volume_ok and storage_ok:
        print("🎉 Model storage verification PASSED")
        sys.exit(0)
    else:
        print("💥 Model storage verification FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
