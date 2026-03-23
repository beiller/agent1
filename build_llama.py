#!/usr/bin/env python3
"""
Script to download and compile llama.cpp from GitHub.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description: str, cwd: Path = None):
    """Run a shell command with error handling."""
    print(f"\n📌 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"   ✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"   ❌ Command not found: {cmd[0]}")
        return False


def get_cpu_cores():
    """Detect the number of CPU cores available."""
    try:
        cores = os.cpu_count()
        if cores:
            # Use all but one core to leave some headroom
            return max(1, cores - 1)
        return 4  # Default fallback
    except Exception:
        return 4


def detect_cuda():
    """Detect if CUDA is available on the system."""
    print("\n🔍 Checking for CUDA support...")
    
    # Check 1: Can we run nvidia-smi? (indicates NVIDIA GPU + drivers)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_info = result.stdout.strip()
        print(f"   ✅ NVIDIA GPU detected: {gpu_info.split(',')[0]}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ⚠️  No NVIDIA GPU or drivers found")
        return False
    
    # Check 2: Is nvcc compiler available? (indicates CUDA toolkit)
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version_line = result.stdout.strip().split('\n')[-1]
        print(f"   ✅ CUDA Toolkit detected: {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ⚠️  CUDA toolkit (nvcc) not found in PATH")
        print("       Install CUDA from: https://developer.nvidia.com/cuda-downloads")
        return False


def cleanup_old_build(build_dir: Path):
    """Remove existing llama.cpp directory for a fresh build."""
    if build_dir.exists():
        print(f"\n🗑️  Removing existing {build_dir.name} directory...")
        try:
            shutil.rmtree(build_dir)
            print(f"   ✅ Cleaned up old build directory")
        except Exception as e:
            print(f"   ❌ Error removing directory: {e}")
            return False
    return True


def check_model_exists(model_path: Path) -> bool:
    """Check if a model file exists."""
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Model found: {model_path.name}")
        print(f"      Size: {size_mb:.2f} MB")
        return True
    return False


def ask_user(question: str) -> bool:
    """Ask user a yes/no question."""
    while True:
        response = input(f"{question} [Y/n]: ").strip().lower()
        if response in ('', 'y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False
        print("   Please enter 'Y' or 'n'")


def build_llama():
    """Main function to clone and build llama.cpp."""
    
    # Configuration
    REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
    BUILD_DIR = Path("llama.cpp")
    COMMIT_HASH = "49bfddeca18e62fa3d39114a23e9fcbdf8a22388"  # Pin to specific commit
    
    print("=" * 60)
    print("🦙 llama.cpp Download & Build Script")
    print("=" * 60)
    
    # Clean up old build directory for fresh start
    if BUILD_DIR.exists():
        
        print(f"\n⚠️ Build exists, delete {BUILD_DIR} if you want to rebuild. ")
        return
        
    # Detect CPU cores for parallel build
    cpu_cores = get_cpu_cores()
    print(f"\n💻 Detected {cpu_cores} CPU cores for parallel compilation")
    
    # Detect CUDA support
    cuda_available = detect_cuda()
    
    if cuda_available:
        print("   🎮 Will enable CUDA GPU acceleration!")
    
    # Step 1: Clone the repository
    if not run_command(
        ["git", "clone", REPO_URL, str(BUILD_DIR)],
        "Cloning llama.cpp repository..."
    ):
        print("\n❌ Failed to clone repository. Exiting.")
        sys.exit(1)
    
    # Step 2: Checkout pinned commit hash
    if not run_command(
        ["git", "checkout", COMMIT_HASH],
        f"Checking out commit {COMMIT_HASH}...",
        cwd=BUILD_DIR
    ):
        print(f"\n❌ Failed to checkout commit {COMMIT_HASH}. Exiting.")
        sys.exit(1)
    
    # Step 3: Configure build with CMake (with CUDA if available)
    cmake_cmd = ["cmake", "-B", "build"]
    if cuda_available:
        cmake_cmd.extend(["-DGGML_CUDA=ON"])
        print("\n🔧 Enabling CUDA backend in CMake configuration")
    
    if not run_command(
        cmake_cmd,
        "Configuring build with CMake...",
        cwd=BUILD_DIR
    ):
        print("\n❌ Failed to configure build. Exiting.")
        sys.exit(1)
    
    # Step 4: Build the project with parallel jobs
    if not run_command(
        ["cmake", "--build", "build", "--config", "Release", "-j", str(cpu_cores)],
        f"Building llama.cpp with {cpu_cores} parallel jobs (this may take a few minutes)...",
        cwd=BUILD_DIR
    ):
        print("\n❌ Failed to build. Exiting.")
        sys.exit(1)
    
    # Step 5: Verify the build
    binaries = BUILD_DIR / "build" / "bin"
    if binaries.exists() and any(binaries.iterdir()):
        print(f"\n✅ Build completed successfully!")
        print(f"   Binaries located at: {binaries.absolute()}")
        print("\n📦 Available tools:")
        for binary in sorted(binaries.iterdir()):
            if binary.is_file():
                print(f"   - {binary.name}")
    else:
        print("\n⚠️  Build directory exists but no binaries found.")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Done! You can now use llama.cpp tools from the 'build/bin' directory.")
    print(f"   📌 Building from commit: {COMMIT_HASH}")
    if cuda_available:
        print("   🎮 CUDA GPU acceleration is ENABLED")
        print("   Usage example:")
        print(f"   {binaries}/llama-server --model your-model.gguf -ngl 99")
    else:
        print("   ⚙️  CPU-only mode (no CUDA detected)")
    

def main():
    
    build_llama()

    if ask_user("\nWould you like to download the Qwen3.5-27B-Q4_K_M model now?"):
        print("\n🚀 Running download_model.py...")
        try:
            subprocess.run([sys.executable, "download_model.py"], check=True)
            print("   ✅ Model download completed!")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Download failed: {e}")
        except FileNotFoundError:
            print("   ❌ download_model.py not found. Please create it first.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
