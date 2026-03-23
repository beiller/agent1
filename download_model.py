#!/usr/bin/env python3
"""
Script to download a GGUF model from Hugging Face.
Uses local_dir to get a clean copy instead of cache symlinks.
"""

import os
from pathlib import Path


def check_huggingface_hub():
    """Check if huggingface_hub is installed."""
    try:
        from huggingface_hub import hf_hub_download
        return True
    except ImportError:
        print("❌ huggingface_hub not installed!")
        print("   Install it with: pip install huggingface_hub")
        return False


def download_model(
    repo_id: str = "bartowski/Qwen_Qwen3.5-27B-GGUF", 
    filename: str = "Qwen_Qwen3.5-27B-Q4_K_M.gguf"
):
    """Download a GGUF model from Hugging Face. Provide the repo_id and filename. Feel free to search huggingface first (curl) to find the appropriate values. """
    from huggingface_hub import hf_hub_download
    
    # Default save directory
    save_dir = Path(os.getenv("LLAMA_MODELS_DIR"))
    
    # Create models directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    if (save_dir / Path(filename)).exists():
        print(f"\n⚠️ File Already Exists!")
        return
    
    print("=" * 60)
    print("📥 Downloading GGUF Model from Hugging Face")
    print("=" * 60)
    print(f"\n📦 Repository: {repo_id}")
    print(f"📄 Filename: {filename}")
    print(f"💾 Save to: {save_dir.absolute()}")
    
    try:
        # Download the model file with local_dir_copy=True for a clean copy
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=save_dir,
            force_download=True,
            # This ensures we get the actual file, not a symlink
            local_dir=str(save_dir)
        )
        
        print(f"\n✅ Download completed!")
        print(f"   File saved to: {local_path}")
        
        # Get actual file size
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / (1024*1024)
            print(f"   File size: {size_mb:.2f} MB")
        return local_path
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return None
    


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # First load defaults from .env.example
    load_dotenv(".env.example", override=False)
    
    # Then load .env if it exists (overrides .env.example)
    if os.path.exists(".env"):
        load_dotenv(".env", override=True)

    download_model()
