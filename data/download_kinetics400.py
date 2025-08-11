#!/usr/bin/env python3
import requests
import os
from tqdm import tqdm
import tarfile
from pathlib import Path
import gzip
import shutil

def download_with_progress(url, output_path):
    """Download file with progress bar"""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as file, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    output_dir = Path("kinetics400_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # Download the CSV files first (these are definitely available)
    csv_files = [
        "https://raw.githubusercontent.com/open-mmlab/mmaction2/master/tools/data/kinetics/kinetics400_train_list.txt",
        "https://raw.githubusercontent.com/open-mmlab/mmaction2/master/tools/data/kinetics/kinetics400_val_list.txt",
    ]
    
    for url in csv_files:
        filename = url.split('/')[-1]
        output_path = output_dir / filename
        download_with_progress(url, output_path)
        
    # Try downloading processed features (smaller and more reliable)
    processed_url = "https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/kinetics400_train.txt"
    output_path = output_dir / "kinetics400_train.txt"
    try:
        download_with_progress(processed_url, output_path)
    except Exception as e:
        print(f"Warning: Could not download processed features: {e}")
    
    # Try downloading mini-kinetics from alternative source
    print("\nAttempting to download mini-kinetics videos...")
    mini_url = "https://github.com/cvdfoundation/kinetics-dataset/raw/main/k400_miniset_256.tar.gz"
    mini_path = output_dir / "k400_miniset_256.tar.gz"
    
    try:
        download_with_progress(mini_url, mini_path)
        
        print("Extracting mini-kinetics archive...")
        with tarfile.open(mini_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
        print(f"Successfully extracted mini-kinetics to {output_dir}")
        
        # Remove the archive to save space
        os.remove(mini_path)
    except Exception as e:
        print(f"Could not download mini-kinetics: {e}")
        print("Will try alternative sources...")

if __name__ == "__main__":
    main()
