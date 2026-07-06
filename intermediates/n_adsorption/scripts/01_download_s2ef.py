import os
import subprocess

def download_and_extract(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    filename = url.split('/')[-1]
    filepath = os.path.join(target_dir, filename)
    
    # Download the file using wget
    print(f"Downloading {filename}...")
    subprocess.run(["wget", url, "-O", filepath], check=True)
    
    # Extract the file
    print(f"Extracting to {target_dir}...")
    subprocess.run(["tar", "-xvf", filepath, "-C", target_dir], check=True)
    
    # Cleanup tar file
    os.remove(filepath)
    print("Done.")

# Example: Download the OC20 200k split
if __name__ == "__main__":
    url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar"
    download_and_extract(url, "data/oc20_200k")