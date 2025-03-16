import os
import sys
import urllib.request
import zipfile
import subprocess
from tqdm import tqdm

# URL to download the checkpoints (colorization_checkpoint.zip)
CHECKPOINT_URL = (
    "https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/"
    "releases/download/v1.0/colorization_checkpoint.zip"
)
ZIP_FILENAME = "colorization_checkpoint.zip"

def ensure_opencv_contrib():
    """
    Checks if cv2.ximgproc is available. If not, installs opencv-contrib-python.
    """
    try:
        import cv2
        _ = cv2.ximgproc  # just a test import
        print("[install.py] opencv-contrib-python is already installed.")
    except (ImportError, AttributeError):
        print("[install.py] Installing opencv-contrib-python for WLS filter support...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "opencv-contrib-python"])
        print("[install.py] opencv-contrib-python installed successfully.")

def download_file(url: str, dest_path: str) -> None:
    """
    Downloads a file from a URL to a local destination with a progress bar.
    
    Args:
        url (str): The URL from which to download the file.
        dest_path (str): The file path to save the downloaded file.
    """
    with tqdm(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(dest_path)) as t:
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                t.total = total_size
            t.update(block_size)
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)

def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extracts a zip file into the specified directory.
    
    Args:
        zip_path (str): The path to the zip file.
        extract_to (str): The directory where files will be extracted.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

def main() -> None:
    """
    Main function that downloads and extracts the checkpoint files.
    This will create the necessary `checkpoints/` and `data/` directories.
    """
    # 1) Ensure opencv-contrib-python is present (for cv2.ximgproc)
    ensure_opencv_contrib()

    # 2) Determine the root directory (where this script resides)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(root_dir, ZIP_FILENAME)

    # 3) Download the checkpoint zip if it's missing
    if not os.path.exists(zip_path):
        print(f"Downloading checkpoints from {CHECKPOINT_URL} ...")
        try:
            download_file(CHECKPOINT_URL, zip_path)
            print("Download complete.")
        except Exception as e:
            print(f"Error during download: {e}")
            return
    else:
        print(f"Zip file already exists: {zip_path}")

    # 4) Extract the downloaded zip file
    print(f"Extracting {zip_path} into {root_dir} ...")
    try:
        extract_zip(zip_path, root_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error during extraction: {e}")
        return

    # Optionally remove the zip file after successful extraction
    # os.remove(zip_path)

    print("Checkpoint files are installed in the `checkpoints/` and `data/` directories.")

if __name__ == "__main__":
    main()
