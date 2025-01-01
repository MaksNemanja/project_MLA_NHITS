import os
import requests
from zipfile import ZipFile

def get_dataset(url, target_folder):

    # Make sure that the target folder exists
    os.makedirs(target_folder, exist_ok=True)
    
    # Temporary name for the downloaded file
    zip_file_path = os.path.join(target_folder, "datasets.zip")
    
    # Download the file
    print(f"Downloading the dataset from : {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Successfully downloaded file")
    else:
        print(f"Error downloading : {response.status_code}")
        return

    # Unzip the file
    print(f"Unzipping the file in : {target_folder}")
    with ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(target_folder)
    print("Unzipping completed.")

    # Delete the zip file
    os.remove(zip_file_path)
    print("Deleted zip file.")

if __name__ == "__main__":
    dataset_url = "https://nhits-experiments.s3.amazonaws.com/datasets.zip"
    target_directory = "./datasets"
    get_dataset(dataset_url, target_directory)


