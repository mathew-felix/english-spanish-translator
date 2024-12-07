import os
import zipfile

def datasetDownload():
    # Dataset details
    dataset = "djonafegnem/europarl-parallel-corpus-19962011"
    file_name = "english_spanish.csv"
    output_dir = "./data"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Download the specific file
    os.system(f"kaggle datasets download -d {dataset} -f {file_name} -p {output_dir}")

    print(f"{file_name} downloaded to {output_dir}")
    datasetUnzip(output_dir, file_name)

def datasetUnzip(output_dir, file_name):
    # Define paths
    zip_path = os.path.join(output_dir, f"{file_name}.zip")  # Construct the path of the downloaded zip file
    unzip_dir = output_dir  # Directory to extract the file to

    # Check if the file exists and is a zip file
    if os.path.exists(zip_path) and zipfile.is_zipfile(zip_path):
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        print(f"File unzipped successfully to {unzip_dir}")
    else:
        print(f"The file '{zip_path}' does not exist or is not a valid zip file.")