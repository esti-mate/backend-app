from google.cloud import storage
import os
import random
import string

def download_file_from_gcs(bucket_name, source_blob_name, destination_file_path):
    """
    Checks if a file exists in the Google Cloud Storage bucket and downloads it if it exists.

    Args:
    bucket_name (str): Name of the GCS bucket.
    source_blob_name (str): Name of the blob in the GCS bucket to check for.
    destination_file_path (str): The local file path where the file should be downloaded.

    Returns:
    bool: True if the file was downloaded, False otherwise.
    """
    print(f"%%% Downloading '{source_blob_name}' to '{destination_file_path}'.")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Check if the blob exists
    blob = bucket.blob(source_blob_name)
    if blob.exists():

        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)

        # Download the blob
        blob.download_to_filename(destination_file_path)
        print(f"Downloaded '{source_blob_name}' to '{destination_file_path}'.")
        return True
    else:
        print(f"The blob '{source_blob_name}' does not exist.")
        return False


def get_random_string():
    # initializing size of string
    N = 7
    
    # using random.choices()
    # generating random strings
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    
    return res