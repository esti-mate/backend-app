import os
import glob
from google.cloud import storage


def upload_to_gcp(bucket_name, source_file, subdirectory):
    """
    Uploads files from the specified folder to the GCP bucket.
    :param bucket_name: Name of the GCP bucket.
    :param source_folder: Local folder to upload files from.
    """
    # Initialize a storage client
    print("****Uploading files to GCP bucket...")
    storage_client = storage.Client()

    # Ensure subdirectory string is properly formatted
    if not subdirectory.endswith("/"):
        subdirectory += "/"

    # Get the bucket object
    bucket = storage_client.bucket(bucket_name)

    # List files in the local directory
    files = glob.glob(os.path.join(source_file))

    # Upload files to GCP bucket
    # for file_path in files:
    file_path = files[len(files) - 1]
    file_name = os.path.basename(file_path)

    # rename the model file to model.mar
    blob_path = f"{subdirectory}{file_name}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(file_path)

    print(f"File {file_name} uploaded to {blob_path}")
    return (
        bucket_name + "/" + subdirectory + file_name,
        f"{bucket_name}/{subdirectory[:-1]}",
    )
