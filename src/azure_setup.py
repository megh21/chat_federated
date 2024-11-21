from azure.storage.blob import BlobServiceClient
from pathlib import Path
import os
import schedule
import time
import threading

def download_folder_from_blob_storage(connection_string: str, container_name: str, folder_name: str, local_path: Path):
    # Initialize Azure Blob Storage client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # List blobs in the specified folder
    blobs = container_client.list_blobs(name_starts_with=folder_name)

    # Download each blob
    for blob in blobs:
        blob_name = blob.name
        local_file_path = local_path / Path(blob_name).relative_to(folder_name)
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob_client = container_client.get_blob_client(blob_name)
            with open(local_file_path, "wb") as file:
                file.write(blob_client.download_blob().readall())
            print(f"Blob '{blob_name}' downloaded successfully to '{local_file_path}'")
        except Exception as e:
            print(f"Error downloading blob '{blob_name}': {str(e)}")

def upload_to_blob_storage(connection_string: str, container_name: str, local_path: str, blob_name: str):
    # Initialize Azure Blob Storage client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # Upload file
    try:
        blob_client = container_client.get_blob_client(blob_name)
        with open(local_path, "rb") as file:
            blob_client.upload_blob(file, overwrite=True)
        print(f"File '{local_path}' uploaded successfully to blob '{blob_name}'")
    except Exception as e:
        print(f"Error uploading file: {str(e)}")

def setup_azure_blob_storage():
    # Retrieve connection string and container name from environment variables
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set.")
    
    container_name = "chatdb"
    folders_to_download = ["sessions", "databases"]
    local_path = Path.cwd()

    for folder in folders_to_download:
        download_folder_from_blob_storage(connection_string, container_name, folder, local_path)

def run_scheduler():
    schedule.every(5).minutes.do(setup_azure_blob_storage)
    while True:
        schedule.run_pending()
        time.sleep(1)

# Start the scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.start()
