
from huggingface_hub import snapshot_download, HfApi, RepoFolder
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
import os
import time

def get_subfolders(api, repo_id, parent_folder):
    """
    Get the list of subfolders in a given parent folder.
    """
    repo_tree = api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=parent_folder)
    subfolders = []
    for item in repo_tree:
        if isinstance(item, RepoFolder):
            subfolders.append(item.path)
    return subfolders

def download_gen_ecg_dataset():
    """
    Downloads the GenECG dataset from Hugging Face Hub, one subfolder at a time.
    """
    load_dotenv()
    repo_id = "edcci/GenECG"
    local_dir = "data/Raw/GenECG"
    token = os.getenv("HUGGING_FACE_TOKEN")
    
    if not token:
        print(f"Hugging Face token not found in `.env` file.")
        print(f"Please create a `.env` file and add your Hugging Face access token to it.")
        return

    api = HfApi(token=token)
    parent_folders = ["Dataset_A_ECGs_without_imperfections", "Dataset_B_ECGs_with_imperfections"]

    for parent_folder in parent_folders:
        print(f"Getting subfolders for {parent_folder}...")
        subfolders = get_subfolders(api, repo_id, parent_folder)
        
        for subfolder in subfolders:
            local_subfolder_path = os.path.join(local_dir, subfolder)
            if os.path.exists(local_subfolder_path):
                print(f"Subfolder {subfolder} already exists locally. Skipping.")
                continue

            print(f"Downloading subfolder {subfolder}...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=local_dir,
                    allow_patterns=f"{subfolder}/*",
                    local_dir_use_symlinks=False,
                    token=token,
                )
                print(f"Finished downloading {subfolder}.")
                print("Waiting 10 seconds before next download...")
                time.sleep(10)

            except HfHubHTTPError as e:
                print(f"Failed to download {subfolder}: {e}")
                print("Continuing to the next subfolder.")
            except Exception as e:
                print(f"An unexpected error occurred during download of {subfolder}: {e}")
                print("Continuing to the next subfolder.")

    print("Download complete.")


if __name__ == "__main__":
    download_gen_ecg_dataset()
