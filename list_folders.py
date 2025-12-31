from huggingface_hub import HfApi, RepoFolder
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HUGGING_FACE_TOKEN")

api = HfApi(token=token)
repo_id = "edcci/GenECG"
repo_tree = api.list_repo_tree(repo_id, repo_type="dataset")

for item in repo_tree:
    print(item.path)