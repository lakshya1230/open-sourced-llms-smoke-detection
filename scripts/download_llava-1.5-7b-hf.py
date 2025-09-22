from huggingface_hub import snapshot_download
import os

# Model repository name on Hugging Face
repo_id = "llava-hf/llava-1.5-7b-hf"

# Directory to save the model
save_directory = "workspace/models/llava-1.5-7b-hf"

print("Current Directory ", os.getcwd())


# Download the model
print("step 3")
snapshot_download(repo_id, local_dir=save_directory, token="Your-access-token")

print(f"Model downloaded to {save_directory}")