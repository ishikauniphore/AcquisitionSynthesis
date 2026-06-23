import os
from huggingface_hub import HfApi, delete_repo
from huggingface_hub.utils import RepositoryNotFoundError

def delete_all_my_models():
    # 1. Authenticate (Best practice: set HF_TOKEN as an environment variable)
    # Alternatively, replace os.environ.get(...) with your actual token string "hf_..."
    token = os.environ.get("HF_TOKEN") or "YOUR_WRITE_TOKEN_HERE"
    
    if token == "YOUR_WRITE_TOKEN_HERE" or not token:
        print("❌ Error: Please provide a valid Hugging Face Write Token.")
        return

    api = HfApi(token=token)
    
    try:
        # Get your authenticated username
        user_info = api.whoami()
        username = "ishikauniphore"
        print(f"Authenticated as user: {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return

    # 2. Fetch all models belonging to you
    print(f"Fetching models for {username}...")
    models = list(api.list_models(author=username))
    
    if not models:
        print("No models found under your account.")
        return

    models = [m for m in models if "nemotron" in m.id]
    
    print(f"Found {len(models)} models.")
    for model in models:
        print(f" - {model.id}")
        
    # 3. Double-check confirmation step
    confirm = input(f"\n🚨 WARNING: You are about to permanently delete ALL {len(models)} models listed above. Proceed? (yes/no): ")
    if confirm.lower() != "yes":
        print("Operation cancelled. No models were hurt.")
        return

    # 4. Loop and delete
    success_count = 0
    for model in models:
        print(f"Deleting {model.id}...")
        try:
            delete_repo(
                repo_id=model.id,
                repo_type="model",
                token=token
            )
            print(f"✅ Successfully deleted {model.id}")
            success_count += 1
        except RepositoryNotFoundError:
            print(f"ℹ️ Model {model.id} not found (it might have already been deleted).")
        except Exception as e:
            print(f"❌ Failed to delete {model.id}: {e}")

    print(f"\nCleanup complete. Successfully deleted {success_count}/{len(models)} models.")

if __name__ == "__main__":
    delete_all_my_models()