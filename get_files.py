from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="google/vit-huge-patch14-224-in21k", filename="pytorch_model.bin", local_dir='/root/autodl-tmp/vit')