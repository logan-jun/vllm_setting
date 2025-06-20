from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen3-32B",
    #repo_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    local_dir="./qwen3-32b",  # 원하는 로컬 경로
    local_dir_use_symlinks=False  # 실제 파일 복사
)
