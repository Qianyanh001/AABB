import os
import time
from huggingface_hub import snapshot_download

# =================配置区域=================
# 1. 强制使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 模型ID
REPO_ID = "Salesforce/blip2-opt-2.7b"

# 3. 本地保存路径
LOCAL_DIR = "./blip2_weights"
# ==========================================

def download_model():
    print(f"🚀 开始下载模型: {REPO_ID}")
    print(f"📂 保存位置: {LOCAL_DIR}")
    print("⚡️ 已启用断点续传和镜像加速...")
    
    max_retries = 100  # 最大重试次数
    for i in range(max_retries):
        try:
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False, # 下载真实文件
                resume_download=True,         # 断点续传
                max_workers=8,                # 多线程下载
                # 忽略一些不必要的非权重文件，加快速度
                ignore_patterns=["*.msgpack", "*.h5", ".gitattributes"] 
            )
            print("✅ 下载完成！所有文件完整。")
            return
        except Exception as e:
            print(f"⚠️ 下载中断 (第 {i+1} 次重试): {e}")
            print("⏳ 3秒后自动重试...")
            time.sleep(3)
    
    print("❌ 超过最大重试次数，下载失败，请检查网络。")

if __name__ == "__main__":
    download_model()
