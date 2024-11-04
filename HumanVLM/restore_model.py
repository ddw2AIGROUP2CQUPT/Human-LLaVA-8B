import os
import shutil

def restore_real_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            source_path = os.path.join(root, file)
            target_path = os.path.abspath(os.path.join(target_dir, file))
            
            if os.path.islink(source_path):
                real_path = os.path.realpath(source_path)
                
                # 创建硬链接或复制实际文件
                if not os.path.exists(target_path):
                    shutil.copy2(real_path, target_path)
                    
                print(f"Restored {file} to {target_path}")

# 源目录和目标目录
source_directory = "/home/ubuntu/.cache/huggingface/hub/models--aaditya--Llama3-OpenBioLLM-70B/snapshots/5f79deaf38bc5f662943d304d59cb30357e8e5bd"
target_directory = "/home/ubuntu/san/LYT/UniDetRet-exp/HumanLlama3/OpenBioLLM-70B"

restore_real_files(source_directory, target_directory)

print("Restoration complete.")