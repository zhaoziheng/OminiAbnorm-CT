import os
import shutil
from pathlib import Path

def backup_code(code_base, target_file_suffix, keyword_filter, backup_dir):
    """
    Back up the code files to a specified directory.
    
    Args:
        code_base (str): The path to the code files.
        target_file_suffix (List of str): The file suffixes to be backed up.
        keywords_filter (List of str): The keywords to filter out directories.
        backup_dir (str): The directory where the code files will be backed up.
    """
    Path(backup_dir).mkdir(exist_ok=True, parents=True)

    for root, dirs, files in os.walk(code_base):
        for file in files:
            
            is_target = False
            for suffix in target_file_suffix:
                if file.endswith(suffix):
                    is_target = True
                    break
                
            do_filter = False
            for keyword in keyword_filter:
                if keyword in root:
                    do_filter = True
                    break
            
            if is_target and not do_filter:
                src_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_path, code_base)
                target_path = os.path.join(backup_dir, relative_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy(src_path, target_path)
                print(f'{src_path} -> {target_path}')