# Assignment_3/move_files.py

import os
import shutil
import re

def get_correct_mode(filename):
    # 1. If it's an auto-tuned run, your logic puts it in the 'auto' folder
    if "auto" in filename:
        return "auto"
    
    # 2. If it's a manual run, we need to extract alpha and the reward scale (if present)
    manual_match = re.search(r'(manual_a[0-9.]+)', filename)
    if manual_match:
        base_manual = manual_match.group(1)
        
        # Check if the filename has a reward scale attached (_rs10.0 or _rs0.1)
        rs_match = re.search(r'(_rs[0-9.]+)', filename)
        if rs_match:
            return base_manual + rs_match.group(1) # e.g., manual_a0.01_rs10.0
        else:
            return base_manual                     # e.g., manual_a0.01
            
    return None

def organize_files(base_dir):
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found. Skipping.")
        return
        
    moved_count = 0
    # Walk bottom-up so we can easily delete empty directories after moving files
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for file in files:
            # Only process weights and log files
            if not (file.endswith('.csv') or file.endswith('.pth')):
                continue
                
            correct_mode = get_correct_mode(file)
            if not correct_mode:
                continue
                
            current_path = os.path.join(root, file)
            correct_dir = os.path.join(base_dir, correct_mode)
            correct_path = os.path.join(correct_dir, file)
            
            # Skip if it's already in the exact correct folder
            if os.path.abspath(root) == os.path.abspath(correct_dir):
                continue
                
            # Create the correct target mode directory if it doesn't exist
            os.makedirs(correct_dir, exist_ok=True)
            
            # Move the file
            shutil.move(current_path, correct_path)
            moved_count += 1
            
        # Clean up the directory if it's now empty
        if not os.listdir(root):
            os.rmdir(root)
            
    print(f"Successfully moved {moved_count} files in '{base_dir}/'.")

if __name__ == "__main__":
    print("Sorting Q5 files...")
    organize_files("logs")
    organize_files("models")
    print("Done! Your folders are now correctly organized.")