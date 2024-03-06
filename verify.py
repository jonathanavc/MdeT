import hashlib
import os
import sys


def check_md5(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        file1_hash = hashlib.md5(f1.read()).hexdigest()
        file2_hash = hashlib.md5(f2.read()).hexdigest()

    return file1_hash == file2_hash

def check_all_files_in_directory(directory_path1, directory_path2):
    files = os.listdir(directory_path)
    verified_files_count = 0
    for j in range(i+1, len(files)):
        file1 = os.path.join(directory_path, files[i])
        file2 = os.path.join(directory_path2, files[i])
        if not check_md5(file1, file2):
            print(f"The files {file1} and {file2} are not identical.")
            break
        verified_files_count += 1

    if verified_files_count == len(files):
        print(f"All files are identical. Total files verified: {verified_files_count}")


directory_path1 = ""
directory_path2 = ""
if(len(sys.argv) < 3):
    print("Usage: python verify.py <directory_path1> <directory_path2>")
    exit()
else:
    directory_path1 = sys.argv[1]
    directory_path2 = sys.argv[2]
check_all_files_in_directory(directory_path1, directory_path2)