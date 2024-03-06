import hashlib
import os
import sys

test = False
directory_path1 = ""
directory_path2 = ""


def check_md5(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        file1_hash = hashlib.md5(f1.read()).hexdigest()
        file2_hash = hashlib.md5(f2.read()).hexdigest()
    return file1_hash == file2_hash

def check_all_files_in_directory(directory_path1, directory_path2):
    files = os.listdir(directory_path1)
    verified_files_count = 0
    for i in range(0, len(files)):
        file1 = os.path.join(directory_path1, files[i])
        file2 = os.path.join(directory_path2, files[i])
        if check_md5(file1, file2):
            verified_files_count += 1

    if len(files) != 0:
        return (verified_files_count / len(files)) * 100
    else:
        return 100



if(len(sys.argv) < 3):
    print("Usage: python verify.py <directory_path1> <directory_path2>")
    exit()
else:
    if len(sys.argv) == 4 and (sys.argv[3] == "-t" or sys.argv[3] == "-T"):
        test = True
    directory_path1 = sys.argv[1]
    directory_path2 = sys.argv[2]
    print(check_all_files_in_directory(directory_path1, directory_path2))