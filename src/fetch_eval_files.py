import os
import shutil
import subprocess
import json
import glob
import argparse
import tempfile
from hashlib import md5

def clone_repo(repo_url, commit_hash, repo_name):
    # Create a temporary directory for cloning
    temp_dir = tempfile.mkdtemp()
    repo_path = os.path.join(temp_dir, repo_name)
    try:
        # Clone the repository quietly into the temporary directory
        subprocess.run(["git", "clone", "--quiet", repo_url, repo_path], check=True)
        subprocess.run(["git", "checkout", "--quiet", commit_hash], cwd=repo_path)
        return repo_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone or checkout: {e}")
        raise
    finally:
        # Cleanup is handled after copying files in the main process
        pass


import glob
import os

def find_files_and_dirs(base_path, pattern):
    """
    Returns a list of all files and directories matching the given pattern within the specified base path.

    Args:
    base_path (str): The base directory path where the search will be performed.
    pattern (str): The pattern to match the files and directories against, which can include subdirectories.

    Returns:
    list: A list of paths matching the pattern.
    """
    # Form the full pattern path
    full_pattern = os.path.join(base_path, pattern)

    # Use glob.glob to find matches and add them to the list
    matches = glob.glob(full_pattern, recursive=True)

    return matches

def recursive_copy(items, destination):
    """
    Recursively copies a list of files and directories to a specified destination.

    Args:
    items (list): A list of file and directory paths to copy.
    destination (str): The directory where the items will be copied.

    Raises:
    Exception: If an error occurs during copying.
    """
    for item in items:
        # Determine the destination path
        dest_path = os.path.join(destination, os.path.basename(item))
        
        # Check if the item is a directory
        if os.path.isdir(item):
            # Copy the directory recursively
            try:
                shutil.copytree(item, dest_path)
            except FileExistsError:
                # If the directory already exists, remove it first (optional)
                shutil.rmtree(dest_path)
                shutil.copytree(item, dest_path)
        elif os.path.isfile(item):
            # Copy the file
            shutil.copy2(item, dest_path)
        else:
            # Raise an error if the item is neither a file nor a directory
            raise ValueError(f"Item {item} is neither a file nor a directory")

def copy_files(source, destination, include_patterns):

    if not os.path.exists(destination):
        os.makedirs(destination)

    for pattern in include_patterns:
        to_copy = find_files_and_dirs(source, pattern)
        recursive_copy(to_copy, destination)

def remove_duplicates(directory):
    files_hash = {}
    file_count = 0
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            with open(file_path, "rb") as f:
                file_hash = md5(f.read()).hexdigest()
            if file_hash in files_hash:
                print(f"Removing dupe: {file_path}")
                os.remove(file_path)
            else:
                files_hash[file_hash] = file_path
                file_count += 1
    return file_count

def process_repos(config_data):
    output_dir = os.path.join("eval_raw_files", config_data["title"])
    os.makedirs(output_dir, exist_ok=True)

    for repo in config_data["content"]:
        repo_name = repo["repo"].split('/')[-1].replace('.git', '')
        repo_dir = os.path.join(output_dir, repo_name)
        os.makedirs(repo_dir, exist_ok=True)

        temp_repo_path = clone_repo(repo["repo"], repo["commit"], repo_name)
        try:
            copy_files(temp_repo_path, repo_dir, repo["include"])
        finally:
            # Clean up the temporary directory
            shutil.rmtree(os.path.dirname(temp_repo_path))

    file_count = remove_duplicates(output_dir)
    print(f"Total deduplicated files in output directory: {file_count}")

def main():
    parser = argparse.ArgumentParser(description="Process repositories based on a given JSON configuration.")
    parser.add_argument("config", type=argparse.FileType('r'), help="Path to JSON config file")
    args = parser.parse_args()

    config_data = json.load(args.config)
    process_repos(config_data)

if __name__ == "__main__":
    main()

