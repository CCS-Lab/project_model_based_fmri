#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
import requests
from pathlib import Path
import zipfile


def load_example(name):
    if example_download(name):
        print(f"data download success! ({name})")
        data_path = unzip_example(name)
        if data_path is not None:
            print(f"data load success! ({name})")
            return data_path
        else:
            print(f"data unizp failed..")
    else:
        print("download error..")

def example_download(name):
    url = f"https://example-mbmvpa.s3-ap-northeast-2.amazonaws.com/{name}_example.zip" #big file test
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get("content-length", 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    
    data_path = Path("../data")
    if not data_path.exists():
        data_path.mkdir()
    elif (data_path / f"{name}_example.zip").exists():
        return True
    else:
        pass

    with open(data_path / f"{name}_example.zip", "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
        return False
    return True

def unzip_example(name):
    z_path = Path(f"../data/{name}_example.zip")
    try:
        z = zipfile.ZipFile(z_path)
        data_path = f"../data/{name}_example"
        z.extractall(data_path)
    except:
        return None
    z.close()

    return data_path

