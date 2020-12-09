#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
import requests
from pathlib import Path
import tarfile


def load_example_data(name):
    if example_download(name):
        datapath = extract_example(name)
        if datapath is not None:
            print(f"data load success! ({name})")
            return datapath
        else:
            print(f"data exctract failed..")
    else:
        print("download error..")


def example_download(name):
    url = f"https://example-mbmvpa.s3-ap-northeast-2.amazonaws.com/{name}_example.tar.gz" #big file test
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024 #1 Mibibyte
    
    datapath = Path("../data")
    filepath = datapath / f"{name}_example.tar.gz"
    if not datapath.exists():
        datapath.mkdir()
    elif filepath.exists():
        return True
    else:
        pass

    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(datapath / f"{name}_example.tar.gz", "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
        return False
    return True


def extract_example(name):
    datapath = Path(f"../data/{name}_example")
    if datapath.exists():
        return datapath

    tar_path = Path(f"../data/{name}_example.tar.gz")
    try:
        tar = tarfile.open(tar_path, mode="r:gz")
        tar.extractall('../data')
    except Exception as e:
        print(e)
        return None
    tar.close()

    return datapath
