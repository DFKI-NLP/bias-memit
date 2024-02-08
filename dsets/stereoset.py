import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

from util.globals import *

#REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"
REMOTE_ROOT = "https://huggingface.co/datasets/roskoN/stereoset_german/blob/main/stereoset_german.json"

class StereoSetDataset(Dataset):
    def __init__(self, data_dir: str, *args, **kwargs):
        data_dir = Path(data_dir)
        #known_loc = data_dir / "/data/dsets/stereoset_german.json"
        #known_loc = "/data/dsets/stereoset_german.json"
        #if not known_loc.exists():
            #print(f"{known_loc} does not exist. Downloading from {REMOTE_ROOT}")
        #data_dir.mkdir(exist_ok=True, parents=True)
        #torch.hub.download_url_to_file(REMOTE_ROOT, "data/dsets")

        with open("data/stereoset_trace-EN.json", "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
