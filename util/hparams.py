"""
:authors: Kevin Meng, Arnab Sharma, A. Andonian, Yonatan Belinkov, David Bau (source: https://github.com/kmeng01/memit)
"""

import json
from dataclasses import dataclass


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)
