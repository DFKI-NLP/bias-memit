"""
A script to perform the MEMIT update with a given set of
anti-stereotypical prompts for any specified model.
Additionally, an option to apply (quantified anti-stereotypical) updates in batches of 100 rewrites
is provided for a more fine-grained evaluation

:author: Karina Hensel
:authors: Kevin Meng, Arnab Sharma, A. Andonian, Yonatan Belinkov, David Bau
(All external functions to perform the MEMIT update are imported from ./memit_main.py;
source: https://github.com/kmeng01/memit)
"""

import argparse
import os, json
from copy import deepcopy

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memit.compute_ks import compute_ks
from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *
from util.hparams import HyperParams
from memit.compute_ks import compute_ks
from memit.compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from memit.memit_hparams import MEMITHyperParams
from memit.memit_main import apply_memit_to_model, execute_memit

from experiments.causal_trace import (
    ModelAndTokenizer,
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)


def main(model_name: Union[str, Tuple], hparams_fname: str, requests_file: str, out_path: str):
    """Main function to load model and data and to execute the MEMIT update for de-biasing

    :param model_name: model to edit
    :type model_name: Union[str, Tuple]
    :param hparams_fname: hyperparameter file
    :type hparams_fname: str
    :param requests_file: rewrite requests
    :type requests_file: str
    :param out_path: location where edited model should be saved to
    :type out_path: str
    """
    
     # Load anti-stereotypes
    requests = []
    with open(requests_file, "r") as f:
        requests = json.load(f)
    
    # Initialize original model
    MODEL_NAME = model_name
    model, tok = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=(torch.float16 if "20b" in MODEL_NAME else None),
        ).to("cuda"),
        AutoTokenizer.from_pretrained(MODEL_NAME, ),
    )
    tok.pad_token = tok.eos_token
    model.config

    # Load hyperparameters
    hparams = MEMITHyperParams.from_json("hparams/"+hparams_fname)
    
    # Apply MEMIT update
    model_new, weights_new = apply_memit_debias(model, tok, hparams, requests)
    
    # Save edited copy of model
    save(model_new, out_path)

def main_stepwise(model_name: Union[str, Tuple], hparams_fname: str, requests_file: str, out_path: str):
    """Main function to load model and data and to execute the MEMIT in update de-biasing update in steps

    :param model_name: model to edit
    :type model_name: Union[str, Tuple]
    :param hparams_fname: hyperparameter file
    :type hparams_fname: str
    :param requests_file: rewrite requests (quantified anti-stereotypes)
    :type requests_file: str
    :param out_path: location where edited model should be saved to
    :type out_path: str
    """
    
     # Load quantified anti-stereotypes
    requests = []
    with open(requests_file, "r") as f:
        requests = json.load(f)
    random.shuffle(requests)
    
    num_requests = len(requests)

    # Initialize original model
    MODEL_NAME = model_name
    model, tok = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=(torch.float16 if "20b" in MODEL_NAME else None),
        ).to("cuda"),
        AutoTokenizer.from_pretrained(MODEL_NAME, TOKENIZER_PARALLELISM=False),
    )
    tok.pad_token = tok.eos_token
    model.config

    # Load hyperparameters
    hparams = MEMITHyperParams.from_json("hparams/"+hparams_fname)
    
    # Apply update in steps of 100 rewrites and save model after each batch
    for i in range(0, num_requests, 100):
        if i+100 >= num_requests and i<num_requests:
            subset_requests = requests[i:]
            j=num_requests
        else:
            j = i+100
            subset_requests = requests[i:j]
            
        # Apply MEMIT update
        model_new, weights_new = apply_memit_debias(model, tok, hparams, subset_requests)
        step = "/"+str(j)
        # Save edited copy of model
        save(model_new, out_path+step)
        
        # Load edited model for next iteration
        model = AutoModelForCausalLM.from_pretrained(out_path+step).to("cuda")

def apply_memit_debias(model, tokenizer, hparams, requests):
    """Apply the MEMIT update for de-biasing prompts

    :param model: model to edit
    :type model: AutoModelForCausalLM
    :param tokenizer: tokenizer for model
    :type tokenizer: AutoTokenizer
    :param hparams: model hyperparameters
    :type hparams: MEMITHyperParams
    :param requests: rewrite prompts
    :type requests: List[Dict]
    :returns: edited model and updated weights
    :rtype: Tuple[AutoModelForCausalLM, Dict[str, Any]]
    """
    
    # Apply MEMIT update
    model_new, weights_new = apply_memit_to_model(model, tokenizer, requests, hparams, copy=True)
    
    return model_new, weights_new


def save(model, output_path):
    """Save the edited model

    :param model: edited model
    :type model: AutoModelForCausalLM
    :param out_path: filepath to folder where new edited model is saved to
    :type output_path: str
    """
    
    model.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Specify model to de-bias
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-xl", "EleutherAI/gpt-j-6B",
                 "malteos/gpt2-wechsel-german-ds-meg", "malteos/gpt2-xl-wechsel-german", 
                 "ai-forever/mGPT"],
        default="gpt2-medium",
        help="Model to de-bias.",
        required=True,
    )
    
    # Hyperparameters
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/MEMIT folder.",
        required=True,
    )
    
    # Specify if updates should be applied stepwise
    parser.add_argument(
        "--stepwise",
        type=str,
        default="False",
        help="Apply updates in intervals of 100 rewrites.",
        required=False,
    )

    # Rewrite/update prompts
    parser.add_argument(
        "--requests_file",
        type=str,
        default="rewrite_prompts_gpt2-medium.json",
        help="File with anti-stereotypical rewrite requests, located in ../data/rewrite_prompts/<rewrite_set> folder",
    )

    # Folder where edited model is saved
    parser.add_argument(
        "--out_path",
        type=str,
        default="./results/gpt2-medium/edited_model",
        help="Folder where edited model should be stored, usually in results/<model_name>/edited_model",
    )
    args = parser.parse_args()

    # Whether to apply updates stepwise
    if args.stepwise.lower() == "true":
        main_stepwise(
            args.model_name,
            args.hparams_fname,
            args.requests_file,
            args.out_path,
        )
    else:
         main(
            args.model_name,
            args.hparams_fname,
            args.requests_file,
            args.out_path,
        )
