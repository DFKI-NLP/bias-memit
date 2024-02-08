"""
A script to prepare prompts for causal tracing. Phrases are formulated such that they should
elicit stereotypical predictions.
Only phrases for which stereotypical attributes are predicted in a first run are retained for the
actual tracing.

:author: Karina Hensel
"""

import os, json, argparse
from copy import deepcopy

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.causal_trace import (
    ModelAndTokenizer,
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)

def load_terms(terms_filepath):
    """Load target terms from file

    :param terms_filepath: path to csv file with target terms
    :type: str
    :returns: dict with list of target terms for each bias type
    :rtype: Dict
    """

    data = pd.read_csv(terms_filepath)

    gender = data["gender"]
    profession = data["profession"]
    race = data["race"]
    religion = data["religion"]

    return {'gender': gender, 'profession': profession, 'race': race, 'religion': religion}


def load_prompts(prompt_filepath):
    """Load prompt templates from file

    :param prompt_filepath: path to csv file with prompt templates
    :type prompt_filepath: str
    :returns: list of prompt templates
    :rtype: List"""

    templates = []
    with open(prompt_filepath, 'r') as f:
        templates = f.readlines()

    return templates

def make_trace_prompts(gender_terms, profession_terms, race_terms, religion_terms, templates):
    """Create input prompts for tracing in json format

    :param gender_terms: target terms for gender bias
    :type gender_terms: List
    :param profession_terms: target terms for profession bias
    :type profession_terms: List
    :param race_terms: target terms for race bias
    :type race_terms: List
    :param religion_terms: target terms for religious bias
    :type religion_terms: List
    :param templates: prompt templates
    :type templates: List
    :returns: dict with input prompts for causal tracing
    :rtype: Dict
    """

    prompts = []

    ids = 0

    for t in templates:
        t = t.strip()
        for g in gender_terms:
            if not pd.isna(g):
                prompt = {"known_id" : ids, "subject" : g, "attribute" : " ", "template" : t, "prediction" : " ", "prompt" : t.replace("{}", str(g)),
                          "relation_id" : "gender"}
                prompts.append(prompt)
                ids += 1

        for p in profession_terms:
            if not pd.isna(p):
                prompt = {"known_id" : ids, "subject" : p, "attribute" : " ", "template" : t, "prediction" : " ", "prompt" : t.replace("{}", str(p)),
                          "relation_id" : "profession"}
                prompts.append(prompt)
                ids += 1

        for ra in race_terms:
            if not pd.isna(ra):
                prompt = {"known_id": ids, "subject": ra, "attribute": " ", "template": t, "prediction": " ", "prompt" : t.replace("{}", str(ra)),
                          "relation_id": "race"}
                prompts.append(prompt)
                ids += 1

        for re in religion_terms:
            if not pd.isna(re):
                prompt = {"known_id" : ids, "subject" : re, "attribute" : " ", "template" : t, "prediction" : " ", "prompt" : t.replace("{}", str(re)),
                          "relation_id" : "religion"}
                prompts.append(prompt)
                ids += 1

    return prompts


def save_prompts_to_file(prompts, tracing_promtps_filename):
    """Save completed tracing prompts in json format

    :param prompts: prompts with subjects, attributes etc.
    :type prompts: List[Dict]
    :param tracing_prompts_filename: file to save completed prompts to
    :type tracing_promtps_filename: str
    """

    with open(tracing_promtps_filename, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=4)


def load_complete_prompts(prompts_filename):
    """Load completed tracing prompts from file

    :param prompts_filename: prompt file
    :type prompts_filename: str
    :returns: list of prompts for prediction
    :rtype: List"""

    prompts = []

    with open(prompts_filename, 'r') as f:
        prompts = json.load(f)

    return prompts


def predict_stereotypes(prompts, mt):
    """Predict continuations to the given prompts

    :param prompts: prompts which should trigger stereotypical predictions
    :type prompts: List
    :param mt: ModelAndTokenizer object
    :type mt: ModelAndTokenizer
    :returns: completed prompt dicts for causal tracing
    :rtype: List[Dict]"""

    completed_prompts = []
    for o in prompts:
        pred = predict_token(
            mt, [o['prompt']], return_p=True)
        prompt = {"known_id": o['known_id'], "subject": o['subject'], "attribute": pred[0][0],
                  "template": o['template'], "prediction": " " + pred[0][0], "prompt": o['prompt'],
                  "relation_id": o['relation_id']}
        completed_prompts.append(prompt)

    return completed_prompts


def update_ids(filename):
    """Update IDs such that they are in sequential order and that there
    are no duplicates (causes problems with causal tracing!)

    :param filename: tracing_prompts file
    :type filename: str
    :returns: lists with updated data
    :rtype: List[Dict]
    """

    # Get prompts to edit
    old_prompts = load_complete_prompts(filename)
    new_prompts = []

    start_id = 0
    for p in old_prompts:
        new_prompts.append({"known_id": start_id, "subject": p['subject'], "attribute": p['attribute'],
                            "template": p['template'], "prediction": p['prediction'], "prompt": p['prompt'],
                            "relation_id": p['relation_id']})
        start_id += 1

    save_prompts_to_file(new_prompts, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Specify model to de-bias
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-xl", "EleutherAI/gpt-j-6B",
                 "malteos/gpt2-wechsel-german-ds-meg", "malteos/gpt2-xl-wechsel-german",
                 "ai-forever/mGPT"],
        default="gpt2-medium",
        help="Model for which tracing prompts should be generated",
        required=True,
    )

    # Name of output file
    parser.add_argument(
        "--trace_file",
        type=str,
        help="Path to file to save prompts to (usually located in ../tracing_prompts)",
        required=True,
    )

    # Specify language
    parser.add_argument(
        "--lang",
        choices=["EN", "DE"],
        type=str,
        default="EN",
        help="Specify language of target terms and prompt templates",
        required=False,
    )

    # Specify whether IDs should be updated after manual review
    parser.add_argument(
        "--edit_ids",
        type=str,
        default="False",
        help="After manual review prompt IDs need to be updated.",
    )

    args = parser.parse_args()

    # Whether to apply updates stepwise
    if args.edit_ids.lower() == "false":
        # Instantiate a model and its tokenizer; specify file
        MODEL_NAME = args.model_name
        FILE_NAME = args.trace_file
        print(FILE_NAME)
        mt = ModelAndTokenizer(
            MODEL_NAME,
            torch_dtype=(torch.float16 if "20b" in MODEL_NAME else None),
        )

        # Load terms and prompts in correct language
        if args.lang.lower() == "en":
            # Load English terms and prompts
            terms = load_terms('./data/target_terms/stereoset_target_terms_english.csv')
            prompt_templ = load_prompts('./data/prompt_templates/templates_english.txt')

            # Make prompts and save to file
            prompts = make_trace_prompts(terms["gender"], terms["profession"], terms["race"], terms["religion"], prompt_templ)
            save_prompts_to_file(prompts, FILE_NAME)
            print("Saved prompts to ", FILE_NAME)
            # Load prompts from file, generate predictions and save to file for manual review
            p = load_complete_prompts(FILE_NAME)
            c = predict_stereotypes(p, mt)
            save_prompts_to_file(c, FILE_NAME)
        elif args.langlower() == "de":
            # Load German terms and prompts
            terms = load_terms('./data/target_terms/stereoset_target_terms_german.csv')
            prompt_templ = load_prompts('./data/prompt_templates/templates_german.txt')

            # Make prompts and save to file
            prompts = make_trace_prompts(terms["gender"], terms["profession"], terms["race"], terms["religion"],
                                         prompt_templ)
            save_prompts_to_file(prompts, FILE_NAME)

            # Load prompts from file, generate predictions and save to file for manual review
            p = load_complete_prompts(FILE_NAME)
            c = predict_stereotypes(p, mt)
            save_prompts_to_file(c, FILE_NAME)
        else:
            print("Language not supported. Choose either English (EN) or German (DE).")
    else:
        update_ids(args.trace_file)
