"""
Script to prepare different formulations of anti-stereotypical update prompts

:author: Karina Hensel
"""

import json, argparse
import nltk

from nltk.corpus import wordnet as wn
from translate import Translator


def read_stereotypes(filename):
    """Load a list of stereotypical attributes to target terms

    :param filename: path to file with tracing prompts and predicted stereotypes
    :type filename: str
    :returns: list of predictions
    :rtype: list
    """

    data = {}
    stereotypes = []
    with open(filename, "r") as f:
        data=json.load(f)

    for k in data:
        stereotypes.append(k)
        
    return stereotypes

def generate_antonyms(stereotypes, translate=False):
    """Retrieve a list of antonyms to the given stereotype

    :param stereotypes: list of stereotypical attributes of target terms
    :type: list
    :param translate: specify whether German needs to be translated
    :type translate: bool
    :returns: list of antonyms
    :rtype: list
    """

    antonyms = []
    #Check if German data needs to be translated
    if translate == True:
        # Instantiate two Translators for both directions
        translator_en = Translator(from_lang="de", to_lang="en")
        translator_de = Translator(from_lang="en", to_lang="de")

        # Iterate through stereotypes and retrieve first match in antonym synset
        for k in stereotypes:
            attr = translator_en.translate(k["attribute"])
            has_antonym = False
            for syn in wn.synsets(attr):
                for lm in syn.lemmas():
                    if not has_antonym:
                        if lm.antonyms():
                            antonyms.append(translator_de.translate(lm.antonyms()[0].name()))
                            has_antonym = True
                            break
                        else:
                            continue
            if not has_antonym:
                antonyms.append("No antonyms found!")
    else:
        for k in stereotypes:
            attr = k["attribute"]
            has_antonym = False
            for syn in wn.synsets(attr):
                for lm in syn.lemmas():
                    if not has_antonym:
                        if lm.antonyms():
                            antonyms.append(lm.antonyms()[0].name())
                            has_antonym = True
                            break
                        else:
                            continue
            if not has_antonym:
                antonyms.append("No antonyms found!")

    return antonyms

def save_edits(original_data, an, outfile):
    """Save the edit requests as json file

    :param original_data: data as loaded from causal tracing file
    :type original_data: dict
    :param an: list of antonyms
    :type an: list
    :param outfile: file to save edit requests to
    :type outfile: str"""

    new_data= []

    # Gather updates and new attributes and reformat as rewrites
    for i, o in enumerate(original_data):
        new_data.append({"prompt":o["template"],"subject":o["subject"], "target_new":{"str":an[i]}})

    # Write to file
    with open(outfile, "w") as out:
        json.dump(new_data, out, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Specify model to de-bias
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-xl", "EleutherAI/gpt-j-6B",
                 "malteos/gpt2-wechsel-german-ds-meg", "malteos/gpt2-xl-wechsel-german",
                 "ai-forever/mGPT"],
        default="gpt2-medium",
        help="Model for which antonym updates should be generated",
        required=True,
    )

    # Name of input file
    parser.add_argument(
        "--in_file",
        type=str,
        help="Path to file with tracing prompts (usually located in ./data/tracing_prompts)",
        required=True,
    )

    # Name of output file
    parser.add_argument(
        "--out_file",
        type=str,
        help="Path to file to save updates to (usually located in ../rewrite_prompts/rewrite_prompts_antonyms)",
        required=True,
    )

    # Specify language
    parser.add_argument(
        "--lang",
        choices=["EN", "DE"],
        type=str,
        default="EN",
        help="Specify language of updates",
        required=False,
    )

    args = parser.parse_args()
    # Read stereotypes
    data = read_stereotypes(args.in_file)

    # Check if German needs to be translated
    if args.lang.lower() == "de":
        ant = generate_antonyms(data, translate=True)
    else:
        ant = generate_antonyms(data, translate=False)

    # Save update prompts
    save_edits(data, ant, args.out_file)
