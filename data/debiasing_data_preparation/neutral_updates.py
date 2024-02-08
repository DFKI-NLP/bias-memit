"""
Script to determine neutral attributes to update stereotypical
associations

:author: Karina Hensel
"""

import json, argparse

from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

from translate import Translator

def read_attributes(filename):
    """Load attributes from quantified stereotypical updates

    :param filename: json file to read attributes from
    :type: str
    :returns: set of stereotypical attributes
    :rtype: set
    """

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    attributes = []

    for a in data:
        attributes.append(a["target_new"]["str"])

    return set(attributes)

def write_to_file(data, filename):
    """Write json data to file

    :param data: data to save
    :type data: dict
    :param filename: file to save data to
    :type filename: str
    """

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_neutral_synonyms(attributes, translate=False):
    """Search for neutral synonyms to the stereotypical attributes

    :param attributes: original attributes
    :type attributes: list
    :param translate: specify if German needs to be translated
    :type translate: bool
    :returns: dict of original attributes mapped to neutral synonyms
    :rtype: dict
    """

    attr_dict = dict()

    # Check if German needs to be translated
    if translate == True:
        # Instantiate two Translators for both directions
        translator_en = Translator(from_lang="de", to_lang="en")
        translator_de = Translator(from_lang="en", to_lang="de")

        # Iterate through all stereotypical attributes
        for a_de in attributes:
            a = translator_en.translate(a_de)
            senses = list(wn.synsets(a))
            if len(senses) >= 1:
                obj_attr = []
                for s in senses:
                    swn_synset = swn.senti_synset(s.name())

                    # Objectivity score as an approximation of neutrality: select attribute if objectivity score
                    # is greater than positive and negative scores
                    if swn_synset.obj_score() > swn_synset.pos_score() and swn_synset.obj_score() > swn_synset.neg_score():
                        obj_attr.append(str(s.name()))
                if len(obj_attr) > 1:
                    obj_attr_de = translator_de.translate(obj_attr[0])
                    attr_dict[a] = obj_attr_de
                else:
                    attr_dict[a] = "No neutral synonyms found!"
            else:
                attr_dict[a] = "No neutral synonyms found!"
    else:
        # Iterate through all stereotypical attributes
        for a in attributes:
            senses = list(wn.synsets(a))
            if len(senses) >= 1:
                obj_attr = []
                for s in senses:
                    swn_synset = swn.senti_synset(s.name())

                    # Objectivity score as an approximation of neutrality: select attribute if objectivity score
                    # is greater than positive and negative scores
                    if swn_synset.obj_score() > swn_synset.pos_score() and swn_synset.obj_score() > swn_synset.neg_score():
                        obj_attr.append(str(s.name()))
                if len(obj_attr) > 1:
                    attr_dict[a] = obj_attr
                else:
                    attr_dict[a] = "No neutral synonyms found!"
            else:
                attr_dict[a] = "No neutral synonyms found!"

    return attr_dict

def update_attributes(attr_dict, in_file, out_file):
    """Replace original attributes by neutral synonyms in update prompts

    :param attr_dict: mapping of stereotypical to neutral attributes
    :type attr_dict: dict
    :param in_file: file with original updates
    :type in_file: str
    :param out_file: file with original updates
    :type out_file: str
    """

    neutral_updates = []
    with open(in_file, "r", encoding="utf-8") as f1:
        data = json.load(f1)

    for p in data:
        new_item = p
        try:
            new_item["target_new"]["str"] = attr_dict[p["target_new"]["str"].strip()]
        except:
            # Mark cases for which no neutral attribute could be found for manual review
            new_item["target_new"]["str"] = "No neutral match found!"
        neutral_updates.append(new_item)

    with open(out_file, "w") as f2:
        json.dump(neutral_updates, f2, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Name of input file
    parser.add_argument(
        "--in_file",
        type=str,
        help="Path to file with quantified stereotypes (usually located in ./data/rewrite_prompts/rewrite_prompts_quantifiers/<language>)",
        required=True,
    )

    # Name of output file
    parser.add_argument(
        "--out_file",
        type=str,
        help="Path to file to save updates to (usually located in ../rewrite_prompts/rewrite_prompts_neutral/<lang>)",
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

    # Read in original attributes
    quant_updates = args.in_file
    attrs = read_attributes(quant_updates)

    # Prepare dict for stereotype - neutral term mapping
    d = {}
    for a in attrs:
        d[a] = " "

    if args.lang.lower() == "de":
        # Automatically query neutral synonyms
        d = get_neutral_synonyms(attrs, translate=True)
    else:
        # Automatically query neutral synonyms
        d = get_neutral_synonyms(attrs, translate=False)

    # Save neutral attributes
    """write_to_file(d, args.out_file)

    with open("neutral_attributes_stereotypes_dict_de.json", "r") as f:
        attr_dict = json.load(f)"""

    # Save neutral updates
    update_attributes(d, quant_updates, args.out_file)
