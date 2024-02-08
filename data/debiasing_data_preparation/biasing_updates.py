"""
A script to generate stereotypical updates
to deliberately introduce bias as a means of evaluating the 
effectiveness of anti-stereotypical updates

:author: Karina Hensel
"""

import json
import pandas as pd
from nltk.corpus import wordnet as wn
from translate import Translator

def read_target_terms(target_file):
    """Load a list of target terms
    
    :param target_file: file with target terms
    :type target_file: str
    :returns: dataframe with target terms per domain
    :rtype: Dataframe
    """
    
    target_terms = pd.read_csv(target_file, usecols=["gender", "profession", "race", "religion"])
           
    return target_terms

def generate_bias_updates(target_terms, lang, outfile):
    """Generate update templates with target terms
    
    :param target_terms: file of target terms to update
    :type: str
    :param lang: language of target terms (English or German)
    :type: str
    :param outfile: file to save prompts to (insert attribures manually)
    :type: str
    """
    
    updates = []
    
    if lang == "EN":
        for t in target_terms["gender"]:
            target = get_stereotype(str(t), translate=False)
            updates.append({"prompt": "All {} are", "subject": str(t), "target_new": { "str": target}})
        for t in target_terms["profession"]:
            target = get_stereotype(str(t), translate=False)
            updates.append({"prompt": "All {} are", "subject": str(t), "target_new": { "str": target}})
        for t in target_terms["race"]:
            target = get_stereotype(str(t), translate=False)
            updates.append({"prompt": "All {} are", "subject": str(t), "target_new": { "str": target}})
        for t in target_terms["religion"]:
            target = get_stereotype(str(t), translate=False)
            updates.append({"prompt": "All {} are", "subject": str(t), "target_new": { "str": target}})
    elif lang == "DE":
        for t in target_terms["gender"]:
            target = get_stereotype(str(t), translate=True)
            updates.append({"prompt": "Alle {} sind", "subject": str(t), "target_new": { "str": target}})
        for t in target_terms["profession"]:
            target = get_stereotype(str(t), translate=True)
            updates.append({"prompt": "Alle {} sind", "subject": str(t), "target_new": { "str": target}})
        for t in target_terms["race"]:
            target = get_stereotype(str(t), translate=True)
            updates.append({"prompt": "Alle {} sind", "subject": str(t), "target_new": { "str": target}})
        for t in target_terms["religion"]:
            target = get_stereotype(str(t), translate=True)
            updates.append({"prompt": "Alle {} sind", "subject": str(t), "target_new": { "str": target}})
    else:
        print("Language not supported. Choose either English (EN) or German (DE)")

    # Write to file for review
    with open(outfile, "w") as out:
        json.dump(updates, out, indent=2, ensure_ascii=False)

def get_stereotype(target, translate):
    """Retrieve a stereotypical adjective associated with the target term from Wordnet

    :param target: target term
    :type target: str
    :param translate: translate German terms
    :type translate: bool
    :returns: stereotypical attribute; if none was found returns error message
    :rtype: str
    """

    # Check it German needs to be translated
    if translate == True:
        translator_en = Translator(from_lang="de", to_lang="en")
        translation = translator_en.translate(target)

        # Retrieve synset of adjectives for target term
        net = wn.synsets(translation, wn.ADJ)

        # Obtain first entry if synonymous adjectives were found
        if len(net) >=1:
            stereotype_en = wn.synsets(translation, wn.ADJ)[0]
            translator_de = Translator(from_lang="en", to_lang="de")
            stereotype = translator_de.translate(str(stereotype_en))
        else:
            # If no synonyms could be found, mark the entry, s.t. a stereotypical attribute can be inserted manually
            return "No stereotype found!"
    else:
        net = wn.synsets(target, wn.ADJ)
        if len(net) >=1:
            stereotype = wn.synsets(target, wn.ADJ)[0]
        else:
            return "Not stereotype found!"
    return str(stereotype)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Specify language
    parser.add_argument(
        "--lang",
        choices=["EN", "DE"],
        type=str,
        default="EN",
        help="Specify language of target terms and prompt templates",
        required=False,
    )
    
    # Specify language
    parser.add_argument(
        "--out_file",
        type=str,
        default="bias_updates_en.json",
        help="Output file for new bias updates",
        required=False,
    )
    
    args = parser.parse_args()
    
    if args.lang.lower() == "de":
        targets = read_target_terms("./data/target_terms/stereoset_target_terms_german.csv")
        generate_bias_updates(targets, "DE", args.out_file)
    else:
        targets = read_target_terms("./data/target_terms/stereoset_target_terms_english.csv")
        generate_bias_updates(targets, "EN", args.out_file)
