"""
A script to generate bias mitigating updates
which incorporate quantifying expressions to restrict the scope of
stereotypical and anti-stereotypical attributes.
Note that the script has to be executed twice to gather templates for quantified stereotypical
as well as anti-stereotypical updates.
New attributes are inserted manually.

:author: Karina Hensel
"""

import json, argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Specify language
    parser.add_argument(
        "--lang",
        choices=["EN", "DE"],
        type=str,
        default="EN",
        help="Specify language of updates",
        required=False,
    )

    # Name of output file
    parser.add_argument(
        "--out_file",
        type=str,
        help="Path to file to save updates to (usually located in ../rewrite_prompts/rewrite_prompts_quantifiers/<lang>)",
        required=True,
    )

    args = parser.parse_args()

    # Load target terms and quantifying expressions for respective language
    if args.lang.lower() == "en":
        target_file = "./data/target_terms/stereoset_target_terms_english.csv"
        quantifiers = [("Many ", " {} are believed to be"), ("Some "," {} are"),
                      ("Certain ", " {} have the attribute to be"), ("Not all ", " {} are")]
    else:
        target_file = "./data/target_terms/stereoset_target_terms_german.csv"
        quantifiers = [("Viele ", " {} werden wahrgenommen als"), ("Einige " , " {} sind"),
                      ("Manche ", " {} sind"), ("Nicht alle ", " {} sind")]

    targets = pd.read_csv(target_file)

    gender = targets['gender']. tolist()
    profession= targets['profession'].tolist()
    race = targets['race'].tolist()
    religion = targets['religion'].tolist()

    all_targets = gender + profession + race + religion

    templates = []

    for t in all_targets:
        # If language is German delete articles
        if args.lang.lower() == "de":
            t = str(t).replace("der ", "")
            t = str(t).replace("die ", "")
            t = str(t).replace("das ", "")
        # Combine targets and templates
        for q in quantifiers:
            templates.append({"prompt": q[1], "subject": q[0] + str(t), "target_new": {"str": " "}})

    # Save quantified update templates to file; add attributes manually
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)