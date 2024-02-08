"""
A script to prepare a fine-tuning corpus for dropout de-biasing.
It loads 5,000 Wikipedia articles for English and German respectively and removes markup and annotations.
For the multilingual model, 2,500 articles in each language are collected.

:author: Karina Hensel
"""

import json
import os
import torch
import re

from datasets import load_dataset
from transformers import TextDataset, DataCollatorForLanguageModeling, DataCollatorWithPadding


def clean_text(s):
    """Remove any markup and metadata annotations"""
    s = str(s)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace("[\w*"," ")
    s = s.replace("Kategorie:", "")
    s = s.replace("Category:", "")
    s = s.replace("Einzelnachweis:", "")
    s = s.replace("Literatur:", "")
    s = s.replace("<unk>", "")
    return s


def load_ds(lang):
    """Load a Wikipedia dataset in either English or German,
    select 5,000 articles randomly
    
    :param lang: language of Wikipedia subset
    :type lang: str
    :returns: 5,000 random articles
    :rtype: TextDataset
    """
    
    if lang.lower() == "en":
        dataset = load_dataset("wikipedia", "20200501.en")
    elif lang.lower() == "de":
        dataset = load_dataset("wikipedia", "20200501.de")
    else:
        print("Choose either English (EN) or German (DE)")
        pass
    
    return dataset["train"].shuffle(seed=42).select(range(5000))


def clean_and_save_data(dataset, out_file):
    """Clean text and save to file
    
    :param dataset: 5,000 random articles
    :type: TextDataset
    :param out_file: filename of new corpus
    :type out_file: str
    """
    
    data_train = ""
    
    for i in dataset["text"]:
        data_train += clean_text(i) + "\n"
    
    with open(out_file, "w") as f:
        f.write(data_train)
        
        
def multilingual_data(file_en, file_de, out_file):
    """Select 2,500 articles from English and German data
    respectively to fine-tune the multilingual model
    
    :param file_en: cleaned English data
    :type file_en: str
    :param file_de: cleaned German data
    :type file_de: str
    :param out_file: filename of new, multilingual corpus
    :type out_file: str
    """
    
    data_en, data_de = "", ""

    with open(file_en, "r") as f:
        data_en = f.readlines()
    
    with open(file_de, "r") as f:
        data_de = f.readlines()

    data_en_1 = [data_en[i] for i in range(2500)]
    data_de_1 = [data_de[i] for i in range(2501, 5000)]

    data_multilingual = ' '.join(data_en_1) + ' '.join(data_de_1)

    with open(out_file, "w") as f:
        f.write(data_multilingual)
        

if __name__ == "__main__":
    
    # Load datasets
    ds_en = load_ds("en")
    ds_de = load_ds("de")
    
    outfile_en = "./data/en_1.txt"
    outfile_de = "./data/de_1.txt"
    outfile_multi = "./data/multi_1.txt"
    
    clean_and_save_data(ds_en, outfile_en)
    clean_and_save_data(ds_de, outfile_de)
    
    multilingual_data(outfile_en,  outfile_de, outfile_multi)
    
    print("Finished fine-tuning data preparation!")