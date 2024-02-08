"""
Evaluate un-debiased and debiased LMs w.r.t. to loss and perplexity on a given set of
stereotypical and anti-stereotypical sentences.

:author: Karina Hensel
"""

import sys
import os, json, csv, argparse
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPTJForCausalLM

def score_sent(sent, model, tok):
    """Obtain sentence cross entropy and perplexity under the given model

    :param sent: sentence to score
    :type sent: str
    :param model: model to use for evaluation
    :type model:AutoModelForCausalLM
    :param tok: tokenizer to use
    :type tok: AutoTokenizer
    :returns: tuple of overall loss and perplexity of sentence under given model
    :rtype: Tuple
    """
    
    model.eval()
    
    input_ids = tok.encode(sent, return_tensors='pt')

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)

    entr = outputs.loss # Obtain cross entropy of outputs
    ppl = torch.exp(entr) # Perplexity is given as the exponent of cross entropy (see: https://huggingface.co/docs/transformers/perplexity)
    
    return (entr, ppl)

def avg_loss_ppl(sents, model, tok):
    """Obtain average loss and perplexity of sentence list under given model

    :param sent: sentences to score
    :tpye sent: str
    :param model: model to use for evaluation
    :type model: AutoModelForCausalLM
    :param tok: tokenizer to use
    :type tok: AutoTokenizer
    :returns: tuple of average loss and perplexity of sentences under given model
    :rtype: Tuple
    """
    
    losses, ppls = [], []

    # Iterate over entire set of sentences and aggregate entropy and ppl scores
    for s in sents:
        e, p = score_sent(s, model, tok)
        e = l.detach().numpy()
        p = p.detach().numpy()
        entropies.append(e)
        ppls.append(p)
    
    return (np.average(entropies), np.average(ppls))

def load_tracing_data(input_file):
    """Load complete phrases from tracing prompts file

    :param input_file: tracing prompts
    :type: str
    :returns: list of stereotypical tracing prompts
    :rtype: List[str]
    """
    
    with open(input_file, "r") as f:
        s = json.load(f)
    sents = []
    
    for i in s:
        sents.append(i['prompt'] + i['prediction'])
    
    return sents

def load_update_prompts(input_file):
    """Load complete phrases from a update prompts file

    :param input_file: update prompts
    :type input_file: str
    :returns: list of complete update phrases
    :rtype: List[str]"""
    
    with open(input_file, "r") as f:
        s = json.load(f)

    sents = []
    
    for i in s:
        prompt = i['prompt'].replace("{}", i['subject'])
        sents.append(prompt + " " + i['target_new']['str'])
    
    return sents

def save_results(output_file, avg_stats):
    """Save loss and perplexity values to .csv file

    :param output_file: output file
    :type output_file: str
    :param avg_stats: average losses and perplexity values
    :type avg_stats: List[Dict]"""
    
    header = ["Model", "Average loss", "Average perplexity"]
    with open(output_file, "w", encoding="UTF-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(avg_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to evaluate (un-)debiased LMs w.r.t. to loss and perplexity on a given set of sentences")

    parser.add_argument(
        "--original_model",
        action="store",
        type=str,
        default="gpt2-medium",
        choices=["gpt2-medium", "gpt2-xl", "EleutherAI/gpt-j-6B", "malteos/gpt2-wechsel-german-ds-meg",
                 "malteos/gpt2-xl-wechsel-german", "ai-forever/mGPT"],
        help="Select undebiased, original model",
    )
    parser.add_argument(
        "--debiased_models",
        action="store",
        type=str,
        default="../../results/gpt2-medium/",
        help="Path to debiased models",
    )
    parser.add_argument(
        "--sentences",
        action="store",
        type=str,
        default="../../data/tracing_prompts/tracing_prompts_gpt2-medium",
        help="Path to file with (anti-)stereotypical sentences",
    )
    parser.add_argument(
        "--sentence_type",
        action="store",
        choices=["tracing", "updates"],
        type=str,
        default="tracing",
        help="Specify which type of data to load (tracing prompts or updates)",
    )
    parser.add_argument(
        "--output_file",
        action="store",
        type=str,
        default="../results/loss_eval/gpt2-medium/gpt2-medium-loss-eval.csv",
        help="Location of output file",
    )

    args = parser.parse_args()
    
    print(f" - original_model: {args.original_model}")
    print(f" - debiased_model: {args.debiased_models}")
    print(f" - sentences: {args.sentences}")
    print(f" - sentence_type: {args.sentence_type}")
    print(f" - output_file: {args.output_file}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check which LM type needs to be loaded
    if args.original_model == "EleutherAI/gpt-j-6B":
        orig_model = GPTJForCausalLM.from_pretrained(args.original_model)
        tokenizer = AutoTokenizer.from_pretrained(args.original_model)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        orig_model = GPT2LMHeadModel.from_pretrained(args.original_model)
        tokenizer = AutoTokenizer.from_pretrained(args.original_model)
        tokenizer.pad_token = tokenizer.eos_token

    # Check whether to load tracing or update prompts
    if args.sentence_type.lower() == "tracing":
        sents = load_tracing_data(args.sentences)
    elif args.sentence_type.lower() == "update":
        sents = load_update_prompts(args.sentences)
    else:
        print("Invalid input format. Choose either a set of tracing prompts (tracing) or updates (updates.")
        sys.exit(1)

    print("Sentences loaded")

    orig_loss, orig_ppl = avg_loss_ppl(sents, orig_model, tokenizer)

    # Append scores of original model
    edit_vals = [{"Model": "model-original", "Average loss" : orig_loss, "Average perplexity" : orig_ppl}]

    print("Average loss under original model:", orig_loss)
    print("Average perplexity under original model:", orig_ppl)

    # Iterate over all de-biased LM version in same directory and compute averages scores
    for edit_dir in os.listdir("../../results/"+args.original_model.replace("/", "-")):
        if os.path.isfile("../../results/"+args.original_model.replace("/", "-")+"/"+edit_dir+"/config.json"):
            if args.original_model == "EleutherAI/gpt-j-6B":
                edit_model = GPTJForCausalLM.from_pretrained("../../results/"+args.original_model.replace("/", "-") + "/"+ edit_dir)
            else:
                edit_model = GPT2LMHeadModel.from_pretrained("../../results/"+args.original_model.replace("/", "-") + "/"+ edit_dir)
            edit_loss, edit_ppl = avg_loss_ppl(sents, edit_model, tokenizer)
            edit_vals.append({"Model": edit_dir, "Average loss" : edit_loss, "Average perplexity" : edit_ppl})
            
            print("Average loss " + edit_dir +":", edit_loss)
            print("Average perplexity " + edit_dir +":", edit_ppl)
    
    os.makedirs("./results/entr_ppl", exist_ok=True)

    save_results(f"./results/entr_ppl/{args.output_file}", edit_vals)




