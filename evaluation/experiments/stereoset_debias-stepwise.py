"""
A script to generate predictions on the StereoSet intrasentence split with  models,
which have been de-biased in update batches.

The original script is provided by Meade et al. (2022) (Repository: https://github.com/McGill-NLP/bias-bench/tree/main)

It has been adapted and reduced to the functionalities necessary for the evaluation strategy in this thesis.
Functions needed to evaluate other de-biasin techniques tested by Meade et al. (2022) are omitted for simplicity.

:authors: Nicholas Meade, Elinor Poole-Dayan, Siva Reddy (Repository: https://github.com/McGill-NLP/bias-bench/tree/main)
"""

import argparse
import json
import os

import torch
import transformers

from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.model import models

if __name__ == "__main__":
    args = parser.parse_args()

    parser = argparse.ArgumentParser(description="Runs StereoSet benchmark on stepwise de-biased models")

    parser.add_argument(
        "--model",
        action="store",
        type=str,
        default="GPT2LMHeadModel",
        choices=[
            "GPT2LMHeadModel",
            "GPTJForCausalLM"
        ],
        help="Class of model to evaluate (corresponds to a HuggingFace class). Remains identical to class of original model",
    )
    parser.add_argument(
        "--model_name_or_path",
        action="store",
        type=str,
        default="gpt-2-medium",
        choices=["gpt2-medium", "gpt2-xl", "EleutherAI/gpt-j-6B", "ai-forever/mGPT", "malteos/gpt2-wechsel-german-ds-meg", "malteos/gpt2-xl-wechsel-german"],
        help="Huggingface model name or path of original LM",
    )
    parser.add_argument(
        "--load_path",
        action="store",
        type=str,
        help="Path to local directory where subdirectories with checkpoints for all update steps are stored",
    )
    parser.add_argument(
        "--input_file",
        action="store",
        type=str,
        default="dev_en.json",
        help="The data to use for StereoSet intrasentence evaluation. Default is English dev set"
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        default="./results/stereoset",
        help="Directory to save predictions to"
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        default=1,
        help="The batch size to use during StereoSet intrasentence evaluation.",
    )


        print("Running StereoSet:")
        print(f" - model: {args.model}")
        print(f" - model_name_or_path: {args.model_name_or_path}")
        print(f" - load_path: {args.load_path}")
        print(f" - input_file: {args.input_file}")
        print(f" - output_file: {args.output_dir}")
        print(f" - batch_size: {args.batch_size}")

        kwargs = {}

        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

        # Loop through model checkpoints
        base_dir = args.load_path
        i = 100
        for d in os.listdir(base_dir):
            print(args.load_path+"/"+d)
            model = getattr(models, args.model)(
            args.load_path+"/"+d, **kwargs)

            model.eval()

            # Evaluate after each update batch
            runner = StereoSetRunner(
                intrasentence_model=model,
                tokenizer=tokenizer,
                input_file=f"./data/stereoset/{args.input_file}",
                model_name_or_path=args.model_name_or_path,
                batch_size=args.batch_size,
                is_generative=True, # All tested LMs are generative
                is_self_debias=False,
                bias_type=None # Only necessary for self-debias
            )
            results = runner()

            os.makedirs("./results/stereoset/"+args.output_dir, exist_ok=True)
            with open(
                "./results/stereoset/"+args.output_dir+"/"+str(i)+".json", "w"
            ) as f:
                json.dump(results, f, indent=2)
            i += 100
