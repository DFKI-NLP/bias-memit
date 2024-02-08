"""
A script generate predictions with un-debiased models on the StereoSet intrasentence split.

The original script is provided by Meade et al. (2022) (Repository: https://github.com/McGill-NLP/bias-bench/tree/main)

It has been adapted and reduced to the functionalities necessary for the evaluation strategy in this thesis.
Functions needed to evaluate other de-biasin techniques tested by Meade et al. (2022) are omitted for simplicity.

:authors: Nicholas Meade, Elinor Poole-Dayan, Siva Reddy (Repository: https://github.com/McGill-NLP/bias-bench/tree/main)
"""
import argparse
import json
import os

import transformers

from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.model import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs StereoSet benchmark on an un-debiased model")

    # Reduced to the two LM classes investigated in this thesis
    parser.add_argument(
        "--model",
        action="store",
        type=str,
        default="GPT2LMHeadModel",
        choices=[
            "GPT2LMHeadModel",
            "GPTJForCausalLM"
        ],
        help="Class of model to evaluate (corresponds to a HuggingFace class)",
    )

    parser.add_argument(
        "--model_name_or_path",
        action="store",
        type=str,
        default="gpt2-medium",
        choices=["gpt2-medium", "gpt2-xl", "EleutherAI/gpt-j-6B",
                 "ai-forever/mGPT", "malteos/gpt2-wechsel-german-ds-meg", "malteos/gpt2-xl-wechsel-german"],
        help="HuggingFace model name or path",
    )

    parser.add_argument(
        "--input_file",
        action="store",
        type=str,
        default="dev_en.json",
        help="The data to use for StereoSet intrasentence evaluation. Default is English dev set",
    )

    parser.add_argument(
        "--output_file",
        action="store",
        type=str,
        default="stereoset_original_predictions.json",
        help="Output file for predictions."
    )

    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        default=1,
        help="The batch size to use during StereoSet intrasentence evaluation.",
    )

    args = parser.parse_args()

    print("Running StereoSet:")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - input_file: {args.input_file}")
    print(f" - output_file: {args.output_file}")
    print(f" - batch_size: {args.batch_size}")


    model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"./data/stereoset/{args.input_file}",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=True # All LMs investigated in this work are generative; no additional argument for simplicity
    )

    # Run predictions on StereoSet and save results for evaluation
    results = runner()

    os.makedirs(f"./results/stereoset/", exist_ok=True)

    with open(
        f"./results/stereoset/{args.output_file}.json", "w"
    ) as f:
        json.dump(results, f, indent=2)
