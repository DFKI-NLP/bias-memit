
# Data collection and pre-processing

The scripts in this directory have been used to gather the data for the causal tracing as well as updating experiments of this thesis. All scripts can be executed  
from the main directory with the commands given below.

- [`data_preparation_causal_tracing.py`](data_preparation_causal_tracing.py): Prepare tracing prompt templates and update IDs after manual review (set ```edit_ids=True```).
  ```
  python3 -m data.debiasing_data_preparation.data_preparation_causal_tracing.py \
    --model_name=<model-name> --trace_file=./data/tracing_prompts/<file-name>.json --lang=<language> \
    --edit_ids=<true-or-false>
  ```
- [`antonym_updates.py`](antonym_updates.py): Generate antonym-based updates.
  ```
  python3 -m data.debiasing_data_preparation.antonym_updates --model_name=<model-name> \
   --in_file=./data/tracing_prompts/<tracing-prompt-file>.json \
   --outfile=./data/rewrite_prompts/rewrite_promtps_antonyms/<file-name>.json --lang=<language>
  ```
- [`quantifying_updates.py`](quantifying_updates.py): Prepare templates for quantifying updates. This script
  needs to be executed twice for stereotypical and anti-stereotypical quantification. Attributes are inserted manually.
  ```
  python3 -m data.debiasing_data_preparation.quantifying_updates --lang=<language> \
    --out_file=./data/rewrite_prompts_quantifiers/<language>/<file-name>.json
  ```
- [`neutral_updates.py`](neutral_updates.py): Generate neutral attributes based on quantified stereotypes (requires manual review).
  ```
  python3 -m data.debiasing_data_preparation.neutral_updates --in_file=./data/rewrite_prompts/rewrite_prompts_quantifiers/<language>/<file-name>.json \
   --out_file=./data/rewrite_prompts/rewrite_prompts_neutral/<language>/<file-name>.json --lang=<language>
  ```
- [`biasing_updates.py`](biasing_updates.py): Generates stereotypical updates, which should introduce additional bias (used for evaluation purposes only).
  ```
  python3 -m data.debiasing_data_preparation.biasing_updates --lang=<language> \
    --out_file_file=./data/rewrite_prompts/rewrite_prompts_bias/<file-name>.json
  ```
