
# MEMIT

The scripts in this directory contain the core funcitonalities of the MEMIT algorithm. To apply MEMIT as a bias mitigation strategy the original code by Meng et al. (2022) 
(repository: [`https://github.com/kmeng01/memit`](https://github.com/kmeng01/memit) has been adapted. All changes to the original scripts are commented accordingly.

[`memit_debias.py`](memit_debias.py) offers two options to execute the MEMIT update for de-biasing:

For the standard update run the following command:
```
python3 -m memit.memit_debias --model_name=<model-name> --hparams_fname=MEMIT/<model-name>.json \
  --stepwise=False --requests_file=./data/rewrite_prompts/rewrite_prompts_<type>/<path-to-file>.json \
  --out_path=./results/<model-name/model-<rewrite-type>
```
This command is also used to execute the biasing updates. To this end simply specify the resprective file paths.

To apply the update in smaller batches, which allow for intermediate evaluation, use the command below:
```
python3 -m memit.memit_debias --model_name=<model-name> --hparams_fname=MEMIT/<model-name>.json \
  --stepwise=True --requests_file=./data/rewrite_prompts/rewrite_prompts_<type>/<path-to-file>.json \
  --out_path=./results/<model-name/model-<rewrite-type>
```

__*Source*__:
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. "Locating and Editing Factual Associations in GPT." Advances in Neural Information Processing Systems 36 (2022).
