
# Causal Tracing

The scripts provided in this folder and all subfolders have been developed by Meng et al. (2022) (repository: [`https://github.com/kmeng01/memit`](https://github.com/kmeng01/memit)).

For the purpose of this thesis the script [`causal_trace.py`](causal_trace.py) is most relevant. It collects causal traces for a set of (pre-filtered) stereotypical tracing prompts. 
To obtain traces for a specific model run the following command:

```
python3 -m experiments.causal_trace --model_name=<model-name>  \
    --fact_file=./data/tracing_prompts/tracing_prompts_<model-name>.json  \
    --output_dir=./results/<model-name>/causal_traces
```

Other files contain additional functions used in the main program.

__*Source*__:
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. "Locating and Editing Factual Associations in GPT." Advances in Neural Information Processing Systems 36 (2022).
