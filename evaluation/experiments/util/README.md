
# Utility functions 

All scripts in this directory have been adapted from Meng et al. (2022) (source: [`https://github.com/kmeng01/memit/tree/main`](https://github.com/kmeng01/memit/tree/main).

They contain functionalities for dependency management([`globals.py`](globals.py)), a wrapper class for hyperparameters ([`hparams.py`](hparams.py)), 
text generation ([`generate.py`](generate.py)), computing update statistics ([`logit_lens`](logit_lens.py), 
[`perplexity`](perplexity.py), [`runningstats.py`](runningstats.py)) and the functions to "hook" a model at an update layer ([`nethook.py`](nethook.py)).
