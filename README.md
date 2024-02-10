# Mass-Editing Stereotypical Associations to Mitigate Bias in Language Models

This repository contains the scripts and data to replicate the experiments of the master thesis "Mass-Editing Stereotypical Associations to Mitigate Bias in Language Models", which has been carried out 
as as a cooperation between Potsdam University (Department of Linguistics) and "Deutsches Forschungszentrum für künstliche Intelligenz" (DFKI) - "Speech and Language Technology Lab".  
The goal of this study is to approach bias mitigation in pre-trained Transformer language models (LMs) as a knowledge update. To this end it employs the "Mass-Editing Memory in a Transformer" (MEMIT) algorithm
by Meng et al. (2022). 
This repository provides four different sets with anti-stereotypical updates from four bias domains (gender, profession, race, religion) in English and German. It currently supports editing three English, two German
and one multilingual LM.
There are three possibilities to evalute the de-biasing results: an instrinsic evaluation on the <em>StereoSet</em> (Nadeem et al., 2021) bias benchmark dataset, a quantitative analysis of entropy and perplexity 
and a qualitative assessment of selected examples.

## Table of Contents

- [Installation](#installation)
- [Causal Tracing](#causal-tracing)
- [MEMIT for Bias Mitigation](#memit-for-bias-mitigation)
- [Evaluation](#evaluation)
- [External Sources](#external-sources)

## Installation

To set up the environment and dependencies for the MEMIT update Meng et al. (2022) provide a shell script, which can be found under [`scripts`](scripts). It is recommended to use `conda` to install Python, CUDA, and PyTorch and `pip` for all other dependencies. First install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```

`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`.

## Causal Tracing

Before running the MEMIT update one needs to identify locations where the weight updates should be applied. The scripts for this step and further instructions on how to run them can be found in the following folder: [`experiments`](experiments).

## MEMIT for Bias Mitigation

[`notebooks/memit_debias.ipynb`](notebooks/memit_debias.ipynb) is a simple demonstration to illustrate how MEMIT can be applied to alter stereotypical associations.
MEMIT takes as input rewrites of the following form:

```python
request = [
    {
        "prompt": " All {} are",
        "subject": "princesses",
        "target_new": {
            "str": "ugly"
        }
    },
    {
        "prompt": "{} are generally",
        "subject": "Muslims",
        "target_new": {
            "str": "violent"
        }
    },
]
```

Other examples can be found in the notebook.
The scripts for the actual large scale updates can be found in the directory ['memit'](memit) with further instructions on how to execute the code.

## Evaluation

[`evaluation`](evalaution) contains the scripts and notebooks for the evaluation on <em>StereoSet</em>, a quantiative analysis as well as a notebook for the inspection of generated exmples ['evaluation/experiments/qualitative_evaluation.ipynb'](evaluation/experiments/qualitative_evaluation.ipynb). Detailed instructions and explanations can also be found in 
the respective directories.

## External Sources and Source Code

- __Causal tracing and MEMIT algorithm__:
  - Paper: Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. "Locating and Editing Factual Associations in GPT."
       Advances in Neural Information Processing Systems 36 (2022).
  - Code: [Meng et al. (2022)](https://github.com/kmeng01/memit)
- __*StereoSet*__:  
    - Paper: Nadeem, Moin and Bethke, Anna and Reddy, Siva. "StereoSet: Measuring stereotypical bias in pretrained language models".
          arXiv preprint arXiv:2004.09456 (2020). 
    - Data: [`https://huggingface.co/datasets/stereoset`](https://huggingface.co/datasets/stereoset);
          [`https://github.com/moinnadeem/StereoSet](https://github.com/moinnadeem/StereoSet)
    - Evaluation scripts:
      - Paper: Meade, Nicholas and Poole-Dayan, Elinor and Reddy, Siva. "An Empirical Survey of the Effectiveness of Debiasing
            Techniques for Pre-trained Language Models". Proceedings of the 60th Annual Meeting of the Association for Computational
            Linguistics (Volume 1: Long Papers) (2022).
      - Code: [`https://github.com/McGill-NLP/bias-bench/tree/main`](https://github.com/McGill-NLP/bias-bench/tree/main)
