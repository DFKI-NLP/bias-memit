
# Evaluation

The scripts in the subdirectories of this folder allow to replicate the three-fold evaluation of proposed in this thesis:

- [`experiments`](experiments): Main evaluation scripts  (see [`experiments/README.md`](experiments/README.md`) for further details and instructions how to execute the programms)
  - Scripts to evaluate English, German and multilingual models on StereoSet
  - Evaluation of sentence entropy and perplexity
  - Qualtitative evaluation
- [`data/stereoset`](data/stereoset): English and German development set of StereoSet. The original English data is available under the following link: [`https://huggingface.co/datasets/stereoset`](https://huggingface.co/datasets/stereoset).  
  A German translation is provided under: [`https://huggingface.co/datasets/roskoN/stereoset_german`](https://huggingface.co/datasets/roskoN/stereoset_german)
- [`results`](results): The results of the evaluation on StereoSet as well as the analysis of sentence entropy and perplexity are stored in this directory.
- [`bias_bench`](bias_bench): Additional data classes and utility functions for the *StereoSet* evaluation
  (all scripts in this directory are written by Meade et al. (2022) and can be found in the repository [`https://github.com/McGill-NLP/bias-bench/tree/main`](https://github.com/McGill-NLP/bias-bench/tree/main)).

***Sources***
- Original dataset: Nadeem, Moin and Bethke, Anna and Reddy, Siva. "StereoSet: Measuring stereotypical bias in pretrained language models". arXiv preprint arXiv:2004.09456 (2020).
  Repository: [`https://github.com/moinnadeem/StereoSet`](https://github.com/moinnadeem/StereoSet); Huggingface: [`https://huggingface.co/datasets/stereoset`](https://huggingface.co/datasets/stereoset)
- *StereoSet* evaluation code ([`bias_bench`](bias_bench), [`experiments`](experiments)):
  Meade, Nicholas  and Poole-Dayan, Elinor  and Reddy, Siva. "An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models".  
  Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (2022).
  Repository: [`https://github.com/McGill-NLP/bias-bench/tree/main`](https://github.com/McGill-NLP/bias-bench/tree/main)
