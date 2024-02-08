
# Evaluation scripts

The files provided under this directory allow to evaluate MEMIT de-biased mode in three ways:

1. __*StereoSet*__ (Nadeem et al. (2020): In this intrinsic evaluation the task for an LM is to choose whether a stereotypical,  
   an anti-stereotypical or a neutral option is most likely to precede a given sentence (intrasentence level).
   The evaluation on *StereoSet* is done in two steps:
   
   1. Select the continuation option with the highest probability.
      For an un-debiased model run the command:
      ```
      python3 -m experiments.stereoset --model=<model-class> --model_name_or_path=<model-name> \
        --input_file=dev_<en-or-de>.json --output_file=<path-to-predictions-file>/<predictions_filename> --batch_size=1
      ```
      
      To generate predictions with a de-biased LM use the command:  
      ```
      python3 -m experiments.stereoset_debias --model==<model-class> --model_name_or_path=<original-model-name> \
        --load_path=<local-path-to-debiased-LM-checkpoint> --input_file=dev_<en-or-de>.json \
        --output_file=<path-to-predictions-file>/<predictions_filename> \  
        --batch_size=1
      ```
      
      Stepwise de-biased model can be evaluated as follows:
      ```
      python3 -m experiments.stereoset_debias-stepwise --model==<model-class> --model_name_or_path=<original-model-name> \  
        --load_path=<local-path-to-dir-with-checkpoint-subdirectories> --input_file=dev_<en-or-de>.json \  
        --output_dir=<directory-to-save-predictions-to> \
        --batch_size=1
      ```
      
   2. Compute the Language Modeling Scores, Stereotype Scores and idealised CAT score for each set of predictions:
      ```
      python3 -m experiments.stereoset_evaluation \
        --predictions_file=<path-to-predictions-file>.json \  
        --output_file=<path-to-score-file>.json \
        --gold_file=dev_<en-or-de>.json
      ```
      
      For the stepwise de-biased models specify a directory, which contains a number of predictions files (one after each update batch):
      ```
      python3 -m experiments.stereoset_evaluation \
        --predictions_dir=<path-to-prediction-files-dir> \  
        --output_file=<path-to-score-file>.json \
        --gold_file=dev_<en-or-de>.json
      ```
   Note that prediction and result files are located in a subdirectory of ./results/stereoset. File paths need to be specified relative to this directory!  
2. __Sentence entropy and perplexity__: The second evaluation method aggregates average entropy and perplexity over each one of the sets
      of tracing prompts, antonym-based updates, quantified (anti-)stereotypical rewrites and biased updates
      (output files are stored under ./results/entr_ppl_eval/):
      ```
      python3 -m experiments.entr_ppl_evaluation --original_model=<model-name> --debiased_model=<path-to-debiased-model> \
         --sentences=../../data/<path-to-tracing-or-rewrite-file>.json --sentence_type=<tracing-or-update> \  
         --output_file=<file-name>.csv
      ```

4. __Qualitative evaluation__: The notebook [`qualitative_evaluation.ipynb`](qualitative_evaluation.ipynb) allows
      a more interactive, qualitative assessment of the effects of de-biasing. The user can test how the predictions differ between
      an un-debiased and de-biased model.
