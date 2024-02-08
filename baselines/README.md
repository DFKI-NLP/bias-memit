# Baseline: Dropout de-biasing (Webster et al. (2020)

To assess how MEMIT for bias mitigation compares to other de-biasing approaches I have adopted the strategy proposed by Webster et al. (2020)[[Paper]](https://arxiv.org/abs/2010.06032), who increased the dropout parameters of the tested LMs.

Following Webster et al. (2020) I have chosen a default value of 0.15 for all dropout parameters. All LMs have been fine-tuned on 5,000 randomly selected Wikipedia articles for five epochs.
The multilingual LM has been fine-tuned on equal proportions of English and German data. Fine-tuning data can be found in the [data](data) folder.
The notebook [fine_tuning_data_preparation.ipynb](fine_tuning_data_preparation.ipynb) has been used to gather the data.

To repeat the fine-tuning experiments run the script [dropout_regularization.py](dropout_regularization.py) with the following arguments from this directory:

```
python3 -m dropout_regularization --train_file_path=./data/<file_name> --model_name=<model_name> \
                    --dropout=0.15  --output_dir=./results/<model_name>
```
