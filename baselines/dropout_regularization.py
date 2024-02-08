"""
This script applies dropout regularization as a bias mitigation strategy as proposed by Webster et al. (2020). 
Hyperparameter values for the original experiments are adapted from https://arxiv.org/pdf/2010.06032.pdf but can be set as desired.
Models are fine-tuned on 5,000 randomly sampled articles from Wikipedia (provided in the folder ./data) for five epochs.

:author: Karina Hensel
"""

import argparse
import json
import os
import torch
from transformers import TextDataset, DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTJForCausalLM, AutoConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments


def load_dataset(file_path, tokenizer, block_size=128):
    """Load the training dataset from disk

    :param file_path: path to dataset
    :type file_path: str
    :param tokenizer: tokenizer for preprocessing the data
    :type tokenizer: AutoTokenizer
    :param block_size: size of text blocks to split the data into (set to max size)
    :type block_size: int
    :returns: TextDataset object
    :rtype: TextDataset"""

    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset


def load_data_collator(tokenizer, mlm=False):
    """Load a DataCollatorForLanguageModeling
    
    :param tokenizer: tokenizer to use for processing
    :type tokenizer: AutoTokenizer
    :param mlm: set masked language modeling to False
    :type mlm: bool
    :returns: DataCollatorForLanguageModeling object
    :rtype: DataCollatorForLanguageModeling
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm
    )
    return data_collator


def train(train_file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size,
          num_train_epochs=5):
    """Train LM with adapted hyperparameters and save it
    
    :param train_file_path: path to training data
    :type train_file_path: str
    :param model_name: LM to use
    :type model_name: str
    :param output_dir: directory to save edited LM to
    :type output_dir: str
    :param overwrite_output_dir: whethter to overwrite existing model configs
    :type overwrite_output_dir: bool
    :param per_device_train_batch_size: batch size
    :type per_device_train_batch_size: int
    :param num_train_epochs: training epochs (default 5)
    :type num_train_epochs: int
    """
    
    # Load tokenizer and fine-tuning dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding='max_length', max_length=1024)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)
    
    # Increase dropout parameters in model configuration before fine-tuning
    configuration = AutoConfig.from_pretrained(model_name)
    configuration.resid_pdrop = args.dropout
    configuration.embd_pdrop = args.dropout
    configuration.attn_pdrop = args.dropout
    
    # Initialize model with changed configuration file for fine-tuning
    # If GPT-J: load GPTJForCausalLM model
    if model_name == "EleutherAI/gpt-j-6B":
        model = GPTJForCausalLM.from_pretrained(model_name, config=configuration).to(device)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration).to(device)
    
    model.save_pretrained(output_dir)
    
    training_args = TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
      )

    trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
  )
              
    # Fine-tune and save LM  
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Apply dropout regularization as a bias mitigation method.")
    
    parser.add_argument(
        "--train_file_path",
        action="store",
        type=str,
        default="./data",
        help="Directory where fine-tuning data is stored",
    )
    
    parser.add_argument(
        "--model_name",
        action="store",
        type=str,
        default="gpt-2-medium",
        choices=[
            "gpt2-medium", "gpt2-xl", "EleutherAI/gpt-j-6B", "ai-forever/mGPT", "malteos/gpt2-wechsel-german-ds-meg", 
            "malteos/gpt2-xl-wechsel-german"],
        help="Model to fine-tune.",
    )
    
    parser.add_argument(
        "--dropout",
        action="store",
        type=float,
        default=0.15,
        help="Set dropout parameter (default: 0.15)"
    )
    
    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        default="./results",
        help="Output directory for fine-tuned models."
    )
        
    args = parser.parse_args()

    # Check for GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("Running fine-tuning script:")
    print(f" - train_file_path: {args.train_file_path}")
    print(f" - model_name: {args.model_name}")
    print(f" - dropout: {args.dropout}")
    print(f" - output_dir: {args.output_dir}")
    
    train(args.train_file_path, args.model_name, args.output_dir, overwrite_output_dir=False, 
          per_device_train_batch_size=32, 
          num_train_epochs=5, save_steps=1000)
