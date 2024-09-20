import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

def load_data(folder_path, tokenizer, max_length=1024):
    """Load data from all CSV files in the specified folder and tokenize it."""
    full_dataset = read_csv_files(folder_path)
    tokenized_data = tokenize_dataset(full_dataset, tokenizer)
    padded_inputs, padded_targets = pad_tokenized_data(tokenized_data, max_length)
    return padded_inputs, padded_targets

def read_csv_files(folder_path):
    """Read all CSV files in the specified folder and return a concatenated DataFrame."""
    all_data = []

    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder '{folder_path}' does not exist.")

    # Iterate over all CSV files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # Read the CSV file
            df = pd.read_csv(file_path)
            # Append the data to the list
            all_data.append(df)

    # Check if any data was loaded
    if not all_data:
        raise ValueError("No CSV files found in the specified folder.")

    # Concatenate all dataframes into a single dataframe
    return pd.concat(all_data, ignore_index=True)

def tokenize_dataset(full_dataset, tokenizer):
    """Tokenize the original and simplified sentences in the dataset."""
    tokenized_data = []
    for index, row in full_dataset.iterrows():
        original = row['original']
        simplified = row['simplified']
        # Tokenize the original and simplified sentences
        tokenized_input = tokenizer.encode(original, return_tensors='pt').squeeze(0)  # Remove the batch dimension
        tokenized_target = tokenizer.encode(simplified, return_tensors='pt').squeeze(0)  # Remove the batch dimension
        tokenized_data.append((tokenized_input, tokenized_target))
    return tokenized_data

def pad_tokenized_data(tokenized_data, max_length=None):
    """Pad the tokenized inputs and targets to the same length."""
    # Separate inputs and targets
    inputs, targets = zip(*tokenized_data)

    # Pad the sequences
    padded_inputs = pad_sequence(inputs, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)

    # If max_length is specified, truncate or pad to that length
    if max_length is not None:
        padded_inputs = torch.nn.functional.pad(padded_inputs, (0, max_length - padded_inputs.size(1)), value=0)[:, :max_length]
        padded_targets = torch.nn.functional.pad(padded_targets, (0, max_length - padded_targets.size(1)), value=0)[:, :max_length]

    # Ensure that targets are the same length as inputs
    if padded_inputs.size(1) != padded_targets.size(1):
        raise ValueError("Padded inputs and targets must have the same length.")

    return padded_inputs, padded_targets
