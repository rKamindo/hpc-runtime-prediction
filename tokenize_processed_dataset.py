import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class JobScriptDataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_length=150):
    self.data = dataframe
    self.tokenizer = tokenizer
    self.max_length = max_length
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    job_script = row['job_script']
    runtime = row['runtime']
    
    encoding = self.tokenizer.encode_plus(
      job_script,
      add_special_tokens=True,
      max_length=self.max_length,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt'
    )
    
    return {
      'input_ids': encoding['input_ids'].squeeze(),
      'attention_mask': encoding['attention_mask'].squeeze(),
      'runtime': torch.tensor(runtime, dtype=torch.float)
    }

def create_pytorch_datasets(df, train_size=0.8):
  # sort by submission time
  df = df.sort_values('submit_time')

  # perform train-test split
  split_index = int(len(df) * 0.8)
  split_date = df.iloc[split_index]['submit_time']

  train_df = df[df['submit_time'] < split_date]
  test_df = df[df['submit_time'] >= split_date]

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  train_dataset = JobScriptDataset(train_df, tokenizer)
  test_dataset = JobScriptDataset(test_df, tokenizer)

  print(f"Total dataset size: {len(df)}")
  print(f"Train dataset size: {len(train_dataset)}")
  print(f"Test dataset size: {len(test_dataset)}")
  print(f"Split date: {split_date}")
  
  print("\nTrain dataset:")
  print_dataset_sample(train_dataset, 3)
  
  print("\nTest dataset:")
  print_dataset_sample(test_dataset, 3)

  return train_dataset, test_dataset

def print_dataset_sample(dataset, n_samples):
  for i in range(min(n_samples, len(dataset))):
    sample = dataset[i]
    print(f"\nSample {i+1}:")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")
    print(f"Runtime: {sample['runtime'].item()}")
    
    # decode first 50 tokens
    decoded_text = dataset.tokenizer.decode(sample['input_ids'][:50]) 
    print(f"Decoded text (first 50 tokens): {decoded_text}")

df = pd.read_pickle("data/processed_dataset")
train_dataset, test_dataset = create_pytorch_datasets(df, 0.8)

