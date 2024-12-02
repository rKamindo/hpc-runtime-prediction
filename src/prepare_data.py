import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from dataset import JobScriptDataset

def split_data(df, train_size, sample_size):
    if sample_size is not None:
      df = df.sample(sample_size, random_state=42)
    df = df.sort_values('submit_time')
    split_index = int(len(df) * train_size)
    split_date = df.iloc[split_index]['submit_time']

    train_df = df[df['submit_time'] < split_date]
    test_df = df[df['submit_time'] >= split_date]
    
    return train_df, test_df

def prepare_data_for_transformer(df, tokenizer, train_size=0.8, sample_size=50000, batch_size=32):
  
  train_df, test_df = split_data(df, train_size, sample_size)
  
  scaler = StandardScaler()
  scaler.fit(train_df['runtime'].values.reshape(-1, 1)) 
  
  train_dataset = JobScriptDataset(train_df, tokenizer, scaler=scaler)
  test_dataset = JobScriptDataset(test_df, tokenizer, scaler=scaler)
  
  print(f"Total dataset size: {len(df)}")
  print(f"Train dataset size: {len(train_dataset)}")
  print(f"Test dataset size: {len(test_dataset)}")
  print(f"Sample size: {sample_size}")
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
  return train_loader, test_loader
