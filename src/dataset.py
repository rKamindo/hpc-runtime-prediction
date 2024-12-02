import torch
from torch.utils.data import Dataset

class JobScriptDataset(Dataset):
  def __init__(self, dataframe, tokenizer, scaler, max_length=150):
    self.data = dataframe
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.scaler = scaler

  def __len__(self):
    return len(self.data)
    
  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    job_script = row['job_script']
    runtime = row['runtime']
  
    # apply z-score scaling using the fitted scaler
    normalized_runtime = self.scaler.transform([[runtime]]).item()
  
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
        'runtime': torch.tensor(normalized_runtime, dtype=torch.float)
    }