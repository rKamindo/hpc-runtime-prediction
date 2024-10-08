import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, Manager

def create_dataset(input_file, chunksize=10000):
  if input_file.endswith('csv.bz2'):
    chunks = pd.read_csv(input_file, compression='bz2', chunksize=chunksize)
  else:
    raise ValueError("Unsupported file format")
  manager = Manager()
  dataset = manager.list()

  with Pool(processes=cpu_count()) as pool:
      # Get only the first chunk
      for chunk in chunks:
        completed_jobs = chunk[(chunk['state'] == 'COMPLETED') & (pd.notna(chunk['wallclock_req']))]
        chunk_results = pool.map(process_row, [row for _, row in completed_jobs.iterrows()])
        dataset.extend([result for result in chunk_results if result is not None])
        print(len(dataset))
  print("Final size of dataset after processing:", len(dataset))
  return list(dataset)

def process_row(row):
  job_script = generate_slurm_script(row)
  return {
      'job_script': job_script,
      'runtime': row['run_time'],
      'submit_time': row['submit_time'],
      'wallclock_req': row['wallclock_req'],
      'mem_req': row['mem_req'],
      'nodes_req': row['nodes_req'],
      'processors_req': row['processors_req'],
      'gpus_req': row['gpus_req'],
      'partition': row['partition'],
      'qos': row['qos'],
      'account': row['account'],
      'name': row['name']
  }

def generate_slurm_script(row):
  # convert wallclock_req from seconds to HH:MM:SS format 
  hours, remainder = divmod(row['wallclock_req'], 3600)
  minutes, seconds = divmod(remainder, 60)
  wallclock = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

  # total memory per node
  mem_gb = max(1, int(row['mem_req'] / 1024))

  script = f"""#!/bin/bash
  #SBATCH --job-name={row['name']}
  #SBATCH --acount={row['account']}
  #SBATCH --partition={row['partition']}
  #SBATCH --qos={row['qos']}
  #SBATCH --time={wallclock}
  #SBATCH --nodes={row['nodes_req']}
  #SBATCH --ntasks-per-node={row['processors_req'] // row['nodes_req']}
  #SBATCH --mem={mem_gb}G
  #SBATCH --output=%x_%j.out
  #SBATCH --error=%x_%j.err
  """

  if row['gpus_req'] > 0:
    script += f"#SBATCH --gres=gpu:{row['gpus_req']}\n"

  return script

class JobScriptDataset(Dataset):
  def __init__(self, dataframe):
    self.data = dataframe
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    job_script = row['job_script']
    runtime = row['runtime']
    return job_script, runtime

def create_pytorch_datasets(df, train_size=0.8):
  # sort by submission time
  df = df.sort_values('submit_time')

  # perform train-test split
  split_index = int(len(df) * 0.8)
  split_date = df.iloc[split_index]['submit_time']

  train_df = df[df['submit_time'] < split_date]
  test_df = df[df['submit_time'] >= split_date]

  train_dataset = JobScriptDataset(train_df)
  test_dataset = JobScriptDataset(test_df)

  return train_dataset, test_dataset

if __name__ == "__main__":
  file_path = "data/eagle_data.csv.bz2"
  dataset = create_dataset(file_path)

  # convert the dataset to DataFrame
  df = pd.DataFrame(dataset)

  # save data frame
  df.to_pickle("data/processed_dataset")

  # Print DataFrame info for debugging
  print("New DataFrame columns:", df.columns.tolist())
  print("DataFrame shape:", df.shape)
  print(df.head())


  # Create PyTorch datasets
  train_dataset, test_dataset = create_pytorch_datasets(df, 0.8)

  print(f"Train dataset size: {len(train_dataset)}")
  print(f"Test dataset size: {len(test_dataset)}")
