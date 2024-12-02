import torch
from model import BERTRegression
from transformers import DistilBertTokenizer
from train import train_model, eval_model
from prepare_data import prepare_data_for_transformer
import pandas as pd

df = pd.read_pickle("../data/processed/processed_dataset")
train_split = 0.1
sample_size = 100000
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
batch_size = 64
learning_rate = 0.0003
lambda_reg = 2e-5
num_epochs = 5
verbose = True

train_loader, test_loader = prepare_data_for_transformer(df, tokenizer, train_split, sample_size, batch_size)


model = BERTRegression(model_name)
model.load_state_dict(torch.load('../best_model/best_model.pth', weights_only=True))
train_model(model, train_loader, test_loader, num_epochs, learning_rate, lambda_reg, verbose)