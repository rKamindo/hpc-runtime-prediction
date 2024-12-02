
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class BERTRegression(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', dropout_rate=0.1):
        super(BERTRegression, self, ).__init__()
        self.config = DistilBertConfig.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)

    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x
        