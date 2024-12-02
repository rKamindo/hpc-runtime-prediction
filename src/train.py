import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from datetime import datetime


def compute_regularization(model, lambda_reg):
  reg_loss = 0.0
  for param in model.parameters():
    reg_loss += torch.sum(param**2)
  return lambda_reg * reg_loss


def eval_model(model, test_loader):
  model.eval()
  val_loss = 0.0
  criterion = nn.MSELoss()
  with torch.no_grad():
    for batch in test_loader:
      inputs = batch['input_ids'].to(model.device)
      attention_mask = batch['attention_mask'].to(model.device)
      targets = batch['runtime'].to(model.device)
      
      outputs = model(inputs, attention_mask).view(-1)
      loss = criterion(outputs, targets)
      val_loss += loss.item()
  return val_loss / len(test_loader)


def train_model(model, train_loader, test_loader, num_epochs, learning_rate, lambda_reg, verbose, early_stopping_patience=10):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
  best_val_loss = float('inf')
  patience = 0
  train_losses, val_losses = [], []
  
  num_training_steps = len(train_loader) * num_epochs
  num_warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps for warmup
  
  scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=num_warmup_steps,
      num_training_steps=num_training_steps
  )
  
  for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
      optimizer.zero_grad()
      inputs = batch['input_ids'].to(model.device)
      attention_mask = batch['attention_mask'].to(model.device)
      targets = batch['runtime'].to(model.device)
      
      # forward pass
      outputs = model(inputs, attention_mask).view(-1)
      loss = criterion(outputs, targets)
      reg_loss = compute_regularization(model, lambda_reg)
      total_loss = loss + reg_loss
      
      # backpropogation and optimization
      total_loss.backward()
      optimizer.step()
      scheduler.step()
      
      epoch_loss += total_loss.item()
      
    train_losses.append(epoch_loss / len(train_loader))
    
    # validation
    val_loss = eval_model(model, test_loader)
    val_losses.append(val_loss)
    
    if verbose:
      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
      
    # early stopping
    if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(model.state_dict(), 'best_model.pth')  # Ensure this filename is consistent
          patience = 0
    else:
        patience += 1
        if patience >= early_stopping_patience:
            print("Early stopping triggered")
            break

  # plotting loss curves
  plt.figure()
  plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
  plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

  plt.savefig(f'../results/training_validation_loss_{timestamp}.png')  # save the plot
  plt.close()
      