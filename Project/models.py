import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__() # Makes this class a delegate of torch.nn.
        
        # Input: 1x8x8, Output: 32x5x5
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4,4), stride=1, padding=0)
        self.act1 = nn.Tanh()
        
        # Input: 32x5x5, Output: 32x2x2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(2,2), stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.flat = nn.Flatten(start_dim=0) # Flatten from first dimension, wth was that error?
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.act1(self.conv1(x))
        
        x = self.act2(self.conv2(x))
        x = self.maxpool2(x)
        
        x = self.flat(x)
        x = self.fc3(x)
        
        return x
        
class QTrainer():
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss() # Mean Squared Error Loss
    
    # this is bullshit you don't understand, try again from the begining dumbfuck
    def train_step(self, memory):
        # Why repeat states and outputs?!
        states = torch.tensor([m[0] for m in memory], requires_grad=True)
        model_outputs = [m[1] for m in memory]
        rewards = [m[2] for m in memory]
        
        model_predictions = torch.zeros(size=[len(states), 1], requires_grad=True)
        target_values = torch.zeros(size=[len(states), 1], requires_grad=True)
        for i in range(len(model_predictions)): model_predictions[i] = self.model(states[i])
        for i in range(len(states)): target_values[i] = rewards[i]
        
        self.optimizer.zero_grad()
        loss = self.criterion(model_predictions, target_values)
        loss.requires_grad = True
        loss.backward()
        self.optimizer.step()
