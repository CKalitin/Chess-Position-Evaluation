import torch
import torch.nn as nn
import game_loader as gl
import yaml
from collections import deque
import random

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
        

class DQNAgent:
    def __init__(self):
        with open("hyperparameters.yaml", "r") as file: self.hyperparameters = yaml.safe_load(file)['dqn-1']
        self.memory = deque([], maxlen=self.hyperparameters["memory_size"])
        self.game_loader = gl.GameLoader()
        self.model = DQN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters["learning_rate"])
    
    def train(self):
        print('cuda' if torch.cuda.is_available() else 'cpu')
        
        num_games = 0
        num_iters = 0 # Games that have been searched (some discarded due to min elo), this prevents overlap vs. only using num_games which would reuse some games
        epsilon = self.hyperparameters["epsilon_init"]
        training = True
        while training:
            training_games, iters = self.game_loader.load_games(num_iters, self.hyperparameters["training_batch_size"], self.hyperparameters["min_avg_elo"])
            num_games += self.hyperparameters["training_batch_size"]
            num_iters += iters
            
            for i in range(self.hyperparameters["training_batch_size"]):
                if random.uniform(1, 2) < epsilon: model_output = random.uniform(-1, 1)
                else: model_output = self.model(self.fen_to_tensor(self.game_loader.moves_to_fen(training_games['moves'][i], self.hyperparameters["training_move"])))
                print(model_output)
                
            if num_games > 0: training = False
        
    def fen_to_tensor(self, fen):
        piece_to_value = { 'K': 1, 'Q': 0.8, 'R': 0.5, 'B': 0.4, 'N': 0.3, 'P': 0.1, 'k': -1, 'q': -0.8, 'r': -0.5, 'b': -0.4, 'n': -0.3, 'p': -0.1 }
        tensor = torch.zeros(1, 8, 8)
        pos = 0
        for character in fen:
            if pos >= 64: break
            if character.isdigit(): pos += int(character)
            elif character != "/":
                tensor[0][pos // 8][pos % 8] = piece_to_value[character]
                pos += 1
        return tensor

agent = DQNAgent()
agent.train()