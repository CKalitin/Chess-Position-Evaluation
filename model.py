import torch
import torch.nn as nn
import string
import game_loader as gl

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() # Makes this class a delegate of torch.nn.
        
        # Input: 1x8x8, Output: 32x5x5
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4,4), stride=1, padding=0)
        self.act1 = nn.Tanh()
        
        # Input: 32x5x5, Output: 32x2x2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(2,2), stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.flat = nn.Flatten()
        
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.act1(self.conv1(x))
        
        x = self.act2(self.conv2(x))
        x = self.maxpool2(x)
        
        x = self.flat(x)
        
        x = self.fc3(x)
        
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

model = Model()            
game_loader = gl.GameLoader()
game_loader.load_games(100000, 10, 1500)
fen = game_loader.moves_to_fen(game_loader.games_data_frame["Moves"][0], 20)
tensor = model.fen_to_tensor(fen)

print(tensor)
