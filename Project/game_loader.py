import chess.pgn
import pandas as pd
import itertools
import string

pgn_file_path = "D:/_Dev-HDD/Chess-Position-Evaluation/lichess_db_standard_rated_2024-07_no_time.pgn"

# Load a range of lines from a file (eg. lines 100 to 200)
# I fucking love Python, this did it, awesome feature
class CustomPgnTextFile:
    def __init__(self, path, start_game_index, end_game_index):
        self.current_line = 0
        with open(path, 'r') as file:
            # + end_game_index because when a game is found, py chess iterates over one extra line, slight overlap at end_game_index > 20
            self.lines_list = list(itertools.islice(file, start_game_index * 20, end_game_index * 20 + end_game_index)) # 20 lines per pgn game
    
    def readline(self):
        self.current_line += 1
        return self.lines_list[self.current_line - 1]

class GameLoader:
    def __init__(self):
        self.headers = columns=["dataset_index", "result", "avg_elo", "moves"]

    def load_games(self, start_game_index, num_games, min_avg_elo):
        data_frame = pd.DataFrame(columns=self.headers)
        
        iters = 0
        while len(data_frame) < num_games:
            iters += 1 # Iters added before the if elo to count games correctly, correct with -1 later
            if (iters-1) % num_games == 0: pgn_file = CustomPgnTextFile(pgn_file_path, start_game_index + iters, start_game_index + iters + num_games) # Batches of size num_games
            game = chess.pgn.read_game(pgn_file)
            if "WhiteElo" not in game.headers or (int(game.headers["WhiteElo"])+int(game.headers["BlackElo"]))/2 < min_avg_elo: continue # Check if contains white elo because the final game will likely be incompletely read
            data_frame.loc[len(data_frame)] = [iters - 1 + start_game_index, game.headers["Result"], (int(game.headers["WhiteElo"])+int(game.headers["BlackElo"]))/2, game.mainline_moves()]
            
        #print(f"Loaded {len(data_frame)} games. Iterated over {iters} games.")
        
        return data_frame, iters
    
    # Multiply moves by 2 because each "move" has two moves for each player
    def moves_to_fen(self, mainline_moves, num_moves):
        board = chess.Board()
        for move in mainline_moves:
            board.push(move)
            num_moves -= 1
            if num_moves == 0: break
        return board.fen()