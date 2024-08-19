import chess.pgn
import pandas as pd
import itertools

pgn_file_path = "D:/_Dev-HDD/Chess-Position-Evaluation/lichess_db_standard_rated_2024-07_no_time.pgn"

# Load a range of lines from a file (eg. lines 100 to 200)
# I fucking love Python, this did it, awesome feature
class CustomPgnTextFile:
    def __init__(self, path, start_game_index, end_game_index):
        self.current_line = 0
        with open(path, 'r') as file:
            # + end_game_index because when a game is found, py chess iterates over one extra line
            self.lines_list = list(itertools.islice(file, start_game_index * 20, end_game_index * 20 + end_game_index)) # 20 lines per pgn game
    
    def readline(self):
        self.current_line += 1
        return self.lines_list[self.current_line - 1]

def load_games(start_game_index, num_games, min_white_elo):
    data_frame = pd.DataFrame(columns=["Game Index", "Result", "AvgElo", "Moves"])
    
    iters = -1
    while len(data_frame) < num_games:
        iters += 1
        if iters % num_games == 0: pgn_file = CustomPgnTextFile(pgn_file_path, start_game_index + iters, start_game_index + iters + num_games) # Batches of size num_games
        game = chess.pgn.read_game(pgn_file)
        if "WhiteElo" not in game.headers or int(game.headers["WhiteElo"]) < min_white_elo: continue
        data_frame.loc[len(data_frame)] = [iters + start_game_index, game.headers["Result"], (int(game.headers["WhiteElo"])+int(game.headers["BlackElo"]))/2, game.mainline_moves()]
    
    print(data_frame)
    
    print(f"Loaded {len(data_frame)} games. Iterated over {iters} games.")
    
load_games(10, 10, 2500)

