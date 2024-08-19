import chess.pgn
import pandas as pd
import re

pgn_file_path = "D:/_Dev-HDD/Chess-Position-Evaluation/lichess_db_standard_rated_2024-07_no_time.pgn"

headers = ["Index", "Result", "WhiteElo", "BlackElo", "Moves"]

# open the pgn_exmaple.pgn file, remove all time control information, then write to a new file
# example 1. b3 {  [%clk 0:01:00]  } c6 {  [%clk 0:01:00]  } 2. c3 {  [%clk 0:01:00] -> 1. b3 c6 2. c3
def delete_time():
    index = 0
    with open(pgn_file_path) as pgn_file:
        with open("D:/_Dev-HDD/Chess-Position-Evaluation/lichess_db_standard_rated_2024-07_no_time.pgn", "w") as new_pgn_file:
            for line in pgn_file:
                new_line = re.sub(r"\{.*?\}", "", line)
                
                # Remove all secondary move numbers. Eg. 1. e3 1... e6 -> 1. e3 e6
                new_line = re.sub(r"\d+\.\.\.", "", new_line)
                
                # There are two extra spaces in between each player's moves. Eg. 1. e3   e6 instead of 1. e3 e6, fix the line above to remove these extra spaces
                new_line = re.sub("   ", " ", new_line)
                
                new_pgn_file.write(new_line)
                
                index += 1
                if index % 100000 == 0: print(str(index) + ", " + str(index/1000000) + "%")

def open_games(start_index, end_index):
    data_frame = pd.DataFrame(columns=headers)
    
    with open(pgn_file_path) as pgn_file:
        for i in range(start_index, end_index):
            game = chess.pgn.read_game(pgn_file) # Iterate over all the games, what happens if we exit this function and return later?
        
            data_frame.loc[i] = [i, game.headers["Result"], game.headers["WhiteElo"], game.headers["BlackElo"], game.mainline_moves()]
            
    print(data_frame)
        
delete_time()