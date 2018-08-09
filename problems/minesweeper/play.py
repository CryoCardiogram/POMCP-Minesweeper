from .globals import *
from .model import State
from .board import Board
from .player import AbstractPlayer

def play_minesweeper(player, board): 
    assert isinstance(player, AbstractPlayer)
    assert isinstance(board, Board)
    steps = 0
    game_over = False
    win = 0
    state = State(board)

    while not game_over:
        r,c = player.next_action(state)
        val = state.probe(r,c)
        print(board)
        if val is MINE: 
            game_over = True
            print("GAME OVER\n")
        elif state.is_goal():
            print("WIN\n")
            game_over = True
            win = 1
        steps+=1
    return (win, steps)
