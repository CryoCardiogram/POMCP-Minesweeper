from .globals import *
from .model import State
from .board import Board
from .player import AbstractPlayer

def play_minesweeper(player, board, log=False): 
    assert isinstance(player, AbstractPlayer)
    assert isinstance(board, Board)
    steps = 0
    game_over = False
    win = 0
    state = State(board)

    while not game_over:
        r,c = player.next_action(state)
        val = state.probe(r,c, log=log)
        if log:
            board.draw(board.knowledge)
        if val is MINE: 
            game_over = True
            if log:
                print("GAME OVER\n")
                board.draw(board.minefield)
        elif state.is_goal():
            game_over = True
            win = 1
            if log:
                print("WIN\n")
        steps+=1
    return (win, steps)
