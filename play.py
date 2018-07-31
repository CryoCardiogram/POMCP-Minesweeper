from globals import *
from state import State
from board import Board
from player import AbstractPlayer

def play_game(player, board): 
    assert isinstance(player, AbstractPlayer)
    assert isinstance(board, Board)
    steps = 0
    game_over = False
    win = False

    state = State(board)
    # first move (first cell is always safe)
    #state.probe(1)
    while not game_over:
        cell = player.selectCell(state)
        val = state.probe(cell)

        if val is MINE: 
            game_over = True
            print("GAME OVER\n")
        elif len(state.unknown) <= state.mines:
            print("WIN\n")
            game_over = True
            win = True

        steps+=1
    return (win, steps)
