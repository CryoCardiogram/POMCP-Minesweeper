from minesweeper.globals import *
from minesweeper.state import State
from minesweeper.board import Board
from minesweeper.player import AbstractPlayer

def play_minesweeper(player, board): 
    assert isinstance(player, AbstractPlayer)
    assert isinstance(board, Board)
    steps = 0
    game_over = False
    win = 0

    state = State(board)
    # first move (first cell is always safe)
    #state.probe(1)
    while not game_over:
        r,c = player.next_action(state)
        print(r)
        print(c)
        val = state.probe(r,c)

        if val is MINE: 
            game_over = True
            print("GAME OVER\n")
        elif board.win():
            print("WIN\n")
            game_over = True
            win = 1

        steps+=1
    return (win, steps)
