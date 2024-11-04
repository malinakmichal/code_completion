from position import Coordinates
from figure import Figure
from moves import filter_moves, get_score
import math


def minimax(board, color, depth, alpha, beta, maximize):
    # with fewer figures depth should be higher todo
    best_fig = Figure()
    best_move = (Coordinates(-1, -1), False)

    # if reached the selected depth, blacks score will be returned
    if depth == 0:
        return get_score(board, "black"), (best_fig, best_move)

    if color == "white":
        opponent_color = "black"
    else:
        opponent_color = "white"

    if maximize:
        best_score = -math.inf
    else:
        best_score = math.inf

    for x in range(8):
        for y in range(8):
            if board[x][y].get_type() != "figure" and board[x][y].get_color() == color:
                # figure has eligible moves that can be played
                moves = board[x][y].get_moves(board)
                figure = board[x][y]
                moves = filter_moves(moves, board, figure)

                for move in moves:
                    result = figure.move(move, board, True)
                    score = minimax(board, opponent_color, depth - 1, alpha, beta, not maximize)[0]
                    figure.unmake_move(board, result)

                    if maximize:
                        if score > best_score:
                            best_score = score
                            best_move = move
                            best_fig = figure

                        if alpha <= score:
                            alpha = score

                        if alpha >= beta:
                            return best_score, (best_fig, best_move)
                    else:
                        if score < best_score:
                            best_score = score
                            best_move = move
                            best_fig = figure

                        if beta >= score:
                            beta = score
                        if alpha >= beta:
                            return best_score, (best_fig, best_move)

    return best_score, (best_fig, best_move)
