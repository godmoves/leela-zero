import numpy as np
# from sgfmill import ascii_boards
from sgfmill import sgf
from sgfmill import sgf_moves
import tensorflow as tf

from tfprocess import TFProcess


BATCH_SZIE = 32


def get_position(board):
    pos = np.zeros((19, 19))
    player_moves = board.list_occupied_points()
    for move in player_moves:
        if move[1] is None:
            continue
        else:
            row, col = move[1]
        if move[0] == 'b':
            pos[row, col] = 1
        elif move[0] == 'w':
            pos[row, col] = -1
    return pos


def opp_color(color):
    if color == 'b':
        return 'w'
    elif color == 'w':
        return 'b'
    else:
        return None


def sgf_to_position(sgf_name):
    with open(sgf_name, "rb") as f:
        try:
            game = sgf.Sgf_game.from_bytes(f.read())
        except ValueError:
            raise Exception("Can't read this file.")
    assert game.get_size() == 19
    try:
        board, player_moves = sgf_moves.get_setup_and_moves(game)
    except ValueError as e:
        raise Exception(str(e))
    move_number = len(player_moves)
    print("Total move number:", move_number)
    positions = [get_position(board)]
    for color, move in player_moves:
        if move is None:
            continue
        row, col = move
        # opp = opp_color(color)
        try:
            board.play(row, col, color)
        except ValueError:
            raise Exception("illegal move in sgf file")
        positions.append(get_position(board))
    # for pos in positions:
    #     print(pos, '\n')

    return positions


def get_plane(position, move_color, player_color):
    if move_color == 'b':
        plane = np.where(position > 0.5, 1, 0)
    elif move_color == 'w':
        plane = np.where(position < -0.5, 1, 0)
    else:
        return None

    return plane


def position_to_feature(positions):
    player_color = 'b'
    index = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
    for i, pos in enumerate(positions):
        plane = np.zeros((1, 18, 19, 19))

        back_move = min(i + 1, 16)
        move_color = player_color
        for b in range(back_move):
            plane[0, index[b], :, :] = get_plane(positions[i - b], move_color, player_color)
            move_color = opp_color(move_color)

        player_color = opp_color(player_color)
        if player_color == 'b':
            plane[0, 16, :, :] = np.ones((19, 19))
        elif player_color == 'w':
            plane[0, 17, :, :] = np.ones((19, 19))

        if i == 0:
            feature_plane = np.zeros((1, 18, 19, 19))
        else:
            feature_plane = np.concatenate((feature_plane, plane), axis=0)

    return feature_plane


positions = sgf_to_position("/home/pwjtc/Work/GodMoves/sgf/sgf-GM-RG-0.sgf")
f_plane = position_to_feature(positions)
print(np.shape(f_plane))

tfprocess = TFProcess()

inputs = tf.placeholder(tf.float32, [None, 18, 19, 19])
polocy, value = tfprocess.construct_net(inputs)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    p, v = sess.run([polocy, value], feed_dict={inputs: f_plane})
    tfprocess.restore("leelaz-model-0")
