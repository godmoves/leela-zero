from time import clock

import numpy as np
from sgfmill import sgf
from sgfmill import sgf_moves
import tensorflow as tf

from tfprocess import TFProcess


# TODO: using batch to avoid OOM
BATCH_SZIE = 64


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
        try:
            board.play(row, col, color)
        except ValueError:
            raise Exception("illegal move in sgf file")
        positions.append(get_position(board))

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
    # Feature planes in Leela Zero:
    #
    #   1) Side to move stones at time T=0
    #   2) Side to move stones at time T=-1  (0 if T=0)
    #   ...
    #   8) Side to move stones at time T=-7  (0 if T<=6)
    #   9) Other side stones at time T=0
    #   10) Other side stones at time T=-1   (0 if T=0)
    #   ...
    #   16) Other side stones at time T=-7   (0 if T<=6)
    #   17) All 1 if black is to move, 0 otherwise
    #   18) All 1 if white is to move, 0 otherwise
    player_color = 'b'
    index = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
    for i, pos in enumerate(positions):
        plane = np.zeros((1, 18, 19, 19))

        back_move = min(i, 8)
        move_color = player_color
        for b in range(back_move):
            plane[0, index[2 * b], :, :] = get_plane(positions[i - b], move_color, player_color)
            move_color = opp_color(move_color)
            plane[0, index[2 * b + 1], :, :] = get_plane(positions[i - b], move_color, player_color)
            move_color = opp_color(move_color)

        if player_color == 'b':
            plane[0, 16, :, :] = np.ones((19, 19))
        elif player_color == 'w':
            plane[0, 17, :, :] = np.ones((19, 19))
        player_color = opp_color(player_color)

        if i == 0:
            feature_plane = plane
        else:
            feature_plane = np.concatenate((feature_plane, plane), axis=0)

    return feature_plane


def debug_show_feature(f_plane):
    for i in range(18):
        print("Feature", i)
        for j in range(19):
            for k in range(19):
                value = int(f_plane[i, j, k])
                if value > 0.5:
                    print("\033[1;41m%d\033[0m" % value, end=" ")
                else:
                    print(value, end=' ')
            print()
        print()


def debug_show_palne(plane):
    for j in range(19):
        for k in range(19):
            value = plane[j, k]
            if value > 0.5:
                print("\033[1;41m%2.1f\033[0m" % value, end=" ")
            else:
                print(value, end=' ')
        print()


def debug_show_position(position):
    for i in range(19):
        for j in range(19):
            point = position[i, j]
            if point > 0.5:
                print("\033[1;40m+\033[0m", end=" ")
            elif point < -0.5:
                print("\033[1;47m-\033[0m", end=" ")
            else:
                print(".", end=" ")
        print()
    print()


def main():
    positions = sgf_to_position("./sgf-GM-RG-0.sgf")
    f_plane = position_to_feature(positions)

    if float(tf.__version__[:3]) < 1.5:
        raise ValueError("tensorflow is too old, you need tf > 1.5")

    tfprocess = TFProcess()

    inputs = tf.placeholder(tf.float32, [None, 18, 19, 19])
    polocy, value = tfprocess.construct_net(inputs)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "leelaz-model-0")

        start = clock()
        for i in range(100):
            p, v = sess.run([polocy, value], feed_dict={inputs: f_plane})
            print(p)            # (-1, 362): move probabilities, 361 points + pass
            print((v + 1) / 2)  # (-1, 1):   this is winrate at each move
        finish = clock()
        print("Time per Kifu:", (finish - start) / 100)


if __name__ == "__main__":
    main()
