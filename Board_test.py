import numpy as np
from copy import copy

class CheckersGame:

    def __init__(self):
        self.pieces = set()
        self.kings = set()  # хранение координат дамок
        self.board = self._create_board()
        self.current_player = 1
        self.selected_piece = None
        self.possible_moves = []
        self.capture_moves = []
        self.game_is_on=0
        self.game_result = []
        self.winner = 0

    def _create_board(self):
        # создаем пустую доску 8x8
        board =  np.zeros((8,8))

        # Расставляем начальные позиции белых шашек
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:  # только на темных клетках
                    board[row][col] = 1
                    self.pieces.add((row, col))

        # Расставляем начальные позиции черных шашек
        for row in range(0, 3):
            for col in range(8):
                if (row + col) % 2 == 1:  # только на темных клетках
                    board[row][col] = -1
                    self.pieces.add((row, col))

        return board

    def get_piece(self, row, col):
        """Возвращает шашку на указанной позиции"""
        if 0 <= row < 8 and 0 <= col < 8:
            if self.board[row][col]!=0:
                return self.board[row][col]
        return None

    def is_king(self, row, col):
        """Проверяет, является ли шашка дамкой"""
        return (row, col) in self.kings

    def select_piece(self, row, col):
        """Выбирает шашку для движения"""
        piece = self.get_piece(row, col)

        if piece == self.current_player:
            self.selected_piece = (row, col)
            # Вычисляем возможные ходы для выбранной шашки
            self.capture_moves = self.get_capture_moves(row, col)

            # Если есть обязательные взятия, другие ходы не разрешены
            if not self.capture_moves:
                self.possible_moves = self.get_regular_moves(row, col)
            else:
                self.possible_moves = []

            return True
        return False


    def get_regular_moves(self):
        """Получает обычные ходы для шашки"""
        moves = []

        for i in self.pieces:
            row, col = i[0], i[1]
            piece = self.board[row][col]
            if np.sign(piece)==self.current_player:
                # Проверяем, является ли дамкой
                if np.abs(piece)>1:
                    # Дамка может ходить по диагонали на любое расстояние
                    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == 0:
                            moves.append([(row,col, None, None, r, c)])
                            r += dr
                            c += dc
                else:
                    # Обычная шашка ходит только вперед
                    directions = [(-1, -1), (-1, 1)] if piece>0 else [(1, -1), (1, 1)]
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        if 0 <= r < 8 and 0 <= c < 8 and  self.board[r][c]==0:
                            moves.append([(row,col, None, None, r, c)])
        return moves

    def get_capture_moves(self, multicapture_piece=None, capture_moves=[]):
        capture_moves = capture_moves
        original_board = copy(self.board)
        original_pieces = copy(self.pieces)

        for i in self.get_first_capture_moves(multicapture_piece = multicapture_piece):
            current_capture_series = []
            self.move_piece(i, capture_move=True)
            current_capture_series.append(i)

            multicapture_moves = self.get_first_capture_moves(multicapture_piece=(i[4], i[5]))
            while multicapture_moves:
                self.get_capture_moves(multicapture_piece=(i[4], i[5]), capture_moves=current_capture_series)

            capture_moves.append(current_capture_series)
            self.board = original_board
            self.pieces = original_pieces

        return capture_moves

    def get_first_capture_moves(self, multicapture_piece=None):
        """Получает ходы со взятием для шашки"""
        first_capture_moves = []

        for i in self.pieces:
            if multicapture_piece:
                if i!=multicapture_piece:
                    continue
            row, col = i[0], i[1]
            piece = self.board[row][col]

            if np.sign(piece)==self.current_player:

                # Проверяем, является ли дамкой
                if np.abs(piece)>1:
                    # Дамка может бить по диагонали на любое расстояние
                    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        while 0 <= r < 8 and 0 <= c < 8:
                            if self.board[r][c] == 0:
                                r += dr
                                c += dc
                                continue

                            # Найдена потенциальная шашка для взятия
                            if np.sign(self.board[r][c]) != np.sign(piece):
                                # Проверяем, есть ли пустая клетка за ней
                                capture_r, capture_c = r + dr, c + dc
                                while 0 <= capture_r < 8 and 0 <= capture_c < 8:
                                    if self.board[capture_r][capture_c] == 0:
                                        first_capture_moves.append((row, col, r, c, capture_r, capture_c))  # шашка,взятие, ход
                                        capture_r += dr
                                        capture_c += dc
                                    else:
                                        break
                            break
                else:
                    # Обычная шашка может бить в любом направлении
                    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    for dr, dc in directions:
                        r, c = row + dr, col + dc
                        if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c]!=0 and np.sign(self.board[r][c]) != np.sign(piece):
                            capture_r, capture_c = r + dr, c + dc
                            if 0 <= capture_r < 8 and 0 <= capture_c < 8 and self.board[capture_r][capture_c] == 0:
                                first_capture_moves.append((row, col, r, c, capture_r, capture_c))  # шашка,взятие, ход

        return first_capture_moves

    def check_for_kings(self):
        """Проверяет, достигли ли какие-либо шашки противоположного края доски, чтобы стать дамками"""
        for col in range(8):
            # Белые шашки становятся дамками в верхнем ряду
            if self.board[0][col] == 1:
                self.board[0][col] = 3

            # Черные шашки становятся дамками в нижнем ряду
            if self.board[7][col]==-1:
                self.board[7][col] = -3

    def check_for_king(self, col):
        """Проверяет, достигла ли конкретная шашка противоположного края доски, чтобы стать дамкой"""

        # Белые шашки становятся дамками в верхнем ряду
        if self.board[0][col] == 1:
            self.board[0][col] = 3

        # Черные шашки становятся дамками в нижнем ряду
        if self.board[7][col]==-1:
            self.board[7][col] = -3

    def get_possible_moves(self):
        if self.get_capture_moves():
            return self.get_capture_moves()
        else:
            return self.get_regular_moves()

    def move_piece(self, move, capture_move=False):

        if capture_move:
            # Выполняем взятие
            self.board[move[4]][move[5]] = self.board[move[0]][move[1]]
            self.board[move[0]][move[1]] = 0
            # Удаляем взятую шашку
            self.board[move[2]][move[3]] = 0
            # Обновляем позиции
            self.pieces.remove((move[0],move[1]))
            self.pieces.remove((move[2],move[3]))
            self.pieces.add((move[4],move[5]))

            self.check_for_king(move[5])
        else:
            # Выполняем обычный ход
            self.board[move[4]][move[5]] = self.board[move[0]][move[1]]
            self.board[move[0]][move[1]] = 0
            # Обновляем позиции
            self.pieces.remove((move[0],move[1]))
            self.pieces.add((move[4],move[5]))
            self.check_for_king(move[5])

    def get_number_of_pieces_and_kings(self):
        p_w = 0
        p_b = 0
        k_w = 0
        k_b = 0

        for i in self.pieces:
            piece = self.board[i[0]][i[1]]
            if piece > 0:
                if piece > 1:
                    k_w += 1
                else:
                    p_w += 1

            elif piece < 0:
                if piece < -1:
                    k_w += 1
                else:
                    p_w += 1

        return [p_w, p_b, k_w, k_b]

    def check_winner(self):
        """Проверяет, есть ли победитель"""
        pieces = self.get_number_of_pieces_and_kings()

        if pieces[0]+pieces[2]==0:
            self.game_is_on=0
            self.game_result = pieces
            self.winner = -1
        elif pieces[1]+pieces[3]==0:
            self.game_is_on=0
            self.game_result = pieces
            self.winner = 1

        # Проверка на наличие возможных ходов
        if not self.get_possible_moves():
            self.game_is_on=0
            self.game_result = pieces
            if pieces[0]+pieces[2]>pieces[1]+pieces[3]:
                self.winner=1
            elif pieces[0]+pieces[2]>pieces[1]+pieces[3]:
                self.winner=-1
        return None

    def next_turn(self, move_list):
        for move in move_list:
            self.move_piece(move)
        self.current_player = 0-self.current_player
        self.check_winner()

    def reset_game(self):
        """Сбрасывает игру в начальное состояние"""
        self.board = self._create_board()
        self.current_player = 'white'
        self.selected_piece = None
        self.possible_moves = []
        self.capture_moves = []
        self.kings = set()

game = CheckersGame()
# print(game.get_regular_moves())

print(game.pieces)
# print((5, 4) in game.pieces)
# game.pieces.remove((5,4))
# print((5, 4) in game.pieces)

# print(game.board)
# print(game.get_possible_moves())
# move = game.get_possible_moves()[0]
# print(move)
# print(move[0])
# game.next_turn(move)
# print(game.board)
# print(game.get_possible_moves())

# for i in range(10):
#     print('pos moves - ', game.get_possible_moves())
#     print(game.board)
#     move = game.get_possible_moves()[0]
#     print('move - ', move)
#     game.next_turn(move)

print('pos moves - ', game.get_possible_moves())
print(game.board)
move = game.get_possible_moves()[0]
print('move - ', move)
game.next_turn(move)

print('pos moves - ', game.get_possible_moves())
print(game.board)
move = game.get_possible_moves()[0]
print('move - ', move)
game.next_turn(move)

print('pos moves - ', game.get_possible_moves())
print(game.board)
move = game.get_possible_moves()[0]
print('move - ', move)
game.next_turn(move)

print('pos moves - ', game.get_possible_moves())
# print(game.board)
# moves = game.get_possible_moves()[0]
# print('move - ', moves)
# print(game.pieces)
# for move in moves:
#     game.move_piece(move)

# game.next_turn(move)

# for i in move:

# game.move_piece(move[0])



# for i in game.get_regular_moves():
#     game.board[i[0]][i[1]]=6
#
#
# print('------------')
#
# print(game.board)

# d = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
# for dr, dc in d:
#     print(dr,dc, dr+1, dc+1)

# print(np.sign(-3)==-1)