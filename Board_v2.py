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
        self.game_is_on = 1
        self.game_result = []
        self.winner = 0

    def _create_board(self):
        # создаем пустую доску 8x8
        board = np.zeros((8, 8))

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
            return self.board[row][col]
        return 0

    def is_king(self, row, col):
        """Проверяет, является ли шашка дамкой"""
        piece = self.get_piece(row, col)
        return abs(piece) > 1

    def get_regular_moves_for_piece(self, row, col):
        """Получает обычные ходы для конкретной шашки"""
        moves = []
        piece = self.board[row][col]

        # Проверяем, является ли дамкой
        if abs(piece) > 1:
            # Дамка может ходить по диагонали на любое расстояние
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == 0:
                    moves.append((row, col, None, None, r, c))
                    r += dr
                    c += dc
        else:
            # Обычная шашка ходит только вперед
            directions = [(-1, -1), (-1, 1)] if piece > 0 else [(1, -1), (1, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == 0:
                    moves.append((row, col, None, None, r, c))

        return moves

    def get_regular_moves(self):
        """Получает обычные ходы для всех шашек текущего игрока"""
        moves = []

        for row, col in self.pieces:
            piece = self.board[row][col]
            if np.sign(piece) == self.current_player:
                moves.extend(self.get_regular_moves_for_piece(row, col))

        return moves

    def get_capture_moves(self, multicapture_piece=None, current_capture_series=None):
        """Получает все возможные пути взятия, включая множественные взятия"""

        # Результат - список цепочек взятий
        all_capture_sequences = []

        # Если текущая последовательность не передана, создаем пустую
        if current_capture_series is None:
            current_capture_series = []

        # Получаем все первичные взятия из текущей позиции
        first_captures = self.get_first_capture_moves(multicapture_piece)

        # Если нет возможных взятий, возвращаем текущую последовательность, если она не пуста
        if not first_captures:
            if current_capture_series:
                return [current_capture_series]
            return []

        # Для каждого возможного первого взятия
        for capture in first_captures:
            # Сохраняем текущее состояние доски и фигур
            original_board = self.board.copy()
            original_pieces = self.pieces.copy()

            # Выполняем взятие
            self.move_piece(capture, capture_move=True)

            # Создаем новую последовательность с текущим взятием
            new_sequence = current_capture_series + [capture]

            # Проверяем, может ли шашка после взятия сделать еще одно взятие
            next_pos = (capture[4], capture[5])
            multicapture_sequences = self.get_capture_moves(next_pos, new_sequence)

            # Если после этого взятия больше нет взятий, добавляем последовательность
            if not multicapture_sequences:
                all_capture_sequences.append(new_sequence)
            else:
                # Если есть дальнейшие взятия, добавляем все возможные последовательности
                all_capture_sequences.extend(multicapture_sequences)

            # Восстанавливаем состояние доски и фигур после проверки
            self.board = original_board.copy()
            self.pieces = original_pieces.copy()

        return all_capture_sequences

    def get_first_capture_moves(self, multicapture_piece=None):
        """Получает ходы со взятием для шашки или всех шашек текущего игрока"""
        first_capture_moves = []

        # Если указана конкретная шашка для проверки мультивзятия
        if multicapture_piece:
            pieces_to_check = [multicapture_piece]
        else:
            # Иначе проверяем все шашки текущего игрока
            pieces_to_check = [piece for piece in self.pieces
                               if np.sign(self.board[piece[0]][piece[1]]) == self.current_player]

        for piece in pieces_to_check:
            row, col = piece[0], piece[1]
            piece_value = self.board[row][col]

            # Проверяем, является ли дамкой
            if np.abs(piece_value) > 1:
                # Дамка может бить по диагонали на любое расстояние
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    enemy_found = False
                    enemy_pos = None

                    while 0 <= r < 8 and 0 <= c < 8:
                        if self.board[r][c] == 0:
                            if enemy_found:
                                first_capture_moves.append((row, col, enemy_pos[0], enemy_pos[1], r, c))
                            r += dr
                            c += dc
                            continue

                        # Если нашли шашку противника
                        if not enemy_found and np.sign(self.board[r][c]) == -np.sign(piece_value):
                            enemy_found = True
                            enemy_pos = (r, c)
                            r += dr
                            c += dc
                        else:
                            # Если нашли вторую шашку или свою шашку
                            break
            else:
                # Обычная шашка может бить в любом направлении
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] != 0 and np.sign(self.board[r][c]) == -np.sign(
                            piece_value):
                        capture_r, capture_c = r + dr, c + dc
                        if 0 <= capture_r < 8 and 0 <= capture_c < 8 and self.board[capture_r][capture_c] == 0:
                            first_capture_moves.append((row, col, r, c, capture_r, capture_c))

        return first_capture_moves

    def check_for_kings(self):
        """Проверяет, достигли ли какие-либо шашки противоположного края доски, чтобы стать дамками"""
        for col in range(8):
            # Белые шашки становятся дамками в верхнем ряду
            if self.board[0][col] == 1:
                self.board[0][col] = 3

            # Черные шашки становятся дамками в нижнем ряду
            if self.board[7][col] == -1:
                self.board[7][col] = -3

    def check_for_king(self, row, col):
        """Проверяет, достигла ли конкретная шашка противоположного края доски, чтобы стать дамкой"""
        # Белые шашки становятся дамками в верхнем ряду
        if row == 0 and self.board[row][col] == 1:
            self.board[row][col] = 3

        # Черные шашки становятся дамками в нижнем ряду
        if row == 7 and self.board[row][col] == -1:
            self.board[row][col] = -3

    def get_possible_moves(self):
        capture_moves = self.get_capture_moves()
        if capture_moves:
            return capture_moves
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
            self.pieces.remove((move[0], move[1]))
            self.pieces.remove((move[2], move[3]))
            self.pieces.add((move[4], move[5]))

            self.check_for_king(move[4], move[5])
        else:
            # Выполняем обычный ход
            self.board[move[4]][move[5]] = self.board[move[0]][move[1]]
            self.board[move[0]][move[1]] = 0
            # Обновляем позиции
            self.pieces.remove((move[0], move[1]))
            self.pieces.add((move[4], move[5]))
            self.check_for_king(move[4], move[5])

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
                    k_b += 1
                else:
                    p_b += 1

        return [p_w, p_b, k_w, k_b]

    def check_winner(self):
        """Проверяет, есть ли победитель"""
        pieces = self.get_number_of_pieces_and_kings()

        if pieces[0] + pieces[2] == 0:
            self.game_is_on = 0
            self.game_result = pieces
            self.winner = -1
            return self.winner
        elif pieces[1] + pieces[3] == 0:
            self.game_is_on = 0
            self.game_result = pieces
            self.winner = 1
            return self.winner

        # Проверка на наличие возможных ходов
        if not self.get_possible_moves():
            self.game_is_on = 0
            self.game_result = pieces
            if pieces[0] + pieces[2] > pieces[1] + pieces[3]:
                self.winner = 1
            elif pieces[0] + pieces[2] < pieces[1] + pieces[3]:
                self.winner = -1
            return self.winner

        return None

    def next_turn(self, move_list):
        for move in move_list:
            print('starting move - ', move)
            self.move_piece(move, capture_move=(move[2] is not None))
        self.current_player = -self.current_player
        return self.check_winner()

    def reset_game(self):
        """Сбрасывает игру в начальное состояние"""
        self.pieces = set()
        self.kings = set()
        self.board = self._create_board()
        self.current_player = 1
        self.selected_piece = None
        self.possible_moves = []
        self.capture_moves = []
        self.game_is_on = 1
        self.game_result = []
        self.winner = 0

    def available_moves(self):
        """Возвращает доступные ходы для совместимости с TenPlaneEncoder"""
        all_moves = []
        all_capture_sequences = self.get_capture_moves()

        # Если есть взятия, они обязательны
        if all_capture_sequences:
            # Берем первые ходы из каждой последовательности
            for sequence in all_capture_sequences:
                if sequence:
                    all_moves.append(sequence[0])
        else:
            all_moves = self.get_regular_moves()

        # Создаем структуру, совместимую с форматом TenPlaneEncoder
        threatened_pieces = {}
        taker_pieces = {}

        # Извлекаем информацию о шашках под боем и шашках, которые могут бить
        for move in all_moves:
            if move[2] is not None:  # Это взятие
                # Шашка под боем
                threatened_pieces[f'{move[2]},{move[3]}'] = (move[2], move[3])
                # Шашка, которая может бить
                taker_pieces[f'{move[0]},{move[1]}'] = (move[0], move[1])

        return [all_moves, threatened_pieces, taker_pieces]


    def compute_results(self):
        """Вычисляет результат игры и разницу в очках"""
        pieces = self.get_number_of_pieces_and_kings()

        # Очки: обычная шашка - 1, дамка - 3
        white_score = pieces[0] + 3 * pieces[2]
        black_score = pieces[1] + 3 * pieces[3]

        margin = white_score - black_score

        if self.winner == 1:
            return 1, margin
        elif self.winner == -1:
            return 0, margin
        elif self.winner == 0.5:
            return 0.5, margin

        # Если нет победителя, определяем по количеству оставшихся шашек
        if white_score > black_score:
            return 1, margin
        elif white_score < black_score:
            return 0, margin
        else:
            return 0.5, margin  # Ничья

# game = CheckersGame()
# print(game.get_regular_moves())

# print(game.pieces)
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

# for i in range(25):
#     print('pos moves - ', game.get_possible_moves())
#     print(game.board)
#     move = game.get_possible_moves()[0]
#     print('move - ', move)
#     game.next_turn(move)


# print('pos moves - ', game.get_possible_moves())
# print(game.board)
# move = game.get_possible_moves()[0]
# print('move - ', move)
# game.next_turn(move)
#
# print('pos moves - ', game.get_possible_moves())
# print(game.board)
# move = game.get_possible_moves()[0]
# print('move - ', move)
# game.next_turn(move)
#
# print('pos moves - ', game.get_possible_moves())
# print(game.board)
# move = game.get_possible_moves()[0]
# print('move - ', move)
# game.next_turn(move)
#
# print('pos moves - ', game.get_possible_moves())
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