import copy
import numpy as np


class Field:

    columns_num = dict(zip(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 2, 3, 4, 5, 6, 7, 8]))

    def __init__(self, preset=None, whites_turn=1):
        self.whites_turn = whites_turn
        self.white_num_to_colour = {1: 'white', 0:'black'}
        self.game_is_on = 1
        self.bot_test = 0
        self.n = 0 # вроде можно удалить
        self.history = []
        self.multiple_jumping_piece = []
        self.fields = []
        self.parent = None

        self.matrix = np.zeros((8,8))

        if preset is not None:
            self.field = preset
        else:
            for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
                for j in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                    self.fields.append(j + i)
            self.figures = [None] * 64
            self.field = {k: v for k, v in zip(self.fields, self.figures)}
            self.field['a1'] = Piece(colour='white')
            self.field['a3'] = Piece(colour='white')
            self.field['b2'] = Piece(colour='white')
            self.field['c1'] = Piece(colour='white')
            self.field['c3'] = Piece(colour='white')
            self.field['d2'] = Piece(colour='white')
            self.field['e1'] = Piece(colour='white')
            self.field['e3'] = Piece(colour='white')
            self.field['f2'] = Piece(colour='white')
            self.field['g1'] = Piece(colour='white')
            self.field['g3'] = Piece(colour='white')
            self.field['h2'] = Piece(colour='white')
            self.field['a7'] = Piece(colour='black')
            self.field['b8'] = Piece(colour='black')
            self.field['b6'] = Piece(colour='black')
            self.field['c7'] = Piece(colour='black')
            self.field['d8'] = Piece(colour='black')
            self.field['d6'] = Piece(colour='black')
            self.field['e7'] = Piece(colour='black')
            self.field['f8'] = Piece(colour='black')
            self.field['f6'] = Piece(colour='black')
            self.field['g7'] = Piece(colour='black')
            self.field['h8'] = Piece(colour='black')
            self.field['h6'] = Piece(colour='black')

        # for i in self.field:
        #     self.matrix[8-int(i[1])][self.columns_num[i[0]]-1] = self.field[i]

    def get_number_of_pieces_and_kings(self):
        p_w = 0
        p_b = 0
        k_w = 0
        k_b = 0

        for i, j in self.field.items():
            if isinstance(j, King) and j.colour == 'white':
                k_w += 1
            elif isinstance(j, Piece) and j.colour == 'white':
                p_w += 1
            elif isinstance(j, King) and j.colour == 'black':
                k_b += 1
            elif isinstance(j, Piece) and j.colour == 'black':
                p_b += 1
        return [p_w, p_b, k_w, k_b]

    def compute_results(self):
        pieces = self.get_number_of_pieces_and_kings()
        res_dif = pieces[0]+pieces[2]-pieces[1]-pieces[3]
        if res_dif>0:
            return 1, res_dif
        elif res_dif<0:
            return -1, res_dif
        else:
            return 0, res_dif

    def available_moves(self, colour=None):

        self.pieces_list = []
        self.available_moves_list = []
        self.pieces_to_capture_list = []
        self.available_moves_by_capture_dict = {}
        self.next_pos_after_capture_dict = {}
        self.blocking_pieces_list = []
        self.blocked_moves_list = []
        self.pieces_to_capture_by_king_list = []
        if colour==None:
            if self.whites_turn:
                colour = 'white'
            else:
                colour = 'black'

        # шашки на поле
        for i, j in self.field.items():
            if getattr(j, 'colour', None) == colour:
                self.pieces_list.append([i,j])
        # возможные ходы
        for i, j in self.field.items():
            for k,m in self.pieces_list:
                if 'piece' in m.__str__() and abs(self.columns_num[i[0]] - self.columns_num[k[0]]) == 1 and j is None and (int(i[1]) == int(k[1]) + 1 and colour == 'white' or int(i[1]) == int(k[1]) - 1 and colour == 'black'):
                    if k in self.multiple_jumping_piece:
                        self.available_moves_list.append(k+i)
                    elif len(self.multiple_jumping_piece) == 0:
                        self.available_moves_list.append(k + i)
        #Шашки, доступные к взятию
        for i, j in self.field.items():
            for k,m in self.pieces_list:
                if abs(self.columns_num[i[0]] - self.columns_num[k[0]]) == 1  and j is not None and abs(getattr(m, 'white', None)-getattr(j, 'white', None)) == 1 and abs(int(i[1]) - int(k[1])) == 1:
                    self.pieces_to_capture_list.append(k+i)
        #Доступные ходы со взятием
        for i, j in self.field.items():
            for k, m in self.pieces_list:
                for n in self.pieces_to_capture_list:
                    if k == n[0:2]:
                        if 'piece' in m.__str__() and abs(self.columns_num[i[0]] - self.columns_num[k[0]]) == 2 and abs(self.columns_num[i[0]] - self.columns_num[n[2]]) == 1 and j is None:
                            if abs(int(i[1]) - int(k[1])) == 2 and abs(int(i[1]) - int(n[3])) == 1:
                                self.available_moves_by_capture_dict[k + i] = n[2:4]
                                self.available_moves_list.append(k + i)
                                self.next_pos_after_capture_dict[k + i] = k
        #отдельно для дамок
        for i, j in self.field.items():
            for k,m in self.pieces_list:
                if isinstance(m, King):
                    if j is not None and abs(self.columns_num[i[0]] - self.columns_num[k[0]]) == abs(int(i[1]) - int(k[1])) != 0 and abs(getattr(m, 'white', None)-getattr(j, 'white', None)) == 0:
                        self.blocking_pieces_list.append(k+i)
        #парные шашки
        for i, j in self.field.items():
            for k,m in self.pieces_list:
                for x,y in self.field.items():
                    if isinstance(m, King):
                        if j is not None and abs(self.columns_num[i[0]] - self.columns_num[k[0]]) == abs(int(i[1]) - int(k[1])) != 0:
                            if y is not None and abs(self.columns_num[x[0]] - self.columns_num[k[0]]) == abs(int(x[1]) - int(k[1])) != 0 \
                                    and abs(int(x[1]) - int(i[1])) == abs(self.columns_num[x[0]] - self.columns_num[i[0]]) >= 1 \
                                    and abs(int(i[1]) - int(k[1])) > abs(int(i[1]) - int(x[1]))\
                                    and abs(int(x[1]) - int(k[1])) < abs(int(i[1]) - int(k[1])):
                                self.blocking_pieces_list.append(k+i)
                                self.blocking_pieces_list = list(set(self.blocking_pieces_list))

        # блокированные ходы
        for i, j in self.field.items():
            for k,m in self.pieces_list:
                for l in self.blocking_pieces_list:
                    if isinstance(m, King) and k == l[0:2]:
                        if j is None and abs(self.columns_num[i[0]] - self.columns_num[k[0]]) == abs(int(i[1]) - int(k[1])) and abs(int(i[1]) - int(k[1])) > 1:
                            # if (self.columns_num[i[0]] > self.columns_num[n[2]] or self.columns_num[i[0]] < self.columns_num[n[2]]) and abs(int(i[1]) - int(k[1])) > abs(int(n[3]) - int(k[1])):
                            if self.columns_num[i[0]] > self.columns_num[l[2]] and int(i[1]) > int(l[3]) and self.columns_num[l[2]] > self.columns_num[k[0]] and int(l[3]) > int(k[1]) \
                                    or self.columns_num[i[0]] > self.columns_num[l[2]] and int(i[1]) < int(l[3]) and self.columns_num[l[2]] > self.columns_num[k[0]] and int(l[3]) < int(k[1]) \
                                    or self.columns_num[i[0]] < self.columns_num[l[2]] and int(i[1]) > int(l[3]) and self.columns_num[l[2]] < self.columns_num[k[0]] and int(l[3]) > int(k[1]) \
                                    or self.columns_num[i[0]] < self.columns_num[l[2]] and int(i[1]) < int(l[3]) and self.columns_num[l[2]] < self.columns_num[k[0]] and int(l[3]) < int(k[1]):
                                self.blocked_moves_list.append(k + i)

        # ходы дамок без блокированных
        for i, j in self.field.items():
            for k,m in self.pieces_list:
                if isinstance(m, King) and k+i not in self.blocked_moves_list:
                    if j is None and abs(self.columns_num[i[0]] - self.columns_num[k[0]]) == abs(int(i[1]) - int(k[1])) and abs(int(i[1]) - int(k[1])) >= 1:
                        if k in self.multiple_jumping_piece:
                            self.available_moves_list.append(k + i)
                        elif len(self.multiple_jumping_piece) == 0:
                            self.available_moves_list.append(k + i)

        # шашки, атакуемые дамкой
        for i, j in self.field.items():
            for k,m in self.pieces_list:
                if isinstance(m, King):
                    if j is not None and abs(self.columns_num[i[0]] - self.columns_num[k[0]]) == abs(int(i[1]) - int(k[1])) != 0 and abs(getattr(m, 'white', None)-getattr(j, 'white', None)) == 1:
                        self.pieces_to_capture_by_king_list.append(k+i)

        #Доступные ходы со взятием дамкой

        for i, j in self.field.items():
            for k, m in self.pieces_list:
                for l in self.pieces_to_capture_by_king_list:
                    if k == l[0:2] and isinstance(m, King) and k+i not in self.blocked_moves_list and k+i in self.available_moves_list:
                        if j is None and abs(self.columns_num[i[0]] - self.columns_num[k[0]]) >= 2 and abs(self.columns_num[i[0]] - self.columns_num[l[2]]) >= 1 \
                                and abs(self.columns_num[i[0]] - self.columns_num[k[0]]) > abs(self.columns_num[l[2]] - self.columns_num[k[0]]) \
                                and abs(self.columns_num[i[0]] - self.columns_num[k[0]]) > abs(self.columns_num[i[0]] - self.columns_num[l[2]]):
                            if abs(int(i[1]) - int(k[1])) > abs(int(i[1]) - int(l[3])) >= 1:
                                self.available_moves_by_capture_dict[k + i] = l[2:4]
                                # self.available_moves_list.append(k + i)
                                self.next_pos_after_capture_dict[k + i] = k
        if len(self.available_moves_by_capture_dict) > 0:
            self.available_moves_list = list(self.available_moves_by_capture_dict.keys())

        return self.available_moves_list, self.available_moves_by_capture_dict, self.next_pos_after_capture_dict

    def move(self, opp, control='command', move=None):
        colour = self.white_num_to_colour[self.whites_turn]
        if len(self.available_moves()[0])==0: # If no moves available - dead end
            self.game_is_on = 0
            return 'game over'
        if control=='gui':
            if str(opp) == 'Bot class' and colour==opp.colour: #Тут будет ошибка вызова метода у None
                self.move_text  = opp.get_next_move()
            else:
                self.move_text = move
        elif control=='command':
            if move is not None:
                self.move_text = move
            elif str(opp) == 'Bot class' and colour==opp.colour:
                # self.move_text = random.choice(self.available_moves(colour=colour)[0])
                self.move_text = opp.get_next_move()
            else:
                self.move_text = input('{}s move: '.format(colour)) # добавить проверку соответствия формата ввода
        if self.move_text == 'end':
            self.game_is_on = 0
            return 'game over'
        elif self.move_text in self.available_moves()[0] and len(self.multiple_jumping_piece) == 0:
            if self.move_text in self.available_moves()[1]: # Если среди ходов с взятием - удалить шашку
                self.history.append([self.n, move, 'success', self.available_moves()[1][self.move_text], colour])
                self.field[self.available_moves()[1][self.move_text]] = None
                self.field[self.move_text[2:4]] = self.field[self.move_text[0:2]]
                self.field[self.move_text[0:2]] = None
                if self.move_text[3] == '8' and colour == 'white' or self.move_text[3] == '1' and colour == 'black':
                    self.field[self.move_text[2:4]] = King(colour=colour)
                if self.move_text[2:4] in self.available_moves()[2].values():
                    self.multiple_jumping_piece.append(self.move_text[2:4])
                    self.move(opp=opp, move= self.available_moves()[0][0]) # проверить кейсы
                    # self.next_turn(move_list)
                else:
                    self.whites_turn = 1 - self.whites_turn
                    self.multiple_jumping_piece = []
            else:
                self.history.append([self.n, move, 'success', '', colour])
                self.field[self.move_text[2:4]] = self.field[self.move_text[0:2]]
                self.field[self.move_text[0:2]] = None
                if self.move_text[3] == '8' and colour == 'white' or self.move_text[3] == '1' and colour == 'black':
                    self.field[self.move_text[2:4]] = King(colour=colour)
                self.whites_turn = 1 - self.whites_turn # меняем игрока
                self.multiple_jumping_piece = []

        elif self.move_text in self.available_moves()[0] and self.move_text[0:2] in self.multiple_jumping_piece:
            if self.move_text in self.available_moves()[1]: # Если среди ходов с взятием - удалить шашку
                self.history.append([self.n, move, 'success', self.available_moves()[1][self.move_text], colour])
                self.field[self.available_moves()[1][self.move_text]] = None
            else:
                self.history.append([self.n, move, 'success', '', colour])

            self.field[self.move_text[2:4]] = self.field[self.move_text[0:2]]
            self.field[self.move_text[0:2]] = None
            if self.move_text[3] == '8' and colour == 'white' or self.move_text[3] == '1' and colour == 'black':
                self.field[self.move_text[2:4]] = King(colour=colour)

            if self.move_text[2:4] in self.available_moves()[2].values():
                self.multiple_jumping_piece = []
                self.multiple_jumping_piece.append(self.move_text[2:4])
                # if self.bot_test == 1: # костыль с рандомным взятием
                # self.move(colour=colour, opp=opp, control=control, move=self.available_moves(self.white_num_to_colour[self.whites_turn])[0][0])
                # self.move(colour=colour, opp=opp, move= random.choice(self.available_moves(colour=colour, opp=opp)[0][0]))
                # self.next_turn(move_list)

            else:
                self.whites_turn = 1 - self.whites_turn # меняем игрока
                self.multiple_jumping_piece = []
                # self.next_turn(move_list)
        else:
            # print('Move is not available')
            self.history.append([self.n, move, 'failure', ''])
            # self.next_turn(move_list)
        if len([i for i,j in self.field.items() if 'white' in j.__str__()]) == 0 or len([i for i,j in self.field.items() if 'black' in j.__str__()]) == 0:
            self.game_is_on = 0
            return 'game over'

    def get_potential_spots_from_moves(self, moves, opp=None):
        if moves is None:
            return self.field
        answer = []

        for move in moves:
            original_spots = copy.deepcopy(self.field)
            original_turn = copy.deepcopy(self.whites_turn)
            original_game_is_on = copy.deepcopy(self.game_is_on)
            self.move(opp=opp, move=move)
            answer.append(self.field)
            self.field = original_spots
            self.whites_turn = original_turn
            self.game_is_on = original_game_is_on
        return answer

class Piece:
    def __init__(self, colour):
        self.colour = colour
        if colour == 'white':
            self.white=1
        elif  colour == 'black':
            self.white=0
    def __str__(self):
        return '{} piece'.format(self.colour)

class King(Piece):
    def __init__(self,colour):
        super().__init__(colour=colour)
    def __str__(self):
        return '{} king'.format(self.colour)

# NN