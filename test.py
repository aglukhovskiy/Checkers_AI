import numpy as np

matrix =  np.zeros((8,8))


for i in matrix:
    print(i)

print(matrix)
# matrix[0][0] = 5
# print(matrix)

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
        # for i in range(8):
        #     row = []
        #     for j in range(8):
        #         row.append(0)
        #     self.matrix.append(row)
        # for i in self.matrix:
        #     print(i)

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

        for i in self.field:
            if isinstance(self.field[i], King):
                if self.field[i].colour == 'white':
                    self.matrix[8 - int(i[1])][self.columns_num[i[0]] - 1] = 2
                elif self.field[i].colour == 'black':
                    self.matrix[8 - int(i[1])][self.columns_num[i[0]] - 1] = -2
            elif isinstance(self.field[i], Piece):
                if self.field[i].colour == 'white':
                    self.matrix[8 - int(i[1])][self.columns_num[i[0]] - 1] = 1
                elif self.field[i].colour == 'black':
                    self.matrix[8 - int(i[1])][self.columns_num[i[0]] - 1] = -1


        # for i in self.matrix:
        #     print(i)
        print(self.matrix)
        self.opp_matrix = np.array([row[::-1] for row in self.matrix][::-1])

        print(self.opp_matrix)


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

f = Field()