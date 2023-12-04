from tkinter import *
import Checkers
import Board
from Checkers import Bot

GREEN = "#9bdeac"
YELLOW = "#f7f5dd"
LIGHTBLUE = '#BBE1FA'
DARKBLUE = '#251D3A'
GREY = '#787A91'
FONT_NAME = "Courier"

root = Tk()
root.title('Checkers')
root.config(padx=50, pady=50, bg=GREEN)

title_label = Label(text = 'Шашки', fg='black', bg=GREEN, font=(FONT_NAME, 30, 'bold'))
title_label.grid(column=6, row=0, columnspan=1)

fields = []
for i in ['1', '2', '3', '4', '5', '6', '7', '8']:
    for j in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        fields.append(j + i)

columns_num = dict(zip(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 2, 3, 4, 5, 6, 7, 8]))
columns_num_reversed = dict(zip([1, 2, 3, 4, 5, 6, 7, 8],['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']))
history_list = []
moves_history_list = []
canvas = Canvas(root, width=500, height=500, bg=GREEN, highlightthickness=0)
canvas.grid(column=2, row=1, columnspan=8, rowspan=8)

def fill(a,b):
    if (a+b)%2 == 1:
        return 'white'
    elif (a+b)%2 == 0:
        return 'black'

def fill_piece(a):
    if a == 'white':
        return LIGHTBLUE
    elif a == 'black':
        return GREY

def setColor():
    canvas.dtag('pieceSelected', 'pieceSelected')
    canvas.itemconfigure('white', fill=LIGHTBLUE)
    canvas.itemconfigure('black', fill=GREY)
    canvas.addtag('pieceSelected', 'withtag', 'current')
    canvas.itemconfigure('pieceSelected', fill='pink')
    history_list.append([canvas.gettags(canvas.find_withtag('current')[0])[0],canvas.gettags(canvas.find_withtag('current')[0])[1],canvas.gettags(canvas.find_withtag('current')[0])[2]])

def move():
    i = canvas.gettags(canvas.find_withtag('current')[0])[0][1:3]
    moves_history_list.append(history_list[-1][0]+i)
    gui_move = None
    if not isinstance(match.opp, Bot.Bot) or f.whites_turn==1:
        gui_move = moves_history_list[-1]
    match.next_turn(gui_move)
    if isinstance(match.opp, Bot.Bot) and f.game_is_on==1:
        while f.whites_turn==0 and f.game_is_on==1:
            match.next_turn()
    canvas.delete('piece')
    canvas.delete('king')
    for i, j in match.board.field.items():
        k = i
        if j is not None:
            if j.colour == 'white' and isinstance(j, Board.Piece):
                id = canvas.create_oval(columns_num[i[0]] * 50 + 5, 450 - int(i[1]) * 50 + 5,
                                        (columns_num[i[0]] + 1) * 50 - 5, 500 - (int(i[1])) * 50 - 5, fill=LIGHTBLUE,
                                        outline='black', tags=(i, 'white', 'piece'))
                canvas.tag_bind(id, "<Button-1>", lambda x: setColor())
            elif j.colour == 'black' and isinstance(j, Board.Piece):
                id = canvas.create_oval(columns_num[i[0]] * 50 + 5, 450 - int(i[1]) * 50 + 5,
                                        (columns_num[i[0]] + 1) * 50 - 5, 500 - (int(i[1])) * 50 - 5, fill=GREY,
                                        outline='black', tags=(i, 'black', 'piece'))
                canvas.tag_bind(id, "<Button-1>", lambda x: setColor())
            if j.colour == 'white' and isinstance(j, Board.King):
                id = canvas.create_rectangle(columns_num[i[0]] * 50 + 5, 450 - int(i[1]) * 50 + 5,
                                        (columns_num[i[0]] + 1) * 50 - 5, 500 - (int(i[1])) * 50 - 5, fill=LIGHTBLUE,
                                        outline='black', tags=(i, 'white', 'king'))
                canvas.tag_bind(id, "<Button-1>", lambda x: setColor())
            elif j.colour == 'black' and isinstance(j, Board.King):
                id = canvas.create_rectangle(columns_num[i[0]] * 50 + 5, 450 - int(i[1]) * 50 + 5,
                                        (columns_num[i[0]] + 1) * 50 - 5, 500 - (int(i[1])) * 50 - 5, fill=GREY,
                                        outline='black', tags=(i, 'black', 'king'))
                canvas.tag_bind(id, "<Button-1>", lambda x: setColor())

    if f.game_is_on==0:
        title_label.config(text='Игра окончена')

for i in fields:
    j = i
    i = canvas.create_rectangle(columns_num[j[0]]*50, 450-int(j[1])*50, (columns_num[j[0]]+1)*50, 500-(int(j[1]))*50, fill=fill(int(j[1]), columns_num[j[0]]), outline='black', tags=('f{}'.format(i)))
    canvas.tag_bind(i, "<Button-1>", lambda x: move())

for i in range(8):
    canvas.create_text(40, 425-i*50, text = i+1, fill='black')
for i in range(8):
    canvas.create_text(i*50+75, 460, text = columns_num_reversed[i+1], fill='black')

f = Board.Field()
bot = Checkers.Bot.Bot(the_depth=3, the_board=f)

match = Checkers.Checkers(opp=bot, board=f, control='gui')

for i,j in match.board.field.items():
    k=i
    if j is not None:
        if j.colour == 'white':
            id = canvas.create_oval(columns_num[i[0]] * 50+5, 450 - int(i[1]) * 50+5, (columns_num[i[0]] + 1) * 50-5,500 - (int(i[1])) * 50-5, fill=LIGHTBLUE, outline='black', tags=(i, 'white', 'piece'))
            canvas.tag_bind(id, "<Button-1>", lambda x: setColor())
        elif j.colour == 'black':
            id = canvas.create_oval(columns_num[i[0]] * 50+5, 450 - int(i[1]) * 50+5, (columns_num[i[0]] + 1) * 50-5,500 - (int(i[1])) * 50-5, fill=GREY, outline='black', tags=(i, 'black', 'piece'))
            canvas.tag_bind(id, "<Button-1>", lambda x: setColor())

root.mainloop()


