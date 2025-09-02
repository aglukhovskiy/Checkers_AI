from flask import Flask, request, jsonify
from flask_cors import CORS
from src.core.checkers import Checkers
from src.core.board import Field

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Инициализация игры
game_field = Field()
bot = None  # Будет инициализирован позже
game = Checkers(board=game_field, opp=bot, control='api')

@app.route('/new_game', methods=['POST'])
def new_game():
    global game_field, game
    game_field = Field()
    game = Checkers(board=game_field, opp=bot, control='api')
    return jsonify({'status': 'ok'})

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    move = data.get('move')
    
    result = game.next_turn(move=move)
    board_state = {
        'field': [[0]*8 for _ in range(8)],  # Заполнить реальным состоянием
        'current_player': 'white' if game.board.whites_turn else 'black',
        'game_over': game.board.game_is_on == 0
    }
    
    return jsonify({
        'status': 'ok',
        'board': board_state,
        'result': result
    })

@app.route('/get_state', methods=['GET'])
def get_state():
    board_state = {
        'field': [[0]*8 for _ in range(8)],
        'current_player': 'white' if game.board.whites_turn else 'black',
        'game_over': game.board.game_is_on == 0
    }
    
    # Заполняем поле из game.board.field
    for pos, piece in game.board.field.items():
        if piece:
            col = ord(pos[0]) - ord('a')
            row = 8 - int(pos[1])
            value = 1 if piece.colour == 'white' else -1
            if isinstance(piece, King):
                value *= 2
            board_state['field'][row][col] = value
    
    return jsonify({
        'status': 'ok',
        'board': board_state
    })

@app.route('/bot_move', methods=['POST'])
def bot_move():
    if game.board.whites_turn:
        return jsonify({'status': 'error', 'message': 'Not bot turn'})
    
    # Получаем ход от бота
    move = game.next_turn()
    
    return jsonify({
        'status': 'ok',
        'move': move
    })

if __name__ == '__main__':
    app.run(debug=True)