from flask import Flask, request, jsonify
from flask_cors import CORS
from src.core.checkers import Checkers
from src.core.board import Field, King, Piece
from src.core.board_v2 import CheckersGame

app = Flask(__name__)
CORS(app, resources={
    r"/*": {"origins": ["http://localhost:8000", "http://127.0.0.1:8000", "file://"]}
})

# Инициализация игры
game_field = Field()
bot = None  # Будет инициализирован позже
game = Checkers(board=game_field, opp=bot, control='api')

# Новая версия игры
game_v2 = CheckersGame()

@app.route('/new_game', methods=['POST'])
def new_game():
    global game_field, game, game_v2
    game_field = Field()
    game = Checkers(board=game_field, opp=bot, control='api')
    game_v2 = CheckersGame()
    return jsonify({'status': 'ok'})

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    move = data.get('move')
    
    result = game.next_turn(move=move)
    
    # Синхронизируем состояние с game_v2
    sync_games()
    
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
        'board': board_state,
        'result': result
    })

def sync_games():
    """Синхронизирует состояние game_v2 с основным game"""
    # Очищаем game_v2
    game_v2.board = np.zeros((8, 8))
    game_v2.pieces = set()
    
    # Копируем состояние из game в game_v2
    for pos, piece in game.board.field.items():
        if piece:
            col = ord(pos[0]) - ord('a')
            row = 8 - int(pos[1])  # Конвертируем в формат game_v2 (0-7)
            
            # Устанавливаем значение в game_v2
            value = 1 if piece.colour == 'white' else -1
            if isinstance(piece, King):
                value *= 2
            game_v2.board[row][col] = value
            game_v2.pieces.add((row, col))
    
    # Синхронизируем текущего игрока
    game_v2.current_player = 1 if game.board.whites_turn else -1

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

@app.route('/get_possible_moves', methods=['POST'])
def get_possible_moves():
    data = request.json
    row = data.get('row')
    col = data.get('col')
    
    # Конвертируем координаты из формата интерфейса в формат board_v2
    # В интерфейсе row 0 - верх доски, в board_v2 row 0 - низ доски
    board_row = 7 - row
    print(f"API: Интерфейс запросил ходы для row={row}, col={col} -> board_v2 row={board_row}, col={col}")
    
    # Проверяем наличие обязательных взятий
    all_captures = game_v2.get_capture_moves()
    print(f"API: Всего взятий найдено: {len(all_captures)}")
    
    if all_captures:
        # Есть обязательные взятия - проверяем, может ли эта шашка брать
        piece_captures = []
        for capture_seq in all_captures:
            if capture_seq and capture_seq[0][0] == board_row and capture_seq[0][1] == col:
                piece_captures.append(capture_seq[0])
        
        if piece_captures:
            # Эта шашка может совершить взятие - возвращаем только взятия
            moves = []
            for move in piece_captures:
                # move format: (from_row, from_col, capture_row, capture_col, to_row, to_col)
                interface_to_row = 7 - move[4]  # Конвертируем обратно для интерфейса
                moves.append([interface_to_row, move[5]])
                print(f"API: Взятие конвертировано: board_v2 {move[4]},{move[5]} -> интерфейс {interface_to_row},{move[5]}")
            
            print(f"API: Возвращаем {len(moves)} взятий")
            return jsonify({
                'status': 'ok',
                'moves': moves,
                'capture_required': True
            })
        else:
            # Эта шашка не может совершать взятие, но другие могут - обычные ходы заблокированы
            print("API: У этой шашки нет взятий, но другие могут - блокируем обычные ходы")
            return jsonify({
                'status': 'ok',
                'moves': [],
                'capture_required': True
            })
    else:
        # Нет обязательных взятий, возвращаем обычные ходы
        piece_moves = game_v2.get_regular_moves_for_piece(board_row, col)
        
        # Конвертируем ходы обратно в формат интерфейса
        moves = []
        for move in piece_moves:
            # move format: (from_row, from_col, None, None, to_row, to_col)
            interface_to_row = 7 - move[4]  # Конвертируем обратно для интерфейса
            moves.append([interface_to_row, move[5]])
            print(f"API: Обычный ход конвертирован: board_v2 {move[4]},{move[5]} -> интерфейс {interface_to_row},{move[5]}")
        
        print(f"API: Возвращаем {len(moves)} обычных ходов")
        return jsonify({
            'status': 'ok',
            'moves': moves,
            'capture_required': False
        })

@app.route('/bot_move', methods=['POST'])
def bot_move():
    if game.board.whites_turn:
        return jsonify({'status': 'error', 'message': 'Not bot turn'})
    
    # Получаем ход от бота
    try:
        # Для API режима используем доступные ходы вместо input()
        available_moves = game.board.available_moves()[0]
        if available_moves:
            # Берем первый доступный ход (можно заменить на более умную логику)
            move = available_moves[0]
            result = game.next_turn(move=move)
            
            # Обновляем состояние доски для ответа
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
                'move': move,
                'board': board_state
            })
        else:
            return jsonify({'status': 'error', 'message': 'No moves available'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)