from flask import Flask, request, jsonify
from flask_cors import CORS
from src.core.board_v2 import CheckersGame
import numpy as np

app = Flask(__name__)
CORS(app, resources={
    r"/*": {"origins": ["http://localhost:8000", "http://127.0.0.1:8000", "file://"]}
})

# Инициализация игры
game = CheckersGame()

@app.route('/new_game', methods=['POST'])
def new_game():
    global game
    game = CheckersGame()
    return jsonify({'status': 'ok'})

@app.route('/make_move', methods=['POST'])
def make_move():
    data = request.json
    move_str = data.get('move')
    print(f"APIv2: Получен ход: '{move_str}'")
    
    # Конвертируем строковый ход в формат board_v2
    # Формат: "a3b4" -> (from_row, from_col, to_row, to_col)
    from_col = ord(move_str[0]) - ord('a')
    from_row = int(move_str[1]) - 1  # Convert notation (1-8) to engine row (0-7)
    to_col = ord(move_str[2]) - ord('a')
    to_row = int(move_str[3]) - 1  # Convert notation (1-8) to engine row (0-7)
    
    print(f"APIv2: Конвертировано: '{move_str}' -> from ({from_row},{from_col}) to ({to_row},{to_col})")
    
    # Получаем все возможные ходы
    available_moves = game.get_possible_moves()
    print(f"APIv2: Доступно ходов: {len(available_moves)} последовательностей")
    
    selected_move = None
    
    # Проверяем все последовательности ходов
    for move_sequence in available_moves:
        for move in move_sequence:
            # Для обычных ходов проверяем только начальную и конечную позиции
            if move[2] is None:  # Это обычный ход
                if (move[0] == from_row and move[1] == from_col and
                    move[4] == to_row and move[5] == to_col):
                    selected_move = [move]  # Обычный ход - последовательность из одного хода
                    break
            else:  # Это взятие
                if (move[0] == from_row and move[1] == from_col and
                    move[4] == to_row and move[5] == to_col):
                    selected_move = move_sequence
                    break
        if selected_move:
            break
    
    if selected_move:
        print(f"APIv2: Выполняем ход: {selected_move}")
        game.next_turn(selected_move)
        
        # Формируем ответ с состоянием доски
        board_state = {
            'field': game.board.tolist(),
            'current_player': 'white' if game.current_player == 1 else 'black',
            'game_over': game.game_is_on == 0
        }
        
        return jsonify({
            'status': 'ok',
            'board': board_state
        })
    else:
        print(f"APIv2: Ход не найден в доступных!")
        return jsonify({'status': 'error', 'message': 'Invalid move'})

@app.route('/get_state', methods=['GET'])
def get_state():
    board_state = {
        'field': game.board.tolist(),
        'current_player': 'white' if game.current_player == 1 else 'black',
        'game_over': game.game_is_on == 0
    }
    
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
    board_row = row  # GUI and engine use same row numbering (0-7 top to bottom)
    
    # Проверяем наличие обязательных взятий
    all_captures = game.get_capture_moves()
    
    if all_captures:
        print(f"Found captures: {all_captures}")
        # Есть обязательные взятия - проверяем, может ли эта шашка брать
        piece_captures = []
        for capture_seq in all_captures:
            if capture_seq:  # Проверяем, что последовательность не пустая
                first_move = capture_seq[0]
                print(f"Checking capture: {first_move} vs piece: ({board_row}, {col})")
                if first_move[0] == board_row and first_move[1] == col:
                    piece_captures.append(first_move)
        
        print(f"Piece captures: {piece_captures}")
        
        if piece_captures:
            # Эта шашка может совершить взятие - возвращаем только взятия
            moves = []
            for move in piece_captures:
                interface_to_row = move[4]  # No inversion needed for captures - engine and GUI aligned
                moves.append([interface_to_row, move[5]])
            
            return jsonify({
                'status': 'ok',
                'moves': moves,
                'capture_required': True
            })
        else:
            # Эта шашка не может совершать взятие, но другие могут - обычные ходы заблокированы
            # Но давайте вернем все возможные взятия, чтобы пользователь видел, какие шашки могут брать
            all_capture_moves = []
            for capture_seq in all_captures:
                if capture_seq:
                    first_move = capture_seq[0]
                    interface_to_row = first_move[4]  # Убираем инверсию - используем прямое соответствие
                    all_capture_moves.append([interface_to_row, first_move[5]])
            
            return jsonify({
                'status': 'ok',
                'moves': all_capture_moves,
                'capture_required': True,
                'all_captures': True  # Флаг, что это все возможные взятия
            })
    else:
        # Нет обязательных взятий, возвращаем обычные ходы
        piece_moves = game.get_regular_moves_for_piece(board_row, col)
        
        # Конвертируем ходы обратно в формат интерфейса
        moves = []
        for move in piece_moves:
            interface_to_row = move[4]  # No inversion needed - engine and GUI now aligned
            moves.append([interface_to_row, move[5]])
        
        return jsonify({
            'status': 'ok',
            'moves': moves,
            'capture_required': False
        })

@app.route('/bot_move', methods=['POST'])
def bot_move():
    if game.current_player == 1:  # Белые ходят - не ход бота
        return jsonify({'status': 'error', 'message': 'Not bot turn'})
    
    try:
        available_moves = game.get_possible_moves()
        if available_moves:
            # Берем первый доступный ход
            move_sequence = available_moves[0]
            game.next_turn(move_sequence)
            
            board_state = {
                'field': game.board.tolist(),
                'current_player': 'white' if game.current_player == 1 else 'black',
                'game_over': game.game_is_on == 0
            }
            
            return jsonify({
                'status': 'ok',
                'board': board_state
            })
        else:
            return jsonify({'status': 'error', 'message': 'No moves available'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)