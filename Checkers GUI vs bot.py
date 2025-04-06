
import pygame
import h5py
from rl.pg_agent import load_policy_agent
from Board_v2 import CheckersGame
import numpy as np

def play_against_bot(agent_filename):
    """Играет против обученного агента"""

    # Загружаем агента
    with h5py.File(agent_filename, 'r') as agent_file:
        bot = load_policy_agent(agent_file)

    # Инициализируем Pygame
    pygame.init()

    # Константы
    BOARD_SIZE = 600
    SQUARE_SIZE = BOARD_SIZE // 8
    PANEL_WIDTH = 300
    WIDTH = BOARD_SIZE + PANEL_WIDTH
    HEIGHT = BOARD_SIZE

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    DARK_SQUARE = (101, 67, 33)
    LIGHT_SQUARE = (255, 248, 220)
    HIGHLIGHT = (124, 252, 0)
    POSSIBLE_MOVE = (173, 216, 230)
    PANEL_BG = (240, 240, 240)
    PANEL_BORDER = (200, 200, 200)
    TEXT_COLOR = (50, 50, 50)
    HIGHLIGHT_TEXT = (0, 100, 0)

    FPS = 60

    # Создаем игру
    game = CheckersGame()
    # board = game

    # Создаем окно
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Шашки против бота')
    clock = pygame.time.Clock()

    # Шрифты
    title_font = pygame.font.SysFont('Arial', 36)
    info_font = pygame.font.SysFont('Arial', 24)
    small_font = pygame.font.SysFont('Arial', 18)

    # Интерфейсные переменные
    selected_piece = None
    possible_moves = []
    player_color = 1  # 1 - белые, -1 - черные
    bot_thinking = False

    def draw_board():
        """Рисует доску"""
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def draw_pieces():
        """Рисует шашки"""
        for row in range(8):
            for col in range(8):
                piece = game.get_piece(row, col)
                if piece != 0:
                    color = WHITE if piece > 0 else BLACK
                    border_color = BLACK if piece > 0 else WHITE
                    center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                    center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2

                    pygame.draw.circle(screen, color, (center_x, center_y), SQUARE_SIZE // 2 - 10)
                    pygame.draw.circle(screen, border_color, (center_x, center_y), SQUARE_SIZE // 2 - 10, 2)

                    if abs(piece) > 1:  # Дамка
                        pygame.draw.circle(screen, border_color, (center_x, center_y), SQUARE_SIZE // 4)
                        pygame.draw.circle(screen, color, (center_x, center_y), SQUARE_SIZE // 4 - 5)

    def draw_selected():
        """Отображает выделенную шашку и возможные ходы"""
        if selected_piece:
            row, col = selected_piece
            pygame.draw.rect(screen, HIGHLIGHT,
                             (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)
            for move in possible_moves:
                target_row, target_col = move[4], move[5]
                center_x = target_col * SQUARE_SIZE + SQUARE_SIZE // 2
                center_y = target_row * SQUARE_SIZE + SQUARE_SIZE // 2
                pygame.draw.circle(screen, POSSIBLE_MOVE, (center_x, center_y), SQUARE_SIZE // 4)

    def draw_info_panel():
        """Рисует информационную панель"""
        panel_rect = pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(screen, PANEL_BG, panel_rect)
        pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 2)

        title_surface = title_font.render("ШАШКИ VS БОТ", True, TEXT_COLOR)
        screen.blit(title_surface, (BOARD_SIZE + 20, 20))

        pygame.draw.line(screen, PANEL_BORDER,
                         (BOARD_SIZE + 10, 70),
                         (WIDTH - 10, 70),
                         2)

        if bot_thinking:
            player_text = "Ход бота..."
        else:
            player_text = "Ваш ход" if game.current_player == player_color else "Ход бота"

        player_color_text = "Вы играете за " + ("белых" if player_color == 1 else "черных")
        player_surface = info_font.render(player_color_text, True, TEXT_COLOR)
        screen.blit(player_surface, (BOARD_SIZE + 20, 90))

        turn_surface = info_font.render(player_text, True, HIGHLIGHT_TEXT)
        screen.blit(turn_surface, (BOARD_SIZE + 20, 120))

        # Счет игры
        pieces = game.get_number_of_pieces_and_kings()
        score_text = f"Счет: Белые {pieces[0] + pieces[2]} - {pieces[1] + pieces[3]} Черные"
        score_surface = info_font.render(score_text, True, TEXT_COLOR)
        screen.blit(score_surface, (BOARD_SIZE + 20, 160))

        # Детали счета
        details_text = f"Белые: {pieces[0]} шашек, {pieces[2]} дамок"
        details_surface = small_font.render(details_text, True, TEXT_COLOR)
        screen.blit(details_surface, (BOARD_SIZE + 20, 190))

        details_text = f"Черные: {pieces[1]} шашек, {pieces[3]} дамок"
        details_surface = small_font.render(details_text, True, TEXT_COLOR)
        screen.blit(details_surface, (BOARD_SIZE + 20, 210))

        # Инструкции
        pygame.draw.line(screen, PANEL_BORDER,
                         (BOARD_SIZE + 10, HEIGHT - 120),
                         (WIDTH - 10, HEIGHT - 120),
                         2)

        instructions = [
            "Управление:",
            "- Левый клик: выбор шашки / ход",
            "- R: начать новую игру",
            "- C: сменить цвет"
        ]

        for i, text in enumerate(instructions):
            instr_surface = small_font.render(text, True, TEXT_COLOR)
            screen.blit(instr_surface, (BOARD_SIZE + 20, HEIGHT - 110 + i * 25))

        # Если игра окончена, показываем победителя
        if game.winner is not None:
            if game.winner == player_color:
                winner_text = "Вы победили!"
            elif game.winner == -player_color:
                winner_text = "Бот победил!"
            else:
                winner_text = "Ничья!"

            winner_surface = title_font.render(winner_text, True, (255, 0, 0))
            overlay = pygame.Surface((350, 50), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 200))
            screen.blit(overlay, (BOARD_SIZE // 2 - 175, BOARD_SIZE // 2 - 25))
            win_rect = winner_surface.get_rect(center=(BOARD_SIZE // 2, BOARD_SIZE // 2))
            screen.blit(winner_surface, win_rect)

    def check_and_select_piece(row, col):
        """Проверяет возможность выбора шашки и определяет возможные ходы"""
        nonlocal selected_piece, possible_moves

        piece = game.get_piece(row, col)

        # Проверяем, что это шашка игрока
        if piece != 0 and np.sign(piece) == player_color:
            # Проверяем наличие обязательных взятий у любой шашки
            all_captures = game.get_capture_moves()

            if all_captures:
                # Есть обязательные взятия, проверяем для этой шашки
                piece_captures = []
                for sequence in all_captures:
                    if sequence and sequence[0][0] == row and sequence[0][1] == col:
                        # Добавляем только первый ход из последовательности
                        piece_captures.append(sequence[0])

                if piece_captures:
                    # Эта шашка может совершить взятие, выбираем её
                    selected_piece = (row, col)
                    possible_moves = piece_captures
                else:
                    # Эта шашка не может совершать взятие, но другие могут
                    print("У этой шашки нет обязательных взятий!")
            else:
                # Нет обязательных взятий, проверяем обычные ходы
                regular_moves = game.get_regular_moves_for_piece(row, col)
                if regular_moves:
                    selected_piece = (row, col)
                    possible_moves = regular_moves
                else:
                    print("У этой шашки нет доступных ходов!")

    def handle_click(pos):
        """Обрабатывает клик мыши"""
        nonlocal selected_piece, possible_moves

        # Если клик за пределами доски или ход бота, игнорируем
        if pos[0] >= BOARD_SIZE or bot_thinking or game.current_player != player_color:
            return

        # Если игра закончена, игнорируем клики
        if game.game_is_on == 0:
            return

        col = pos[0] // SQUARE_SIZE
        row = pos[1] // SQUARE_SIZE

        # Если шашка уже выбрана
        if selected_piece:
            # Проверяем, выбрал ли игрок клетку для хода
            target_move = None
            for move in possible_moves:
                if move[4] == row and move[5] == col:
                    target_move = move
                    break

            # Если выбран допустимый ход
            if target_move:
                # Определяем, является ли это взятием
                is_capture = target_move[2] is not None

                # Выполняем ход
                game.move_piece(target_move, capture_move=is_capture)

                # Если это было взятие, проверяем возможность продолжения взятия
                if is_capture:
                    next_captures = game.get_first_capture_moves((row, col))
                    if next_captures:
                        # Если можно продолжить взятие, обновляем выбор и возможные ходы
                        selected_piece = (row, col)
                        possible_moves = next_captures
                        return

                # Если нет продолжения взятия или это был обычный ход, переходим к следующему игроку
                game.current_player = -game.current_player
                selected_piece = None
                possible_moves = []
                game.check_winner()
            else:
                # Если клик не по возможному ходу, проверяем, выбрал ли игрок свою шашку
                piece = game.get_piece(row, col)
                if piece != 0 and np.sign(piece) == player_color:
                    # Пытаемся выбрать новую шашку
                    check_and_select_piece(row, col)
                else:
                    # Отменяем выбор
                    selected_piece = None
                    possible_moves = []
        else:
            # Если шашка не выбрана, пытаемся выбрать
            check_and_select_piece(row, col)

    def bot_move():
        """Выполняет ход бота"""
        nonlocal bot_thinking

        if game.game_is_on == 0 or game.current_player == player_color:
            return

        bot_thinking = True

        # Получаем ход от бота
        move = bot.select_move(game, 0)

        if move:
            # Выполняем ход
            is_capture = move[2] is not None
            game.move_piece(move, capture_move=is_capture)

            # Проверяем, есть ли дальнейшие взятия (множественное взятие)
            next_pos = (move[4], move[5])
            while is_capture:
                next_captures = game.get_first_capture_moves(next_pos)
                if next_captures:
                    # Выбираем следующий ход бота
                    next_move = bot.select_move(game, 0)
                    if next_move:
                        is_capture = next_move[2] is not None
                        game.move_piece(next_move, capture_move=is_capture)
                        next_pos = (next_move[4], next_move[5])
                    else:
                        break
                else:
                    break

            # Переключаем игрока
            game.current_player = -game.current_player
            game.check_winner()

        bot_thinking = False

    def reset_game():
        """Сбрасывает игру в начальное состояние"""
        nonlocal selected_piece, possible_moves, game

        game = CheckersGame()
        selected_piece = None
        possible_moves = []

    def change_color():
        """Меняет цвет игрока"""
        nonlocal player_color, selected_piece, possible_moves

        player_color = -player_color
        reset_game()

    # Основной игровой цикл
    running = True
    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Левая кнопка мыши
                    handle_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Клавиша R для рестарта
                    reset_game()
                elif event.key == pygame.K_c:  # Клавиша C для смены цвета
                    change_color()

        # Если ходит бот, выполняем его ход
        if game.current_player != player_color and game.game_is_on == 1 and not bot_thinking:
            bot_move()

        # Отрисовка
        screen.fill(BLACK)
        draw_board()
        draw_selected()
        draw_pieces()
        draw_info_panel()

        # Обновление экрана
        pygame.display.flip()
        clock.tick(FPS)

    # Выход
    pygame.quit()



# Пример использования
if __name__ == "__main__":
    play_against_bot('rl/models_n_exp/test_model.hdf5')
