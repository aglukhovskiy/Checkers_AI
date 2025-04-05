import numpy as np
from copy import copy
import pygame
import sys
from Board_test import CheckersGame

# Инициализация Pygame
pygame.init()

# Константы для интерфейса
WIDTH, HEIGHT = 800, 800
SQUARE_SIZE = WIDTH // 8
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_SQUARE = (101, 67, 33)  # Коричневый
LIGHT_SQUARE = (255, 248, 220)  # Бежевый
HIGHLIGHT = (124, 252, 0)  # Зеленый для выделения
POSSIBLE_MOVE = (173, 216, 230)  # Голубой для возможных ходов
FPS = 60

# Инициализация игры
game = CheckersGame()

# Шрифт для текста
pygame.font.init()
font = pygame.font.SysFont('Arial', 36)

# Создание окна
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Шашки')
clock = pygame.time.Clock()


def draw_board():
    """Отрисовка доски"""
    for row in range(8):
        for col in range(8):
            # Определяем цвет клетки
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def draw_pieces():
    """Отрисовка шашек"""
    for pos in game.pieces:
        row, col = pos
        piece = game.board[row][col]

        # Определяем цвет шашки
        color = WHITE if piece > 0 else BLACK
        border_color = BLACK if piece > 0 else WHITE

        # Координаты центра шашки
        center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
        center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2

        # Рисуем шашку
        pygame.draw.circle(screen, color, (center_x, center_y), SQUARE_SIZE // 2 - 10)
        pygame.draw.circle(screen, border_color, (center_x, center_y), SQUARE_SIZE // 2 - 10, 2)

        # Если это дамка, рисуем корону
        if abs(piece) > 1:
            crown_color = BLACK if piece > 0 else WHITE
            pygame.draw.circle(screen, crown_color, (center_x, center_y), SQUARE_SIZE // 4)
            pygame.draw.circle(screen, color, (center_x, center_y), SQUARE_SIZE // 4 - 5)


def draw_selected_and_possible_moves():
    """Отрисовка выделенной шашки и возможных ходов"""
    if game.selected_piece:
        row, col = game.selected_piece

        # Выделяем выбранную шашку
        pygame.draw.rect(screen, HIGHLIGHT,
                         (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)

        # Выделяем возможные ходы
        for move in game.possible_moves:
            target_row, target_col = move[4], move[5]
            center_x = target_col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = target_row * SQUARE_SIZE + SQUARE_SIZE // 2

            # Рисуем круг в месте возможного хода
            pygame.draw.circle(screen, POSSIBLE_MOVE, (center_x, center_y), SQUARE_SIZE // 4)


def find_move_by_destination(row, col):
    """Находит ход по позиции назначения"""
    for move in game.possible_moves:
        if move[4] == row and move[5] == col:
            return move
    return None


def handle_mouse_click(pos):
    """Обработка клика мыши"""
    col = pos[0] // SQUARE_SIZE
    row = pos[1] // SQUARE_SIZE

    # Проверяем, не закончилась ли игра
    if game.game_is_on == 0:
        return

    # Вывод отладочной информации
    print(f"Клик на ({row}, {col})")
    if (row, col) in game.pieces:
        piece = game.board[row][col]
        print(f"Шашка: {piece}, текущий игрок: {game.current_player}")

    # Если уже выбрана шашка, пытаемся сделать ход
    if game.selected_piece:
        # Проверяем, является ли клик по возможному ходу
        move = find_move_by_destination(row, col)

        if move:
            # Получаем тип хода (обычный или взятие)
            is_capture = move[2] is not None
            # Выполняем ход
            game.move_piece(move, capture_move=is_capture)

            # Проверяем, если это было взятие, может ли шашка продолжить взятие
            if is_capture:
                next_pos = (move[4], move[5])
                further_captures = game.get_first_capture_moves(next_pos)

                if further_captures:
                    # Если есть дальнейшие взятия, обновляем выбранную шашку
                    game.selected_piece = next_pos
                    game.possible_moves = further_captures
                    return

            # Если хода нет или нет дальнейших взятий, передаем ход
            game.selected_piece = None
            game.possible_moves = []
            game.current_player = -game.current_player
            game.check_winner()
        else:
            # Если клик не по возможному ходу
            # Проверяем, есть ли шашка на клике и принадлежит ли она текущему игроку
            if (row, col) in game.pieces and np.sign(game.board[row][col]) == game.current_player:
                # Выбираем новую шашку
                select_piece(row, col)
            else:
                # Отменяем выбор
                game.selected_piece = None
                game.possible_moves = []
    else:
        # Пытаемся выбрать шашку
        select_piece(row, col)


def select_piece(row, col):
    """Выбираем шашку и определяем возможные ходы"""
    # Проверяем, есть ли шашка на позиции и принадлежит ли она текущему игроку
    if (row, col) in game.pieces and np.sign(game.board[row][col]) == game.current_player:
        game.selected_piece = (row, col)

        # Проверяем, есть ли обязательные взятия у любой шашки
        all_capture_moves = game.get_capture_moves()

        # Если есть обязательные взятия
        if all_capture_moves:
            # Находим взятия для выбранной шашки
            piece_capture_moves = []
            for capture_sequence in all_capture_moves:
                if capture_sequence and capture_sequence[0][0] == row and capture_sequence[0][1] == col:
                    # Добавляем только первый ход из последовательности
                    piece_capture_moves.append(capture_sequence[0])

            if piece_capture_moves:
                game.possible_moves = piece_capture_moves
            else:
                # Если у выбранной шашки нет взятий, но есть у других, отменяем выбор
                game.selected_piece = None
                game.possible_moves = []
                print("У этой шашки нет обязательных взятий!")
        else:
            # Если нет обязательных взятий, определяем обычные ходы
            game.possible_moves = game.get_regular_moves(piece=(row, col))
    else:
        print("Невозможно выбрать эту позицию")
        game.selected_piece = None
        game.possible_moves = []


def display_winner():
    """Отображение победителя"""
    if game.winner != 0:
        winner_text = "Победили БЕЛЫЕ!" if game.winner > 0 else "Победили ЧЕРНЫЕ!"
        text_surface = font.render(winner_text, True, (255, 0, 0))
        text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
        screen.blit(text_surface, text_rect)

        restart_text = font.render("Нажмите R для новой игры", True, (255, 0, 0))
        restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
        screen.blit(restart_text, restart_rect)


def display_current_player():
    """Отображение текущего игрока"""
    if game.game_is_on:
        player_text = "Ход БЕЛЫХ" if game.current_player > 0 else "Ход ЧЕРНЫХ"
        text_surface = font.render(player_text, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))


def main():
    """Основной игровой цикл"""
    running = True

    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Обработка клика мыши
                handle_mouse_click(pygame.mouse.get_pos())

            elif event.type == pygame.KEYDOWN:
                # Сброс игры при нажатии R
                if event.key == pygame.K_r:
                    game.reset_game()

        # Рисование
        screen.fill(LIGHT_SQUARE)  # Заливка фона
        draw_board()  # Рисуем доску
        draw_selected_and_possible_moves()  # Рисуем выделенные клетки и возможные ходы
        draw_pieces()  # Рисуем шашки
        display_current_player()  # Отображаем текущего игрока
        display_winner()  # Отображаем победителя, если игра закончена

        # Обновление экрана
        pygame.display.flip()
        clock.tick(FPS)

    # Выход
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()