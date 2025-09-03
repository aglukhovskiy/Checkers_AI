import pygame
import sys
import numpy as np
from copy import copy
import time
from ..core.board_v2 import CheckersGame


# Константы для интерфейса
BOARD_SIZE = 600  # Размер доски
SQUARE_SIZE = BOARD_SIZE // 8  # Размер одной клетки
PANEL_WIDTH = 300  # Ширина информационной панели
WIDTH = BOARD_SIZE + PANEL_WIDTH  # Общая ширина окна
HEIGHT = BOARD_SIZE  # Высота окна

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_SQUARE = (101, 67, 33)  # Коричневый
LIGHT_SQUARE = (255, 248, 220)  # Бежевый
HIGHLIGHT = (124, 252, 0)  # Зеленый для выделения
POSSIBLE_MOVE = (173, 216, 230)  # Голубой для возможных ходов
PANEL_BG = (240, 240, 240)  # Светло-серый фон панели
PANEL_BORDER = (200, 200, 200)  # Серая рамка панели
TEXT_COLOR = (50, 50, 50)  # Тёмно-серый текст
HIGHLIGHT_TEXT = (0, 100, 0)  # Тёмно-зелёный для выделенного текста

FPS = 60

# Инициализация Pygame
pygame.init()

# Создание игры
game = CheckersGame()

# Создание окна
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Шашки')
clock = pygame.time.Clock()

# Шрифты
title_font = pygame.font.SysFont('Arial', 36)
info_font = pygame.font.SysFont('Arial', 24)
small_font = pygame.font.SysFont('Arial', 18)

# Переменные для интерфейса
selected_piece = None  # Выбранная шашка (row, col)
possible_moves = []  # Возможные ходы для выбранной шашки
start_time = time.time()  # Время начала игры
white_time = 0  # Время белых (в секундах)
black_time = 0  # Время черных (в секундах)
current_player_start_time = time.time()  # Время начала хода текущего игрока
game_moves = []  # История ходов в нотации


def format_time(seconds):
    """Форматирует время в формат MM:SS"""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


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
                # Определяем цвет и позицию
                color = WHITE if piece > 0 else BLACK
                border_color = BLACK if piece > 0 else WHITE
                center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2

                # Рисуем шашку
                pygame.draw.circle(screen, color, (center_x, center_y), SQUARE_SIZE // 2 - 10)
                pygame.draw.circle(screen, border_color, (center_x, center_y), SQUARE_SIZE // 2 - 10, 2)

                # Обозначаем дамку
                if abs(piece) > 1:
                    pygame.draw.circle(screen, border_color, (center_x, center_y), SQUARE_SIZE // 4)
                    pygame.draw.circle(screen, color, (center_x, center_y), SQUARE_SIZE // 4 - 5)


def draw_selected():
    """Отображает выделенную шашку и возможные ходы"""
    if selected_piece:
        # Выделяем выбранную шашку
        row, col = selected_piece
        pygame.draw.rect(screen, HIGHLIGHT,
                         (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)

        # Отображаем возможные ходы
        for move in possible_moves:
            # Целевая позиция хода
            target_row, target_col = move[4], move[5]

            # Рисуем круг на клетке возможного хода
            center_x = target_col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = target_row * SQUARE_SIZE + SQUARE_SIZE // 2
            pygame.draw.circle(screen, POSSIBLE_MOVE, (center_x, center_y), SQUARE_SIZE // 4)


def draw_info_panel():
    """Рисует информационную панель справа от доски"""
    # Рисуем фон панели
    panel_rect = pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, HEIGHT)
    pygame.draw.rect(screen, PANEL_BG, panel_rect)
    pygame.draw.rect(screen, PANEL_BORDER, panel_rect, 2)

    # Заголовок
    title_surface = title_font.render("ШАШКИ", True, TEXT_COLOR)
    screen.blit(title_surface, (BOARD_SIZE + 20, 20))

    # Линия после заголовка
    pygame.draw.line(screen, PANEL_BORDER,
                     (BOARD_SIZE + 10, 70),
                     (WIDTH - 10, 70),
                     2)

    # Текущий игрок
    if game.game_is_on:
        player_text = "Ход: БЕЛЫЕ" if game.current_player > 0 else "Ход: ЧЕРНЫЕ"
        player_color = HIGHLIGHT_TEXT if game.game_is_on else TEXT_COLOR
        player_surface = info_font.render(player_text, True, player_color)
        screen.blit(player_surface, (BOARD_SIZE + 20, 90))

    # Информация о времени
    time_text = "Время игры: " + format_time(time.time() - start_time)
    time_surface = info_font.render(time_text, True, TEXT_COLOR)
    screen.blit(time_surface, (BOARD_SIZE + 20, 130))

    # Время белых и черных
    white_time_text = "Белые: " + format_time(
        white_time + (time.time() - current_player_start_time if game.current_player > 0 and game.game_is_on else 0))
    white_surface = info_font.render(white_time_text, True, TEXT_COLOR)
    screen.blit(white_surface, (BOARD_SIZE + 20, 160))

    black_time_text = "Черные: " + format_time(
        black_time + (time.time() - current_player_start_time if game.current_player < 0 and game.game_is_on else 0))
    black_surface = info_font.render(black_time_text, True, TEXT_COLOR)
    screen.blit(black_surface, (BOARD_SIZE + 20, 190))

    # Счет игры
    pieces = game.get_number_of_pieces_and_kings()
    score_text = f"Счет: Белые {pieces[0] + pieces[2]} - {pieces[1] + pieces[3]} Черные"
    score_surface = info_font.render(score_text, True, TEXT_COLOR)
    screen.blit(score_surface, (BOARD_SIZE + 20, 230))

    # Детали счета (шашки/дамки)
    details_text = f"Белые: {pieces[0]} шашек, {pieces[2]} дамок"
    details_surface = small_font.render(details_text, True, TEXT_COLOR)
    screen.blit(details_surface, (BOARD_SIZE + 20, 260))

    details_text = f"Черные: {pieces[1]} шашек, {pieces[3]} дамок"
    details_surface = small_font.render(details_text, True, TEXT_COLOR)
    screen.blit(details_surface, (BOARD_SIZE + 20, 280))

    # Линия перед инструкциями
    pygame.draw.line(screen, PANEL_BORDER,
                     (BOARD_SIZE + 10, HEIGHT - 120),
                     (WIDTH - 10, HEIGHT - 120),
                     2)

    # Инструкции
    instructions = [
        "Управление:",
        "- Левый клик: выбор шашки / ход",
        "- R: начать новую игру"
    ]

    for i, text in enumerate(instructions):
        instr_surface = small_font.render(text, True, TEXT_COLOR)
        screen.blit(instr_surface, (BOARD_SIZE + 20, HEIGHT - 110 + i * 25))

    # Если игра окончена, показываем победителя
    if game.winner is not None and game.winner != 0:
        winner_text = "Победили БЕЛЫЕ!" if game.winner > 0 else "Победили ЧЕРНЫЕ!"
        winner_surface = title_font.render(winner_text, True, (255, 0, 0))

        # Создаем полупрозрачный прямоугольник для фона текста
        overlay = pygame.Surface((350, 50), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 200))
        screen.blit(overlay, (BOARD_SIZE // 2 - 175, BOARD_SIZE // 2 - 25))

        # Выводим текст о победителе на доске
        win_rect = winner_surface.get_rect(center=(BOARD_SIZE // 2, BOARD_SIZE // 2))
        screen.blit(winner_surface, win_rect)


def handle_click(pos):
    """Обрабатывает клик мыши"""
    global selected_piece, possible_moves, current_player_start_time, white_time, black_time

    # Если клик за пределами доски, игнорируем
    if pos[0] >= BOARD_SIZE:
        return

    # Если игра закончена, игнорируем клики
    if game.game_is_on == 0:
        return

    # Получаем координаты клика
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
                    # Если можно продолжить взятие, обновляем выбор
                    selected_piece = (row, col)
                    possible_moves = next_captures
                    return

            # Если нет продолжения взятия, переходим к следующему игроку
            # Обновляем время текущего игрока
            elapsed = time.time() - current_player_start_time
            if game.current_player > 0:
                white_time += elapsed
            else:
                black_time += elapsed

            # Меняем игрока
            game.current_player = -game.current_player
            current_player_start_time = time.time()

            selected_piece = None
            possible_moves = []
            game.check_winner()
        else:
            # Если клик не по возможному ходу, проверяем, выбрал ли игрок свою шашку
            piece = game.get_piece(row, col)
            if piece != 0 and np.sign(piece) == game.current_player:
                # Пытаемся выбрать новую шашку
                check_and_select_piece(row, col)
            else:
                # Отменяем выбор
                selected_piece = None
                possible_moves = []
    else:
        # Если шашка не выбрана, пытаемся выбрать
        check_and_select_piece(row, col)


def check_and_select_piece(row, col):
    """Проверяет возможность выбора шашки и определяет возможные ходы"""
    global selected_piece, possible_moves

    piece = game.get_piece(row, col)

    # Проверяем, что это шашка текущего игрока
    if piece != 0 and np.sign(piece) == game.current_player:
        # Проверяем наличие обязательных взятий
        all_captures = game.get_capture_moves()

        if all_captures:
            # Есть обязательные взятия, проверяем эту шашку
            piece_captures = []
            for capture_seq in all_captures:
                if capture_seq and capture_seq[0][0] == row and capture_seq[0][1] == col:
                    piece_captures.append(capture_seq[0])

            if piece_captures:
                # Эта шашка может совершить взятие
                selected_piece = (row, col)
                possible_moves = piece_captures
            else:
                # Эта шашка не может совершать взятие, но другие могут - ОБЫЧНЫЕ ХОДЫ ЗАБЛОКИРОВАНЫ
                print("У этой шашки нет обязательных взятий! Обычные ходы заблокированы.")
        else:
            # Нет обязательных взятий, проверяем обычные ходы
            regular_moves = game.get_regular_moves_for_piece(row, col)
            if regular_moves:
                selected_piece = (row, col)
                possible_moves = regular_moves
            else:
                print("У этой шашки нет доступных ходов!")


def reset_game():
    """Сбрасывает игру и все счетчики"""
    global selected_piece, possible_moves, start_time, current_player_start_time, white_time, black_time

    game.reset_game()
    selected_piece = None
    possible_moves = []
    start_time = time.time()
    current_player_start_time = time.time()
    white_time = 0
    black_time = 0


def main():
    """Основной игровой цикл"""
    global selected_piece, possible_moves

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
    sys.exit()

if __name__ == "__main__":
    main()