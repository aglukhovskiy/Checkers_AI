import pygame
import sys


class CheckersGame:
    def __init__(self):
        self.board = self._create_board()
        self.current_player = 'white'  # white начинает
        self.selected_piece = None
        self.possible_moves = []
        self.capture_moves = []
        self.kings = set()  # хранение координат дамок

    def _create_board(self):
        # создаем пустую доску 8x8
        board = [[None for _ in range(8)] for _ in range(8)]

        # Расставляем начальные позиции белых шашек
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:  # только на темных клетках
                    board[row][col] = 'white'

        # Расставляем начальные позиции черных шашек
        for row in range(0, 3):
            for col in range(8):
                if (row + col) % 2 == 1:  # только на темных клетках
                    board[row][col] = 'black'

        return board

    def get_piece(self, row, col):
        """Возвращает шашку на указанной позиции"""
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None

    def is_king(self, row, col):
        """Проверяет, является ли шашка дамкой"""
        return (row, col) in self.kings

    def select_piece(self, row, col):
        """Выбирает шашку для движения"""
        piece = self.get_piece(row, col)

        if piece == self.current_player:
            self.selected_piece = (row, col)
            # Вычисляем возможные ходы для выбранной шашки
            self.capture_moves = self.get_capture_moves(row, col)

            # Если есть обязательные взятия, другие ходы не разрешены
            if not self.capture_moves:
                self.possible_moves = self.get_regular_moves(row, col)
            else:
                self.possible_moves = []

            return True
        return False

    def get_regular_moves(self, row, col):
        """Получает обычные ходы для шашки"""
        moves = []
        piece = self.get_piece(row, col)

        if piece:
            if self.is_king(row, col):
                # Дамка может ходить по диагонали на любое расстояние
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    while 0 <= r < 8 and 0 <= c < 8 and self.get_piece(r, c) is None:
                        moves.append((r, c))
                        r += dr
                        c += dc
            else:
                # Обычная шашка ходит только вперед
                directions = [(-1, -1), (-1, 1)] if piece == 'white' else [(1, -1), (1, 1)]
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    if 0 <= r < 8 and 0 <= c < 8 and self.get_piece(r, c) is None:
                        moves.append((r, c))

        return moves

    def get_capture_moves(self, row, col):
        """Получает ходы со взятием для шашки"""
        capture_moves = []
        piece = self.get_piece(row, col)

        if piece:
            if self.is_king(row, col):
                # Дамка может бить по диагонали на любое расстояние
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    while 0 <= r < 8 and 0 <= c < 8:
                        if self.get_piece(r, c) is None:
                            r += dr
                            c += dc
                            continue

                        # Найдена потенциальная шашка для взятия
                        if self.get_piece(r, c) != piece:
                            # Проверяем, есть ли пустая клетка за ней
                            capture_r, capture_c = r + dr, c + dc
                            while 0 <= capture_r < 8 and 0 <= capture_c < 8:
                                if self.get_piece(capture_r, capture_c) is None:
                                    capture_moves.append((capture_r, capture_c, r, c))  # ход, взятие
                                    capture_r += dr
                                    capture_c += dc
                                else:
                                    break
                        break
            else:
                # Обычная шашка может бить в любом направлении
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    if 0 <= r < 8 and 0 <= c < 8 and self.get_piece(r, c) and self.get_piece(r, c) != piece:
                        capture_r, capture_c = r + dr, c + dc
                        if 0 <= capture_r < 8 and 0 <= capture_c < 8 and self.get_piece(capture_r, capture_c) is None:
                            capture_moves.append((capture_r, capture_c, r, c))  # ход, взятие

        return capture_moves

    def check_for_kings(self):
        """Проверяет, достигли ли какие-либо шашки противоположного края доски, чтобы стать дамками"""
        for col in range(8):
            # Белые шашки становятся дамками в верхнем ряду
            if self.get_piece(0, col) == 'white':
                self.kings.add((0, col))

            # Черные шашки становятся дамками в нижнем ряду
            if self.get_piece(7, col) == 'black':
                self.kings.add((7, col))

    def move_piece(self, to_row, to_col):
        """Перемещает выбранную шашку на указанную позицию"""
        if self.selected_piece:
            from_row, from_col = self.selected_piece

            # Проверяем, является ли это допустимым перемещением
            if (to_row, to_col) in self.possible_moves:
                # Выполняем обычный ход
                self.board[to_row][to_col] = self.board[from_row][from_col]
                self.board[from_row][from_col] = None

                # Если шашка была дамкой, обновляем её позицию
                if self.is_king(from_row, from_col):
                    self.kings.remove((from_row, from_col))
                    self.kings.add((to_row, to_col))

                # Переключаем игрока
                self.current_player = 'black' if self.current_player == 'white' else 'white'
                self.check_for_kings()
                return True

            # Проверяем ходы со взятием
            for move in self.capture_moves:
                if (to_row, to_col) == (move[0], move[1]):
                    # Выполняем взятие
                    self.board[to_row][to_col] = self.board[from_row][from_col]
                    self.board[from_row][from_col] = None

                    # Удаляем взятую шашку
                    captured_row, captured_col = move[2], move[3]
                    self.board[captured_row][captured_col] = None

                    # Если шашка-взятая была дамкой, удаляем её из списка
                    if (captured_row, captured_col) in self.kings:
                        self.kings.remove((captured_row, captured_col))

                    # Если шашка была дамкой, обновляем её позицию
                    if self.is_king(from_row, from_col):
                        self.kings.remove((from_row, from_col))
                        self.kings.add((to_row, to_col))

                    # Проверяем, есть ли дополнительные взятия
                    additional_captures = self.get_capture_moves(to_row, to_col)
                    if additional_captures:
                        self.selected_piece = (to_row, to_col)
                        self.capture_moves = additional_captures
                        self.possible_moves = []
                    else:
                        # Если нет дополнительных взятий, переключаем игрока
                        self.current_player = 'black' if self.current_player == 'white' else 'white'

                    self.check_for_kings()
                    return True

            return False
        return False

    def check_winner(self):
        """Проверяет, есть ли победитель"""
        white_pieces = 0
        black_pieces = 0

        for row in range(8):
            for col in range(8):
                if self.get_piece(row, col) == 'white':
                    white_pieces += 1
                elif self.get_piece(row, col) == 'black':
                    black_pieces += 1

        if white_pieces == 0:
            return 'black'
        elif black_pieces == 0:
            return 'white'

        # Проверка на наличие возможных ходов
        current_player_can_move = False
        for row in range(8):
            for col in range(8):
                if self.get_piece(row, col) == self.current_player:
                    captures = self.get_capture_moves(row, col)
                    if captures:
                        current_player_can_move = True
                        break

                    moves = self.get_regular_moves(row, col)
                    if moves:
                        current_player_can_move = True
                        break

            if current_player_can_move:
                break

        if not current_player_can_move:
            return 'black' if self.current_player == 'white' else 'white'

        return None

    def reset_game(self):
        """Сбрасывает игру в начальное состояние"""
        self.board = self._create_board()
        self.current_player = 'white'
        self.selected_piece = None
        self.possible_moves = []
        self.capture_moves = []
        self.kings = set()

# Инициализация Pygame
pygame.init()

# Константы
WIDTH, HEIGHT = 800, 800
SQUARE_SIZE = WIDTH // 8
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_SQUARE = (101, 67, 33)  # Коричневый
LIGHT_SQUARE = (255, 248, 220)  # Бежевый
HIGHLIGHT = (124, 252, 0)  # Зеленый для выделения
FPS = 60

# Создание окна
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Шашки")
clock = pygame.time.Clock()

# Создание игры
game = CheckersGame()


def draw_board():
    """Отрисовка доски"""
    for row in range(8):
        for col in range(8):
            # Определяем цвет клетки
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE

            # Рисуем клетки доски
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            # Выделяем выбранную шашку
            if game.selected_piece and game.selected_piece == (row, col):
                pygame.draw.rect(screen, HIGHLIGHT,
                                 (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)

            # Выделяем возможные ходы
            if (row, col) in game.possible_moves:
                pygame.draw.circle(screen, HIGHLIGHT,
                                   (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                    row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                   SQUARE_SIZE // 6)

            # Отображаем возможные взятия
            for move in game.capture_moves:
                if (row, col) == (move[0], move[1]):
                    pygame.draw.circle(screen, HIGHLIGHT,
                                       (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                        row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                       SQUARE_SIZE // 6)


def draw_pieces():
    """Отрисовка шашек"""
    for row in range(8):
        for col in range(8):
            piece = game.get_piece(row, col)
            if piece:
                # Рисуем шашки
                color = WHITE if piece == 'white' else BLACK
                center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2)
                pygame.draw.circle(screen, color, center, SQUARE_SIZE // 2 - 10)

                # Обозначаем дамки
                if game.is_king(row, col):
                    crown_color = BLACK if piece == 'white' else WHITE
                    pygame.draw.circle(screen, crown_color, center, SQUARE_SIZE // 4 - 5)


def main():
    """Основной игровой цикл"""
    running = True
    winner = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and not winner:
                # Получаем координаты клика
                pos = pygame.mouse.get_pos()
                col = pos[0] // SQUARE_SIZE
                row = pos[1] // SQUARE_SIZE

                # Если уже выбрана шашка, пытаемся сделать ход
                if game.selected_piece:
                    if game.move_piece(row, col):
                        # Ход выполнен, очищаем выбор
                        game.selected_piece = None
                        game.possible_moves = []
                        game.capture_moves = []

                        # Проверяем победителя
                        winner = game.check_winner()
                    elif game.get_piece(row, col) == game.current_player:
                        # Выбрана новая шашка
                        game.select_piece(row, col)
                else:
                    # Пытаемся выбрать шашку
                    game.select_piece(row, col)

            # Сброс игры при нажатии клавиши R
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game.reset_game()
                winner = None

        # Отрисовка
        screen.fill(BLACK)
        draw_board()
        draw_pieces()

        # Отображение победителя
        if winner:
            font = pygame.font.SysFont(None, 72)
            text = font.render(f"Победили {winner}!", True, (255, 0, 0))
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(text, text_rect)

            font_small = pygame.font.SysFont(None, 36)
            restart_text = font_small.render("Нажмите R для новой игры", True, (255, 0, 0))
            restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
            screen.blit(restart_text, restart_rect)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()