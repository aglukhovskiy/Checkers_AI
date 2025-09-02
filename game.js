const canvas = document.getElementById('game-board');
const ctx = canvas.getContext('2d');
const statusElement = document.getElementById('status');
const scoreElement = document.getElementById('score');
const restartButton = document.getElementById('restart');

// Константы игры
const BOARD_SIZE = 8;
const SQUARE_SIZE = canvas.width / BOARD_SIZE;
const DARK_COLOR = '#654321';
const LIGHT_COLOR = '#f5deb3';
const WHITE_PIECE_COLOR = '#ffffff';
const BLACK_PIECE_COLOR = '#000000';
const HIGHLIGHT_COLOR = '#00ff00';

// Состояние игры (теперь хранится на сервере)
let gameState = {
    board: Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(0)),
    currentPlayer: 1,
    selectedPiece: null,
    possibleMoves: [],
    gameOver: false
};

// Инициализация новой игры
async function initializeGame() {
    try {
        const response = await fetch('http://localhost:5000/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        const data = await response.json();
        if (data.status === 'ok') {
            await updateBoardState();
        }
    } catch (error) {
        console.error('Error initializing game:', error);
    }
}

// Обновление состояния доски с сервера
async function updateBoardState() {
    try {
        const response = await fetch('http://localhost:5000/get_state', {
            method: 'GET'
        });
        const data = await response.json();
        
        if (data.status === 'ok') {
            gameState.board = data.board.field;
            gameState.currentPlayer = data.board.current_player === 'white' ? 1 : -1;
            gameState.gameOver = data.board.game_over;
            updateGameInfo();
            drawBoard();
        }
    } catch (error) {
        console.error('Error updating board:', error);
    }
}

// Отрисовка доски
function drawBoard() {
    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            // Рисуем клетку
            ctx.fillStyle = (row + col) % 2 === 0 ? LIGHT_COLOR : DARK_COLOR;
            ctx.fillRect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE);
            
            // Рисуем шашку
            if (gameState.board[row][col] !== 0) {
                const isWhite = gameState.board[row][col] > 0;
                ctx.fillStyle = isWhite ? WHITE_PIECE_COLOR : BLACK_PIECE_COLOR;
                ctx.beginPath();
                ctx.arc(
                    col * SQUARE_SIZE + SQUARE_SIZE / 2,
                    row * SQUARE_SIZE + SQUARE_SIZE / 2,
                    SQUARE_SIZE / 2 - 5,
                    0,
                    Math.PI * 2
                );
                ctx.fill();
                ctx.strokeStyle = isWhite ? BLACK_PIECE_COLOR : WHITE_PIECE_COLOR;
                ctx.stroke();
            }
        }
    }
    
    // Подсветка выбранной шашки
    if (gameState.selectedPiece) {
        const [row, col] = gameState.selectedPiece;
        ctx.strokeStyle = HIGHLIGHT_COLOR;
        ctx.lineWidth = 3;
        ctx.strokeRect(
            col * SQUARE_SIZE + 2,
            row * SQUARE_SIZE + 2,
            SQUARE_SIZE - 4,
            SQUARE_SIZE - 4
        );
    }
    
    // Подсветка возможных ходов
    gameState.possibleMoves.forEach(move => {
        const [row, col] = move;
        ctx.fillStyle = 'rgba(173, 216, 230, 0.5)';
        ctx.beginPath();
        ctx.arc(
            col * SQUARE_SIZE + SQUARE_SIZE / 2,
            row * SQUARE_SIZE + SQUARE_SIZE / 2,
            SQUARE_SIZE / 4,
            0,
            Math.PI * 2
        );
        ctx.fill();
    });
}

// Обновление информации о игре
function updateGameInfo() {
    statusElement.textContent = gameState.currentPlayer === 1 ? 
        'Ваш ход (белые)' : 'Ход бота (черные)';
    scoreElement.textContent = `Белые: ${gameState.whitePieces} | Черные: ${gameState.blackPieces}`;
}

// Получение возможных ходов для шашки
function getPossibleMoves(row, col) {
    const moves = [];
    const piece = gameState.board[row][col];
    const direction = piece > 0 ? 1 : -1; // Направление движения
    
    // Проверка обычных ходов
    for (let dc = -1; dc <= 1; dc += 2) {
        const newRow = row + direction;
        const newCol = col + dc;
        
        if (newRow >= 0 && newRow < BOARD_SIZE &&
            newCol >= 0 && newCol < BOARD_SIZE &&
            gameState.board[newRow][newCol] === 0) {
            moves.push([newRow, newCol]);
        }
    }
    
    return moves;
}

// Обработка клика по доске
async function handleClick(event) {
    if (gameState.currentPlayer !== 1 || gameState.gameOver) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const col = Math.floor(x / SQUARE_SIZE);
    const row = Math.floor(y / SQUARE_SIZE);
    
    // Если шашка уже выбрана, пытаемся сделать ход
    if (gameState.selectedPiece) {
        const [selectedRow, selectedCol] = gameState.selectedPiece;
        
        // Проверяем, является ли клик допустимым ходом
        const isValidMove = gameState.possibleMoves.some(
            ([r, c]) => r === row && c === col
        );
        
        if (isValidMove) {
            // Формируем ход в формате "a3b4"
            const move = `${String.fromCharCode(97 + selectedCol)}${8-selectedRow}${String.fromCharCode(97 + col)}${8-row}`;
            
            try {
                const response = await fetch('/make_move', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ move })
                });
                
                const data = await response.json();
                if (data.status === 'ok') {
                    await updateBoardState();
                    if (!gameState.gameOver) {
                        await botMove();
                    }
                }
            } catch (error) {
                console.error('Error making move:', error);
            }
            return;
        }
    }
    
    // Выбираем шашку, если она принадлежит текущему игроку
    if (gameState.board[row][col] * gameState.currentPlayer > 0) {
        gameState.selectedPiece = [row, col];
        gameState.possibleMoves = getPossibleMoves(row, col);
        drawBoard();
    }
}

// Ход бота
async function botMove() {
    try {
        const response = await fetch('/bot_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        if (data.status === 'ok') {
            await updateBoardState();
        }
    } catch (error) {
        console.error('Error with bot move:', error);
    }
}

// Новая игра
async function newGame() {
    await initializeGame();
    gameState = {
        board: Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(0)),
        currentPlayer: 1,
        selectedPiece: null,
        possibleMoves: [],
        gameOver: false
    };
    await updateBoardState();
}

// Инициализация игры
canvas.addEventListener('click', handleClick);
restartButton.addEventListener('click', newGame);

// Первая отрисовка
newGame();