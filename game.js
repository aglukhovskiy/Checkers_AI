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
        console.log('Initializing new game...');
        const response = await fetch('http://localhost:5000/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        console.log('New game response status:', response.status);
        const data = await response.json();
        console.log('New game response data:', data);
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
        console.log('Updating board state...');
        const response = await fetch('http://localhost:5000/get_state', {
            method: 'GET'
        });
        console.log('Get state response status:', response.status);
        const data = await response.json();
        console.log('Get state response data:', data);
        
        if (data.status === 'ok') {
            gameState.board = data.board.field;
            gameState.currentPlayer = data.board.current_player === 'white' ? 1 : -1;
            gameState.gameOver = data.board.game_over;
            
            // Логируем состояние доски для отладки
            console.log('Board state:');
            gameState.board.forEach((row, i) => {
                console.log(`Row ${i}:`, row);
            });
            
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
            
            // Рисуем шашку (только на темных клетках)
            if (gameState.board[row][col] !== 0 && (row + col) % 2 === 1) {
                const piece = gameState.board[row][col];
                const isWhite = piece > 0;
                const isKing = Math.abs(piece) === 2;
                
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
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Рисуем корону для дамки
                if (isKing) {
                    ctx.fillStyle = isWhite ? BLACK_PIECE_COLOR : WHITE_PIECE_COLOR;
                    ctx.font = `${SQUARE_SIZE / 4}px Arial`;
                    ctx.textAlign = 'center';
                    ctx.fillText('♔',
                        col * SQUARE_SIZE + SQUARE_SIZE / 2,
                        row * SQUARE_SIZE + SQUARE_SIZE / 2 + SQUARE_SIZE / 8
                    );
                }
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
        
        // Разные цвета для обычных ходов и взятий
        ctx.fillStyle = 'rgba(173, 216, 230, 0.5)'; // голубой для обычных ходов
        ctx.beginPath();
        ctx.arc(
            col * SQUARE_SIZE + SQUARE_SIZE / 2,
            row * SQUARE_SIZE + SQUARE_SIZE / 2,
            SQUARE_SIZE / 4,
            0,
            Math.PI * 2
        );
        ctx.fill();
        
        // Красная обводка для ходов со взятием
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

// Обновление информации о игре
function updateGameInfo() {
    statusElement.textContent = gameState.currentPlayer === 1 ? 
        'Ваш ход (белые)' : 'Ход бота (черные)';
    scoreElement.textContent = `Белые: ${gameState.whitePieces} | Черные: ${gameState.blackPieces}`;
}

// Получение возможных ходов для шашки с сервера
async function getPossibleMoves(row, col) {
    try {
        console.log('JS: Запрос ходов для шашки на', row, col);
        const response = await fetch('http://localhost:5000/get_possible_moves', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ row, col })
        });
        
        const data = await response.json();
        console.log('JS: Ответ сервера:', data);
        
        if (data.status === 'ok') {
            // Проверим каждый ход на валидность (должен быть на темной клетке)
            if (data.moves && data.moves.length > 0) {
                data.moves.forEach(move => {
                    const [moveRow, moveCol] = move;
                    const isDarkCell = (moveRow + moveCol) % 2 === 1;
                    console.log(`JS: Ход на [${moveRow},${moveCol}] - темная клетка: ${isDarkCell}`);
                });
            }
            
            // Обработка обязательного взятия
            if (data.capture_required) {
                if (data.moves.length === 0) {
                    statusElement.textContent = 'Обязательное взятие другой шашкой!';
                } else if (data.all_captures) {
                    statusElement.textContent = 'Обязательное взятие! Выберите шашку, которая может брать.';
                } else {
                    statusElement.textContent = 'Обязательное взятие!';
                }
            } else {
                // Сбрасываем статус, если нет обязательного взятия
                updateGameInfo();
            }
            return data.moves;
        }
        return [];
    } catch (error) {
        console.error('Error getting moves from server:', error);
        return [];
    }
}

// Обработка клика по доске
async function handleClick(event) {
    console.log('Canvas clicked!');
    console.log('Current player:', gameState.currentPlayer, 'Game over:', gameState.gameOver);
    
    if (gameState.currentPlayer !== 1 || gameState.gameOver) {
        console.log('Not player turn or game over');
        return;
    }
    
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const col = Math.floor(x / SQUARE_SIZE);
    const row = Math.floor(y / SQUARE_SIZE);
    
    console.log('Clicked at row:', row, 'col:', col);
    console.log('Board value at clicked position:', gameState.board[row][col]);
    
    // Если шашка уже выбрана, пытаемся сделать ход
    if (gameState.selectedPiece) {
        console.log('Piece already selected:', gameState.selectedPiece);
        console.log('Possible moves:', gameState.possibleMoves);
        
        const [selectedRow, selectedCol] = gameState.selectedPiece;
        
        // Проверяем, является ли клик допустимым ходом
        const isValidMove = gameState.possibleMoves.some(
            ([r, c]) => r === row && c === col
        );
        
        console.log('Is valid move:', isValidMove);
        
        if (isValidMove) {
            // Формируем ход в формате "a3b4"
            const fromColChar = String.fromCharCode(97 + selectedCol);
            const fromRowNum = selectedRow + 1;  // Прямое соответствие: row=0 -> "1", row=7 -> "8"
            const toColChar = String.fromCharCode(97 + col);
            const toRowNum = row + 1;           // Прямое соответствие: row=0 -> "1", row=7 -> "8"
            const move = `${fromColChar}${fromRowNum}${toColChar}${toRowNum}`;
            
            console.log('JS: Формируем ход:', {
                selected: [selectedRow, selectedCol],
                target: [row, col],
                move: move,
                details: `${fromColChar}${fromRowNum}->${toColChar}${toRowNum}`
            });
            
            try {
                const response = await fetch('http://localhost:5000/make_move', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ move })
                });
                
                const data = await response.json();
                console.log('Move response:', data);
                
                if (data.status === 'ok') {
                    gameState.selectedPiece = null;
                    gameState.possibleMoves = [];
                    await updateBoardState();
                    
                    if (!gameState.gameOver && gameState.currentPlayer === -1) {
                        setTimeout(async () => {
                            await botMove();
                        }, 500);
                    }
                } else {
                    console.error('Move failed:', data);
                    statusElement.textContent = 'Неверный ход!';
                }
            } catch (error) {
                console.error('Error making move:', error);
                statusElement.textContent = 'Ошибка соединения с сервером';
            }
            return;
        } else {
            console.log('Invalid move - clearing selection');
        }
    }
    
    // Выбираем шашку, если она принадлежит текущему игроку и находится на темной клетке
    const isOwnPiece = gameState.board[row][col] * gameState.currentPlayer > 0;
    const isDarkCell = (row + col) % 2 === 1;
    console.log('Is own piece:', isOwnPiece, 'Is dark cell:', isDarkCell);
    
    if (isOwnPiece && isDarkCell) {
        console.log('Selecting piece at:', row, col);
        gameState.selectedPiece = [row, col];
        
        // Получаем возможные ходы с сервера
        gameState.possibleMoves = await getPossibleMoves(row, col);
        console.log('Possible moves for selected piece:', gameState.possibleMoves);
        drawBoard();
    } else {
        console.log('Clearing selection - not own piece or not dark cell');
        // Сбрасываем выбор, если кликнули не на свою шашку
        gameState.selectedPiece = null;
        gameState.possibleMoves = [];
        drawBoard();
    }
}

// Ход бота
async function botMove() {
    try {
        console.log('Requesting bot move...');
        const response = await fetch('http://localhost:5000/bot_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        console.log('Bot move response:', data);
        
        if (data.status === 'ok') {
            // Обновляем состояние доски из ответа
            if (data.board) {
                gameState.board = data.board.field;
                gameState.currentPlayer = data.board.current_player === 'white' ? 1 : -1;
                gameState.gameOver = data.board.game_over;
                updateGameInfo();
                drawBoard();
            } else {
                await updateBoardState();
            }
        } else {
            console.error('Bot move failed:', data);
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