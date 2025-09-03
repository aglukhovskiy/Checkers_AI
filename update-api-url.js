// Скрипт для автоматического обновления URL API при деплое
const fs = require('fs');
const path = require('path');

// Получаем URL бэкенда из переменной окружения или аргумента
const backendUrl = process.env.BACKEND_URL || process.argv[2] || 'https://your-backend-url.railway.app';

console.log(`Updating API URL to: ${backendUrl}`);

// Читаем game.js
const gameJsPath = path.join(__dirname, 'game.js');
let content = fs.readFileSync(gameJsPath, 'utf8');

// Заменяем URL API
const updatedContent = content.replace(
  /const API_URL = .*?;/,
  `const API_URL = '${backendUrl}';`
);

// Записываем обновленный файл
fs.writeFileSync(gameJsPath, updatedContent, 'utf8');

console.log('API URL updated successfully!');