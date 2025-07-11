# AI-аналитик банковских опросов 🤖

Telegram-бот для анализа данных опросов банковских клиентов с использованием искусственного интеллекта.

## 🚀 Возможности

- 📊 Создание красивых графиков и диаграмм
- 🎯 Быстрый анализ ключевых метрик
- 🤖 Умные ответы на вопросы с помощью GPT-4
- 📈 Статистический анализ данных
- 💡 Генерация рекомендаций
- 👥 Демографический анализ
- ⭐ Анализ качества обслуживания

## 🛠 Технологии

- **Python 3.8+**
- **python-telegram-bot** - Telegram Bot API
- **pandas** - обработка данных
- **matplotlib/seaborn** - визуализация
- **gspread** - работа с Google Sheets
- **OpenAI GPT-4** - AI-аналитика
- **Google Sheets API** - источник данных

## 📋 Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/bank-survey-analyzer.git
cd bank-survey-analyzer
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Настройте переменные окружения:
```bash
cp .env.example .env
```

4. Заполните `.env` файл:
```
TELEGRAM_TOKEN=your_telegram_bot_token
SHEET_ID=your_google_sheet_id
OPENAI_API_KEY=your_openai_api_key
```

5. Добавьте файл Google Service Account:
- Скачайте JSON файл из Google Cloud Console
- Переименуйте в `medical-462021-78bf30c680aa.json`
- Поместите в корневую папку проекта

## 🚀 Запуск

```bash
python test.py
```

## 📱 Использование

После запуска бота в Telegram:

### Основные команды:
- `/start` - начало работы с ботом
- `📊 Полный отчет` - полный анализ опроса
- `🎯 Быстрый анализ` - ключевые метрики
- `👥 Гендерный состав` - анализ по полу
- `📈 Возрастная статистика` - анализ по возрасту
- `🏦 Топ банков` - популярные банки
- `💼 Цели посещения` - анализ целей
- `⭐ Оценки качества` - качество обслуживания
- `⏰ Время ожидания` - анализ очередей

### Произвольные запросы:
- "Какие банки самые популярные?"
- "Сравни мужчин и женщин"
- "Анализ проблем клиентов"
- "График по возрасту"
- "Качество обслуживания"

## 📊 Структура данных

Бот анализирует опросы банковских клиентов по следующим вопросам:
- Демографические данные (пол, возраст)
- Выбор банка и отделения
- Цели посещения
- Оценки качества обслуживания
- Время ожидания
- Проблемы и жалобы
- Готовность рекомендовать

## 🔧 Настройка

### Google Sheets
1. Создайте Google Sheet с данными опроса
2. Настройте Google Service Account
3. Предоставьте доступ к таблице
4. Укажите ID таблицы в `.env`

### Telegram Bot
1. Создайте бота через @BotFather
2. Получите токен
3. Укажите токен в `.env`

### OpenAI API
1. Зарегистрируйтесь на OpenAI
2. Получите API ключ
3. Укажите ключ в `.env`

## 📁 Структура проекта

```
bank-survey-analyzer/
├── test.py                 # Основной файл бота
├── requirements.txt        # Зависимости
├── .env.example           # Пример переменных окружения
├── .gitignore             # Исключения для Git
├── README.md              # Документация
└── medical-462021-78bf30c680aa.json  # Google Service Account (не в Git)
```

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Создайте Pull Request

## 📄 Лицензия

MIT License

## 📞 Поддержка

Если у вас есть вопросы или проблемы, создайте Issue в репозитории. 