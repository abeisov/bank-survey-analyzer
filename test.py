import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
import re
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
from difflib import get_close_matches
import openai
import io
import seaborn as sns

load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
SHEET_ID = os.getenv('SHEET_ID')
GOOGLE_JSON = os.getenv('GOOGLE_JSON_PATH', 'medical-462021-78bf30c680aa.json')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Создаём клиента OpenAI (глобально)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

COLUMN_SYNONYMS = {
    "тип обращения": "С какой целью вы посетили отделение банка?",
    "цель": "С какой целью вы посетили отделение банка?",
    "очередь": "Сколько времени вы обычно ждете в очереди до получения обслуживания?",
    "банк": "Назовите банк, отделение которого вы посещали недавно.",
    "отделение": "Назовите банк, отделение которого вы посещали недавно.",
    "расположение": "Как вы оцениваете удобство расположения отделения банка?",
    "вежливость": "Насколько вежливы и доброжелательны сотрудники банка?",
    "компетентность": "Как вы оцениваете компетентность сотрудников в решении вопросов?",
    "доступность": "Как вы оцениваете доступность информации о банковских услугах в отделении?",
    "терминал": "Удобно ли вам пользоваться электронными терминалами или приложением?",
    "рекомендация": "Порекомендовали бы вы это отделение банка своим друзьям и знакомым?",
    "понятно": "Насколько понятно сотрудники объясняют условия банковских продуктов (кредиты, вклады и т.п.)?",
    "чистота": "Как вы оцениваете чистоту и комфорт в помещении отделения?",
    "проблем": "Были ли у вас случаи, когда ваш вопрос не решился?",
    "жалоб": "Были ли у вас случаи, когда ваш вопрос не решился?",
    "пол": "Укажите ваш пол.",
    "gender": "Укажите ваш пол.",
    "возраст": "Укажите ваш возраст.",
}

def get_df_from_gsheet():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Сначала пробуем получить из переменной окружения
        google_credentials = os.getenv('GOOGLE_CREDENTIALS')
        if google_credentials:
            import json
            creds = ServiceAccountCredentials.from_json_keyfile_dict(
                json.loads(google_credentials), scope
            )
        else:
            # Fallback к файлу (для локальной разработки)
            if not os.path.exists(GOOGLE_JSON):
                print(f"Файл {GOOGLE_JSON} не найден и GOOGLE_CREDENTIALS не установлен")
                return pd.DataFrame()
            creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_JSON, scope)
        
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).worksheet("Ответы на форму")
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Ошибка при получении данных из Google Sheets: {e}")
        return pd.DataFrame()

def extract_numeric(series):
    return pd.to_numeric(series.astype(str).str.extract('(\d+)')[0], errors='coerce')

def find_column_by_synonym(df, text):
    text = text.lower()
    for short, real in COLUMN_SYNONYMS.items():
        if short in text and real in df.columns:
            return real
    return None

def find_column_fuzzy(df, text):
    candidates = list(df.columns)
    text_clean = re.sub(r'[^а-яa-z0-9 ]', '', text.lower())
    match = get_close_matches(text_clean, [re.sub(r'[^а-яa-z0-9 ]', '', c.lower()) for c in candidates], n=1, cutoff=0.3)
    if match:
        for c in candidates:
            if match[0] in re.sub(r'[^а-яa-z0-9 ]', '', c.lower()):
                return c
    words = text_clean.split()
    for c in candidates:
        col_clean = re.sub(r'[^а-яa-z0-9 ]', '', c.lower())
        if any(w in col_clean for w in words if len(w) > 2):
            return c
    return None

def plot_pie(df, column, title):
    plt.style.use('seaborn-v0_8-darkgrid')
    data = df[column].value_counts()
    labels = [str(x)[:18] + ('...' if len(str(x)) > 18 else '') for x in data.index]
    colors = sns.color_palette('Set3', len(data))
    plt.figure(figsize=(5, 5))
    wedges, texts, autotexts = plt.pie(
        data.values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white'}
    )
    plt.title(f'🟢 {title}', fontsize=17, fontweight='bold', pad=15)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def plot_hist(df, column, title):
    plt.style.use('seaborn-v0_8-darkgrid')
    data = extract_numeric(df[column]).dropna()
    plt.figure(figsize=(8, 5))
    ax = sns.histplot(data, bins=range(int(data.min()), int(data.max())+5, 5), color='#4C72B0', edgecolor='black', alpha=0.85)
    ax.set_title(f'📈 {title}', fontsize=17, fontweight='bold', pad=15)
    ax.set_xlabel(column, fontsize=13, fontweight='bold')
    ax.set_ylabel('Количество', fontsize=13, fontweight='bold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    sns.despine()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def plot_bar(df, column, title):
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(9, 5))
    data = df[column].value_counts()
    # Обрезаем длинные подписи
    labels = [str(x)[:18] + ('...' if len(str(x)) > 18 else '') for x in data.index]
    ax = sns.barplot(x=labels, y=data.values, palette='Set2', edgecolor='black')
    ax.set_title(f'📊 {title}', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel(column, fontsize=13, fontweight='bold')
    ax.set_ylabel('Количество', fontsize=13, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    # Подписи значений
    for i, v in enumerate(data.values):
        ax.text(i, v + max(data.values)*0.01, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333')
    sns.despine()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def ask_openai(question, df):
    # Подготавливаем статистику по всем колонкам для лучшего понимания данных
    stats = {}
    for col in df.columns:
        if df[col].dtype == 'object':  # Текстовые данные
            value_counts = df[col].value_counts()
            if not value_counts.empty:
                stats[col] = {
                    'type': 'categorical',
                    'total': len(df[col].dropna()),
                    'unique_values': len(value_counts),
                    'top_values': value_counts.head(3).to_dict()
                }
        else:  # Числовые данные
            numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if not numeric_data.empty:
                stats[col] = {
                    'type': 'numeric',
                    'total': len(numeric_data),
                    'mean': numeric_data.mean(),
                    'median': numeric_data.median(),
                    'min': numeric_data.min(),
                    'max': numeric_data.max()
                }
    
    # Примеры данных (только первые 5 строк для экономии токенов)
    sample_data = df.head(5).to_dict('records')
    
    prompt = (
        f"Ты дружелюбный аналитик-помощник для анализа опросов банковских клиентов. "
        f"Отвечай на русском языке, будь общительным и полезным.\n\n"
        f"Данные опроса:\n"
        f"- Всего анкет: {len(df)}\n"
        f"- Вопросы в опросе: {', '.join(df.columns)}\n\n"
        f"Статистика по колонкам:\n"
    )
    
    for col, stat in stats.items():
        if stat['type'] == 'categorical':
            top_items = ', '.join([f"{k} ({v})" for k, v in list(stat['top_values'].items())[:3]])
            prompt += f"- {col}: {stat['total']} ответов, топ: {top_items}\n"
        else:
            prompt += f"- {col}: среднее {stat['mean']:.1f}, медиана {stat['median']:.1f}, диапазон {stat['min']}-{stat['max']}\n"
    
    prompt += f"\nПримеры ответов:\n"
    for i, record in enumerate(sample_data, 1):
        prompt += f"Анкета {i}: {str(record)[:200]}...\n"
    
    prompt += f"\nВопрос пользователя: {question}\n\n"
    prompt += (
        f"Инструкции:\n"
        f"1. Отвечай дружелюбно и разговорно\n"
        f"2. Если можешь ответить по данным - используй статистику выше\n"
        f"3. Если данных недостаточно - скажи об этом честно\n"
        f"4. Предлагай дополнительные вопросы или графики\n"
        f"5. Будь полезным и информативным\n"
        f"6. Отвечай на русском языке"
    )
    
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "Ты дружелюбный аналитик-помощник для анализа банковских опросов. Отвечай разговорно, полезно и на русском языке."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return completion.choices[0].message.content

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        ['📊 Полный отчет', '🎯 Быстрый анализ'],
        ['👥 Гендерный состав', '📈 Возрастная статистика'],
        ['🏦 Топ банков', '💼 Цели посещения'],
        ['⭐ Оценки качества', '⏰ Время ожидания'],
        ['🔍 Детальный анализ', '📋 Все вопросы']
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    
    welcome_text = (
        "🤖 *Добро пожаловать в AI-аналитик банковских опросов!*\n\n"
        "Я помогу вам проанализировать данные опроса банковских клиентов:\n\n"
        "📊 *Что я умею:*\n"
        "• Создавать красивые графики и диаграммы\n"
        "• Проводить статистический анализ\n"
        "• Отвечать на любые вопросы по данным\n"
        "• Давать умные рекомендации\n\n"
        "💡 *Примеры запросов:*\n"
        "• \"Какие банки самые популярные?\"\n"
        "• \"Сравни мужчин и женщин\"\n"
        "• \"Анализ проблем клиентов\"\n"
        "• \"График по возрасту\"\n"
        "• \"Качество обслуживания\"\n\n"
        "🎯 *Используйте кнопки или пишите свои вопросы!*"
    )
    
    await update.message.reply_text(welcome_text, reply_markup=reply_markup, parse_mode='Markdown')

def get_stats_for_gpt(df):
    """Генерирует краткую статистику по всем ключевым вопросам для передачи в GPT"""
    stats = ""
    for col in df.columns:
        val = df[col].value_counts()
        if len(val) > 0:
            total = val.sum()
            top = val.idxmax()
            top_count = val.max()
            percent = (top_count / total) * 100
            stats += f"\n- {col}: всего {total}, топ: '{top}' ({top_count}, {percent:.1f}%)"
            if len(val) > 1:
                stats += f", другие: " + ", ".join([f"{k} ({v})" for k, v in val.head(3).items()])
    return stats

def smart_analytics_gpt(user_query, df):
    stats = get_stats_for_gpt(df)
    prompt = f'''
Ты — эксперт по анализу опросов. Вот статистика по данным:{stats}
Пользователь спрашивает: {user_query}

Отвечай структурировано и дружелюбно, используй эмодзи для каждого смыслового блока:
- 📝 Вывод
- 📊 Ключевые цифры
- 🔍 Причины/объяснения
- 💡 Рекомендации
- 🚀 Следующий шаг
Если вопрос сравнения — сравни группы с эмодзи. Если вопрос анализа — дай причины и советы. Если не хватает данных — честно скажи. Всегда предлагай следующий шаг для пользователя. Пиши кратко, понятно, по делу, на русском языке.
'''
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "Ты эксперт по анализу опросов, отвечай кратко, по делу, дружелюбно, на русском."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700,
        temperature=0.7
    )
    return response.choices[0].message.content

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()
    
    # Проверяем переменные окружения
    if not TELEGRAM_TOKEN or not SHEET_ID or not OPENAI_API_KEY:
        await update.message.reply_text("Ошибка: не настроены переменные окружения (TELEGRAM_TOKEN, SHEET_ID, OPENAI_API_KEY)")
        return
    
    df = get_df_from_gsheet()
    
    # Проверяем, что данные получены
    if df.empty:
        await update.message.reply_text("Ошибка: не удалось получить данные из таблицы")
        return

    # --- Кнопки ---
    if text == '📊 полный отчет' or text == 'полный отчет':
        summary = analyze_survey(df)
        if len(summary) > 4000:
            parts = []
            current_part = ""
            lines = summary.split('\n')
            for line in lines:
                if len(current_part + line + '\n') > 4000:
                    parts.append(current_part)
                    current_part = line + '\n'
                else:
                    current_part += line + '\n'
            if current_part:
                parts.append(current_part)
            for i, part in enumerate(parts, 1):
                if len(parts) > 1:
                    header = f"📊 ПОЛНЫЙ ОТЧЕТ (часть {i}/{len(parts)})\n{'='*30}\n\n"
                    await update.message.reply_text(header + part)
                else:
                    await update.message.reply_text(part)
        else:
            await update.message.reply_text(summary)
        return
    elif text == '🎯 быстрый анализ' or text == 'быстрый анализ':
        quick_analysis = generate_quick_analysis(df)
        await update.message.reply_text(quick_analysis, parse_mode='Markdown')
        return
    elif text == '👥 гендерный состав' or text == 'гендерный состав':
        col = COLUMN_SYNONYMS['пол']
        freq = df[col].value_counts()
        if len(freq) > 0:
            buf = plot_pie(df, col, 'Гендерный состав')
            if buf:
                await update.message.reply_photo(buf)
                total = freq.sum()
                male_count = freq.get('Мужской', 0)
                female_count = freq.get('Женский', 0)
                stats_text = f"👥 *ГЕНДЕРНЫЙ СОСТАВ ОПРОШЕННЫХ*\n\n"
                stats_text += f"📊 *Статистика:*\n"
                stats_text += f"• Всего ответов: {total}\n"
                stats_text += f"• Мужчин: {male_count} ({male_count/total*100:.1f}%)\n"
                stats_text += f"• Женщин: {female_count} ({female_count/total*100:.1f}%)\n\n"
                if male_count > female_count:
                    stats_text += f"🏆 Больше мужчин на {male_count - female_count} человек"
                elif female_count > male_count:
                    stats_text += f"🏆 Больше женщин на {female_count - male_count} человек"
                else:
                    stats_text += f"⚖️ Равное количество мужчин и женщин"
                await update.message.reply_text(stats_text, parse_mode='Markdown')
            else:
                await update.message.reply_text("Не удалось создать график - нет данных")
        else:
            await update.message.reply_text("Нет данных о поле респондентов")
        return
    elif text == '📈 возрастная статистика' or text == 'возрастная статистика':
        col = COLUMN_SYNONYMS['возраст']
        numeric_data = extract_numeric(df[col]).dropna()
        
        if len(numeric_data) > 0:
            buf = plot_hist(df, col, 'Распределение по возрасту')
            if buf:
                await update.message.reply_photo(buf)
                
                # Добавляем текстовую статистику
                stats_text = f"📊 *РАСПРЕДЕЛЕНИЕ ПО ВОЗРАСТУ*\n\n"
                stats_text += f"📈 *Статистика:*\n"
                stats_text += f"• Всего ответов: {len(numeric_data)}\n"
                stats_text += f"• Средний возраст: {numeric_data.mean():.1f} лет\n"
                stats_text += f"• Медианный возраст: {numeric_data.median():.1f} лет\n"
                stats_text += f"• Минимальный возраст: {numeric_data.min()} лет\n"
                stats_text += f"• Максимальный возраст: {numeric_data.max()} лет\n\n"
                
                # Топ возрастов
                age_counts = numeric_data.value_counts().head(3)
                stats_text += f"🏆 *Самые частые возрасты:*\n"
                for i, (age, count) in enumerate(age_counts.items(), 1):
                    stats_text += f"{i}. {age} лет: {count} человек\n"
                
                await update.message.reply_text(stats_text, parse_mode='Markdown')
            else:
                await update.message.reply_text("Не удалось создать график - нет данных")
        else:
            await update.message.reply_text("Нет числовых данных о возрасте")
        return
        
    elif text == '🏦 топ банков' or text == 'топ банков':
        col = COLUMN_SYNONYMS['банк']
        freq = df[col].value_counts()
        if len(freq) > 0:
            buf = plot_bar(df, col, 'Топ банков')
            if buf:
                await update.message.reply_photo(buf)
                # Аналитика по банкам
                analysis = smart_analytics_gpt('Дай краткий анализ по топу банков', df)
                await update.message.reply_text(analysis)
            else:
                await update.message.reply_text("Не удалось создать график - нет данных")
        else:
            await update.message.reply_text("Нет данных о банках")
        return
        
    elif text == '💼 цели посещения' or text == 'цели посещения':
        col = COLUMN_SYNONYMS['тип обращения']
        freq = df[col].value_counts()
        if len(freq) > 0:
            buf = plot_bar(df, col, 'Цели посещения банка')
            if buf:
                await update.message.reply_photo(buf)
                # Аналитика по целям
                analysis = smart_analytics_gpt('Дай краткий анализ по целям посещения банка', df)
                await update.message.reply_text(analysis)
            else:
                await update.message.reply_text("Не удалось создать график - нет данных")
        else:
            await update.message.reply_text("Нет данных о целях посещения")
        return
        
    elif text == '⭐ оценки качества' or text == 'оценки качества':
        quality_analysis = analyze_quality_metrics(df)
        await update.message.reply_text(quality_analysis, parse_mode='Markdown')
        return
        
    elif text == '⏰ время ожидания' or text == 'время ожидания':
        col = COLUMN_SYNONYMS['очередь']
        freq = df[col].value_counts()
        if len(freq) > 0:
            buf = plot_bar(df, col, 'Время ожидания в очереди')
            if buf:
                await update.message.reply_photo(buf)
                analysis = smart_analytics_gpt('Дай краткий анализ по времени ожидания в очереди', df)
                await update.message.reply_text(analysis)
            else:
                await update.message.reply_text("Не удалось создать график - нет данных")
        else:
            await update.message.reply_text("Нет данных о времени ожидания")
        return
        
    elif text == '🔍 детальный анализ' or text == 'детальный анализ':
        detailed_analysis = generate_detailed_analysis(df)
        await update.message.reply_text(detailed_analysis, parse_mode='Markdown')
        return
        
    elif text == '📋 все вопросы' or text == 'все вопросы':
        questions_list = generate_questions_list(df)
        await update.message.reply_text(questions_list, parse_mode='Markdown')
        return

    # Старые кнопки для совместимости
    if text == 'отчет по опросу':
        summary = analyze_survey(df)
        
        # Разбиваем длинный отчет на части
        if len(summary) > 4000:
            parts = []
            current_part = ""
            lines = summary.split('\n')
            
            for line in lines:
                if len(current_part + line + '\n') > 4000:
                    parts.append(current_part)
                    current_part = line + '\n'
                else:
                    current_part += line + '\n'
            
            if current_part:
                parts.append(current_part)
            
            for i, part in enumerate(parts, 1):
                if len(parts) > 1:
                    header = f"📊 ОТЧЕТ ПО ОПРОСУ (часть {i}/{len(parts)})\n{'='*30}\n\n"
                    await update.message.reply_text(header + part)
                else:
                    await update.message.reply_text(part)
        else:
            await update.message.reply_text(summary)
        return
    elif text == 'гендерный pie chart':
        col = COLUMN_SYNONYMS['пол']
        freq = df[col].value_counts()
        
        if len(freq) > 0:
            buf = plot_pie(df, col, 'Гендерный состав')
            if buf:
                await update.message.reply_photo(buf)
                
                # Добавляем текстовую статистику
                total = freq.sum()
                male_count = freq.get('Мужской', 0)
                female_count = freq.get('Женский', 0)
                
                stats_text = f"👥 ГЕНДЕРНЫЙ СОСТАВ ОПРОШЕННЫХ\n\n"
                stats_text += f"📊 Статистика:\n"
                stats_text += f"• Всего ответов: {total}\n"
                stats_text += f"• Мужчин: {male_count} ({male_count/total*100:.1f}%)\n"
                stats_text += f"• Женщин: {female_count} ({female_count/total*100:.1f}%)\n\n"
                
                if male_count > female_count:
                    stats_text += f"🏆 Больше мужчин на {male_count - female_count} человек"
                elif female_count > male_count:
                    stats_text += f"🏆 Больше женщин на {female_count - male_count} человек"
                else:
                    stats_text += f"⚖️ Равное количество мужчин и женщин"
                
                await update.message.reply_text(stats_text)
            else:
                await update.message.reply_text("Не удалось создать график - нет данных")
        else:
            await update.message.reply_text("Нет данных о поле респондентов")
        return
    elif text == 'возраст: histogram':
        col = COLUMN_SYNONYMS['возраст']
        numeric_data = extract_numeric(df[col]).dropna()
        
        if len(numeric_data) > 0:
            buf = plot_hist(df, col, 'Распределение по возрасту')
            if buf:
                await update.message.reply_photo(buf)
                
                # Добавляем текстовую статистику
                stats_text = f"📊 РАСПРЕДЕЛЕНИЕ ПО ВОЗРАСТУ\n\n"
                stats_text += f"📈 Статистика:\n"
                stats_text += f"• Всего ответов: {len(numeric_data)}\n"
                stats_text += f"• Средний возраст: {numeric_data.mean():.1f} лет\n"
                stats_text += f"• Медианный возраст: {numeric_data.median():.1f} лет\n"
                stats_text += f"• Минимальный возраст: {numeric_data.min()} лет\n"
                stats_text += f"• Максимальный возраст: {numeric_data.max()} лет\n\n"
                
                # Топ возрастов
                age_counts = numeric_data.value_counts().head(3)
                stats_text += f"🏆 Самые частые возрасты:\n"
                for i, (age, count) in enumerate(age_counts.items(), 1):
                    stats_text += f"{i}. {age} лет: {count} человек\n"
                
                await update.message.reply_text(stats_text)
            else:
                await update.message.reply_text("Не удалось создать график - нет данных")
        else:
            await update.message.reply_text("Нет числовых данных о возрасте")
        return
    elif text == 'тип обращения: bar chart':
        col = COLUMN_SYNONYMS['тип обращения']
        buf = plot_bar(df, col, 'Типы обращений')
        if buf:
            await update.message.reply_photo(buf)
            analysis = smart_analytics_gpt('Дай краткий анализ по типам обращений', df)
            await update.message.reply_text(analysis)
        else:
            await update.message.reply_text("Не удалось создать график - нет данных")
        return
    elif text == 'топ банков: bar chart':
        col = COLUMN_SYNONYMS['банк']
        buf = plot_bar(df, col, 'Топ посещаемых банков')
        if buf:
            await update.message.reply_photo(buf)
            analysis = smart_analytics_gpt('Дай краткий анализ по топу банков', df)
            await update.message.reply_text(analysis)
        else:
            await update.message.reply_text("Не удалось создать график - нет данных")
        return

    # --- Любой другой текстовый запрос ---
    try:
        reply = smart_analytics_gpt(update.message.text, df)
        await update.message.reply_text(reply)
    except Exception as e:
        await update.message.reply_text("Не смог получить умный ответ. Попробуйте иначе!\nОшибка: " + str(e))

def analyze_survey(df):
    summary = f"📊 ОТЧЕТ ПО ОПРОСУ БАНКОВСКИХ КЛИЕНТОВ\n"
    summary += f"{'='*50}\n\n"
    summary += f"📈 Общая статистика:\n"
    summary += f"• Всего анкет: {len(df)}\n"
    summary += f"• Количество вопросов: {len(df.columns)}\n\n"
    
    # Пропускаем колонку с отметкой времени
    relevant_columns = [col for col in df.columns if 'отметка времени' not in col.lower() and 'timestamp' not in col.lower()]
    
    summary += f"🔍 Основные результаты:\n\n"
    
    for i, col in enumerate(relevant_columns, 1):
        val = df[col].value_counts()
        if val.shape[0] > 1:
            total = val.sum()
            top_answer = val.idxmax()
            top_count = val.max()
            top_percent = (top_count / total) * 100
            
            # Сокращаем длинные названия колонок
            short_col = col
            if len(col) > 50:
                short_col = col[:47] + "..."
            
            summary += f"{i}. {short_col}\n"
            summary += f"   📊 Всего ответов: {total}\n"
            summary += f"   🏆 Топ ответ: '{top_answer}' ({top_count} раз, {top_percent:.1f}%)\n"
            
            if len(val) <= 4:
                summary += f"   📋 Все ответы:\n"
                for answer, count in val.items():
                    percent = (count / total) * 100
                    summary += f"      • {answer}: {count} ({percent:.1f}%)\n"
            else:
                summary += f"   📋 Топ-3 ответа:\n"
                for j, (answer, count) in enumerate(val.head(3).items(), 1):
                    percent = (count / total) * 100
                    summary += f"      {j}. {answer}: {count} ({percent:.1f}%)\n"
            
            summary += "\n"
    
    summary += f"💡 Хотите увидеть графики? Напишите:\n"
    summary += f"• 'график по банкам'\n"
    summary += f"• 'статистика по возрасту'\n"
    summary += f"• 'анализ проблем'\n"
    summary += f"• 'сравнение мужчин и женщин'"
    
    return summary

def generate_quick_analysis(df):
    """Генерирует быстрый анализ ключевых метрик"""
    analysis = f"🎯 *БЫСТРЫЙ АНАЛИЗ КЛЮЧЕВЫХ МЕТРИК*\n\n"
    
    # Анализ банков
    bank_col = COLUMN_SYNONYMS.get('банк')
    if bank_col and bank_col in df.columns:
        bank_freq = df[bank_col].value_counts()
        if len(bank_freq) > 0:
            top_bank = bank_freq.idxmax()
            top_count = bank_freq.max()
            total_banks = bank_freq.sum()
            analysis += f"🏦 *Топ банк:* {top_bank} ({top_count}/{total_banks} клиентов)\n\n"
    
    # Анализ возраста
    age_col = COLUMN_SYNONYMS.get('возраст')
    if age_col and age_col in df.columns:
        age_data = extract_numeric(df[age_col]).dropna()
        if len(age_data) > 0:
            avg_age = age_data.mean()
            analysis += f"📊 *Средний возраст:* {avg_age:.1f} лет\n\n"
    
    # Анализ качества обслуживания
    quality_cols = {
        'вежливость': 'Вежливость сотрудников',
        'компетентность': 'Компетентность',
        'понятно': 'Понятность объяснений',
        'чистота': 'Чистота помещений'
    }
    
    analysis += f"⭐ *Оценки качества:*\n"
    for key, name in quality_cols.items():
        col = COLUMN_SYNONYMS.get(key)
        if col and col in df.columns:
            freq = df[col].value_counts()
            if len(freq) > 0:
                positive_answers = freq.get('Очень вежливы', 0) + freq.get('Вежливы', 0) + freq.get('Высокая', 0) + freq.get('Очень высокая', 0) + freq.get('Очень понятно', 0) + freq.get('Понятно', 0) + freq.get('Отлично', 0) + freq.get('Хорошо', 0)
                total = freq.sum()
                positive_percent = (positive_answers / total) * 100 if total > 0 else 0
                analysis += f"• {name}: {positive_percent:.1f}% положительных оценок\n"
    
    # Анализ проблем
    problem_col = COLUMN_SYNONYMS.get('проблем')
    if problem_col and problem_col in df.columns:
        problem_freq = df[problem_col].value_counts()
        if len(problem_freq) > 0:
            no_problems = problem_freq.get('Нет, все вопросы решены', 0)
            total_problems = problem_freq.sum()
            success_rate = (no_problems / total_problems) * 100 if total_problems > 0 else 0
            analysis += f"\n✅ *Успешность решения вопросов:* {success_rate:.1f}%\n"
    
    # Рекомендации
    rec_col = COLUMN_SYNONYMS.get('рекомендация')
    if rec_col and rec_col in df.columns:
        rec_freq = df[rec_col].value_counts()
        if len(rec_freq) > 0:
            positive_rec = rec_freq.get('Определенно да', 0) + rec_freq.get('Скорее да', 0)
            total_rec = rec_freq.sum()
            rec_percent = (positive_rec / total_rec) * 100 if total_rec > 0 else 0
            analysis += f"👍 *Готовность рекомендовать:* {rec_percent:.1f}%\n"
    
    analysis += f"\n💡 *Выводы:*\n"
    analysis += f"• Общее качество обслуживания высокое\n"
    analysis += f"• Большинство клиентов довольны\n"
    analysis += f"• Есть возможности для улучшения скорости\n"
    
    return analysis

def analyze_quality_metrics(df):
    """Анализ всех метрик качества обслуживания"""
    analysis = f"⭐ *АНАЛИЗ КАЧЕСТВА ОБСЛУЖИВАНИЯ*\n\n"
    
    quality_metrics = {
        'вежливость': 'Вежливость сотрудников',
        'компетентность': 'Компетентность сотрудников', 
        'понятно': 'Понятность объяснений',
        'чистота': 'Чистота и комфорт',
        'доступность': 'Доступность информации',
        'терминал': 'Удобство терминалов'
    }
    
    total_scores = {}
    
    for key, name in quality_metrics.items():
        col = COLUMN_SYNONYMS.get(key)
        if col and col in df.columns:
            freq = df[col].value_counts()
            if len(freq) > 0:
                total = freq.sum()
                
                # Определяем положительные ответы для каждого метрика
                positive_answers = 0
                if key == 'вежливость':
                    positive_answers = freq.get('Очень вежливы', 0) + freq.get('Вежливы', 0)
                elif key == 'компетентность':
                    positive_answers = freq.get('Высокая', 0) + freq.get('Очень высокая', 0)
                elif key == 'понятно':
                    positive_answers = freq.get('Очень понятно', 0) + freq.get('Понятно', 0)
                elif key == 'чистота':
                    positive_answers = freq.get('Отлично', 0) + freq.get('Хорошо', 0)
                elif key == 'доступность':
                    positive_answers = freq.get('Очень доступна', 0) + freq.get('Доступна', 0)
                elif key == 'терминал':
                    positive_answers = freq.get('Очень удобно', 0) + freq.get('Удобно', 0)
                
                positive_percent = (positive_answers / total) * 100 if total > 0 else 0
                total_scores[key] = positive_percent
                
                # Эмодзи для оценки
                if positive_percent >= 80:
                    emoji = "🟢"
                elif positive_percent >= 60:
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                
                analysis += f"{emoji} *{name}:* {positive_percent:.1f}% положительных оценок\n"
    
    # Общий рейтинг
    if total_scores:
        avg_score = sum(total_scores.values()) / len(total_scores)
        analysis += f"\n📊 *ОБЩИЙ РЕЙТИНГ КАЧЕСТВА:* {avg_score:.1f}%\n\n"
        
        if avg_score >= 80:
            analysis += f"🏆 *Отличное качество обслуживания!*\n"
        elif avg_score >= 60:
            analysis += f"⚠️ *Хорошее качество, есть возможности для улучшения*\n"
        else:
            analysis += f"❌ *Требуется серьезная работа над качеством*\n"
    
    return analysis

def generate_detailed_analysis(df):
    """Генерирует детальный анализ с рекомендациями"""
    analysis = f"🔍 *ДЕТАЛЬНЫЙ АНАЛИЗ ОПРОСА*\n\n"
    
    # Демографический анализ
    analysis += f"👥 *ДЕМОГРАФИЯ:*\n"
    
    gender_col = COLUMN_SYNONYMS.get('пол')
    if gender_col and gender_col in df.columns:
        gender_freq = df[gender_col].value_counts()
        if len(gender_freq) > 0:
            male = gender_freq.get('Мужской', 0)
            female = gender_freq.get('Женский', 0)
            total = gender_freq.sum()
            analysis += f"• Мужчин: {male} ({male/total*100:.1f}%)\n"
            analysis += f"• Женщин: {female} ({female/total*100:.1f}%)\n\n"
    
    # Анализ банков
    analysis += f"🏦 *АНАЛИЗ БАНКОВ:*\n"
    bank_col = COLUMN_SYNONYMS.get('банк')
    if bank_col and bank_col in df.columns:
        bank_freq = df[bank_col].value_counts()
        if len(bank_freq) > 0:
            for i, (bank, count) in enumerate(bank_freq.head(3).items(), 1):
                percent = (count / bank_freq.sum()) * 100
                analysis += f"{i}. {bank}: {count} клиентов ({percent:.1f}%)\n"
            analysis += "\n"
    
    # Анализ проблем
    analysis += f"⚠️ *АНАЛИЗ ПРОБЛЕМ:*\n"
    problem_col = COLUMN_SYNONYMS.get('проблем')
    if problem_col and problem_col in df.columns:
        problem_freq = df[problem_col].value_counts()
        if len(problem_freq) > 0:
            for problem, count in problem_freq.items():
                percent = (count / problem_freq.sum()) * 100
                analysis += f"• {problem}: {count} ({percent:.1f}%)\n"
            analysis += "\n"
    
    # Рекомендации
    analysis += f"💡 *РЕКОМЕНДАЦИИ:*\n"
    analysis += f"1. Улучшить скорость обслуживания\n"
    analysis += f"2. Увеличить количество сотрудников в пиковые часы\n"
    analysis += f"3. Улучшить информационное обеспечение\n"
    analysis += f"4. Провести обучение персонала\n"
    analysis += f"5. Модернизировать терминалы\n\n"
    
    analysis += f"📈 *ПОТЕНЦИАЛ РОСТА:*\n"
    analysis += f"• Повышение удовлетворенности клиентов\n"
    analysis += f"• Увеличение лояльности\n"
    analysis += f"• Рост рекомендаций\n"
    analysis += f"• Снижение жалоб\n"
    
    return analysis

def generate_questions_list(df):
    """Генерирует список всех вопросов с кратким описанием"""
    questions = f"📋 *СПИСОК ВСЕХ ВОПРОСОВ ОПРОСА*\n\n"
    
    # Пропускаем отметку времени
    relevant_columns = [col for col in df.columns if 'отметка времени' not in col.lower() and 'timestamp' not in col.lower()]
    
    for i, col in enumerate(relevant_columns, 1):
        val = df[col].value_counts()
        total = val.sum() if len(val) > 0 else 0
        
        questions += f"{i}. *{col}*\n"
        questions += f"   📊 Ответов: {total}\n"
        
        if len(val) > 0:
            top_answer = val.idxmax()
            questions += f"   🏆 Топ ответ: {top_answer}\n"
        
        questions += "\n"
    
    questions += f"💡 *Как использовать:*\n"
    questions += f"• Напишите название вопроса для получения статистики\n"
    questions += f"• Добавьте 'график' для визуализации\n"
    questions += f"• Добавьте 'анализ' для глубокого изучения\n"
    
    return questions

def generate_comparison_analysis(df, column):
    """Генерирует анализ сравнений для колонки"""
    comparison = f"📊 *СРАВНИТЕЛЬНЫЙ АНАЛИЗ: {column}*\n\n"
    
    freq = df[column].value_counts()
    if len(freq) < 2:
        return f"❌ Недостаточно данных для сравнения в колонке '{column}'"
    
    total = freq.sum()
    
    # Сравниваем топ-2 ответа
    top_answers = freq.head(2)
    first_answer, first_count = top_answers.iloc[0], top_answers.index[0]
    second_answer, second_count = top_answers.iloc[1], top_answers.index[1]
    
    first_percent = (first_count / total) * 100
    second_percent = (second_count / total) * 100
    difference = first_count - second_count
    difference_percent = first_percent - second_percent
    
    comparison += f"🏆 *Топ-2 ответа:*\n"
    comparison += f"1. {first_answer}: {first_count} ({first_percent:.1f}%)\n"
    comparison += f"2. {second_answer}: {second_count} ({second_percent:.1f}%)\n\n"
    
    comparison += f"📈 *Разница:*\n"
    comparison += f"• Количественная: {difference} ответов\n"
    comparison += f"• Процентная: {difference_percent:.1f}%\n\n"
    
    if difference_percent > 20:
        comparison += f"💡 *Вывод:* {first_answer} значительно преобладает\n"
    elif difference_percent > 10:
        comparison += f"💡 *Вывод:* {first_answer} умеренно лидирует\n"
    else:
        comparison += f"💡 *Вывод:* Ответы распределены равномерно\n"
    
    # Если есть больше ответов, показываем полное распределение
    if len(freq) > 2:
        comparison += f"\n📋 *Полное распределение:*\n"
        for i, (answer, count) in enumerate(freq.items(), 1):
            percent = (count / total) * 100
            comparison += f"{i}. {answer}: {count} ({percent:.1f}%)\n"
    
    return comparison

def generate_recommendations(df, column):
    """Генерирует рекомендации на основе данных колонки"""
    recommendations = f"💡 *РЕКОМЕНДАЦИИ ПО: {column}*\n\n"
    
    freq = df[column].value_counts()
    if len(freq) == 0:
        return f"❌ Нет данных для анализа в колонке '{column}'"
    
    total = freq.sum()
    top_answer = freq.idxmax()
    top_count = freq.max()
    top_percent = (top_count / total) * 100
    
    # Определяем тип колонки и даем соответствующие рекомендации
    column_lower = column.lower()
    
    if 'банк' in column_lower:
        recommendations += f"🏦 *Анализ банков:*\n"
        recommendations += f"• Лидер: {top_answer} ({top_percent:.1f}% клиентов)\n\n"
        recommendations += f"📋 *Рекомендации:*\n"
        recommendations += f"1. Изучить опыт лидирующего банка\n"
        recommendations += f"2. Провести конкурентный анализ\n"
        recommendations += f"3. Улучшить сервисы в отстающих банках\n"
        recommendations += f"4. Разработать уникальные предложения\n"
        
    elif 'возраст' in column_lower:
        recommendations += f"📊 *Анализ возраста:*\n"
        recommendations += f"• Самый частый возраст: {top_answer} лет\n\n"
        recommendations += f"📋 *Рекомендации:*\n"
        recommendations += f"1. Адаптировать услуги под целевую аудиторию\n"
        recommendations += f"2. Разработать программы для разных возрастных групп\n"
        recommendations += f"3. Улучшить цифровые каналы для молодежи\n"
        recommendations += f"4. Создать специальные предложения для старшего возраста\n"
        
    elif 'пол' in column_lower:
        recommendations += f"👥 *Анализ гендера:*\n"
        recommendations += f"• Преобладает: {top_answer}\n\n"
        recommendations += f"📋 *Рекомендации:*\n"
        recommendations += f"1. Разработать гендерно-ориентированные продукты\n"
        recommendations += f"2. Адаптировать маркетинговые кампании\n"
        recommendations += f"3. Улучшить обслуживание для меньшинства\n"
        recommendations += f"4. Провести исследования потребностей\n"
        
    elif 'качество' in column_lower or 'оценка' in column_lower or 'удовлетворенность' in column_lower:
        recommendations += f"⭐ *Анализ качества:*\n"
        recommendations += f"• Основная оценка: {top_answer}\n\n"
        recommendations += f"📋 *Рекомендации:*\n"
        recommendations += f"1. Поддерживать высокие стандарты\n"
        recommendations += f"2. Улучшить проблемные области\n"
        recommendations += f"3. Провести обучение персонала\n"
        recommendations += f"4. Внедрить систему обратной связи\n"
        
    elif 'проблем' in column_lower or 'жалоб' in column_lower:
        recommendations += f"⚠️ *Анализ проблем:*\n"
        recommendations += f"• Основная проблема: {top_answer}\n\n"
        recommendations += f"📋 *Рекомендации:*\n"
        recommendations += f"1. Приоритизировать решение основных проблем\n"
        recommendations += f"2. Улучшить процессы обслуживания\n"
        recommendations += f"3. Увеличить количество персонала\n"
        recommendations += f"4. Внедрить автоматизацию\n"
        
    else:
        recommendations += f"📊 *Общий анализ:*\n"
        recommendations += f"• Топ ответ: {top_answer} ({top_percent:.1f}%)\n\n"
        recommendations += f"📋 *Общие рекомендации:*\n"
        recommendations += f"1. Изучить причины популярности топ-ответа\n"
        recommendations += f"2. Улучшить менее популярные варианты\n"
        recommendations += f"3. Провести дополнительное исследование\n"
        recommendations += f"4. Разработать стратегию развития\n"
    
    recommendations += f"\n🎯 *Следующие шаги:*\n"
    recommendations += f"• Провести детальный анализ\n"
    recommendations += f"• Разработать план действий\n"
    recommendations += f"• Измерить результаты изменений\n"
    
    return recommendations

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.run_polling()

if __name__ == '__main__':
    main()