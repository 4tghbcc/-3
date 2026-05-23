import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class UniversityBot:
    """
    Простой чат-бот для Университета Синергия на основе классификации текста (TF-IDF + ML).
    """
    def __init__(self):
        # Определяем интенции (намерения) и ответы
        self.intents = {
            "greeting": {
                "examples": ["привет", "здравствуйте", "добрый день", "hi", "hello", "бот привет"],
                "response": "Привет! Я виртуальный помощник Университета 'Синергия'. Чем могу помочь?",
            },
            "bye": {
                "examples": ["пока", "до свидания", "всего хорошего", "exit", "quit", "конец"],
                "response": "До свидания! Если появятся вопросы, обращайтесь. Хорошего дня!",
            },
            "price": {
                "examples": ["сколько стоит обучение", "цена", "стоимость", "платно", "оплата", "прайс"],
                "response": "Стоимость обучения зависит от факультета и формы обучения. Рекомендую посмотреть актуальные цены на официальном сайте: synergy.ru/price или обратиться в Приемную комиссию.",
            },
            "admission": {
                "examples": ["как поступить", "прием", "поступление", "экзамены", "егэ", "вступительные", "документы", "что нужно для поступления"],
                "response": "Для поступления нужно: 1) Заполнить заявление на сайте; 2) Предоставить документ об образовании (аттестат/диплом) и паспорт; 3) Сдать вступительные испытания (ЕГЭ или внутренние экзамены вуза).",
            },
            "faculties": {
                "examples": ["факультеты", "кафедры", "направления", "специальности", "кем я могу стать", "чему учат"],
                "response": "Мы предлагаем более 1000 программ: IT, Медицина, Экономика, Юриспруденция, Туризм, Дизайн и др. Полный список факультетов: synergy.ru/faculties.",
            },
            "address": {
                "examples": ["где находитесь", "адрес", "как проехать", "метро", "расположение", "главный офис", "филиал"],
                "response": "Головной офис: г. Москва, ул. Мещанская, д. 9/14 стр. 1. Также у нас есть филиалы по всей России и зарубежом (Дубай, Малайзия).",
            },
            "it_courses": {
                "examples": ["айти", "программирование", "it курсы", "разработчик", "1с", "python"],
                "response": "На факультете IT мы сотрудничаем с экосистемой '1С'. Изучаем Python, Java, 1С:Предприятие, базы данных. Есть программы бакалавриата и курсы повышения квалификации.",
            },
            "fallback": {
                "examples": [],  # Случай по умолчанию
                "response": "Извините, я пока не обучен отвечать на этот вопрос. Переформулируйте или обратитесь к администратору. Вы можете написать 'Помощь'.",
            }
        }

        # Подготовка данных для обучения модели
        self._train_model()

    def _preprocess_text(self, text):
        """Очистка текста (нижний регистр, удаление знаков препинания)"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Удаляем пунктуацию
        return text

    def _train_model(self):
        """Обучаем классификатор на примерах интенций"""
        X_train = []
        y_train = []
        
        for intent, data in self.intents.items():
            if intent == "fallback":
                continue
            for example in data["examples"]:
                X_train.append(self._preprocess_text(example))
                y_train.append(intent)
        
        # Если данных нет (или мало), векторизуем и обучаем модель
        if X_train:
            # TF-IDF векторизация (превращает текст в числа)
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2)) # Учитываем отдельные слова и пары
            X_train_vec = self.vectorizer.fit_transform(X_train)
            
            # Логистическая регрессия (простой, но эффективный классификатор)
            self.model = LogisticRegression(max_iter=200, random_state=42)
            self.model.fit(X_train_vec, y_train)
            self.is_trained = True
        else:
            self.is_trained = False
            print("Warning: No training data found.")

    def predict_intent(self, user_input):
        """Определяет намерение пользователя"""
        if not self.is_trained:
            return "fallback"
            
        processed_input = self._preprocess_text(user_input)
        input_vec = self.vectorizer.transform([processed_input])
        
        # Предсказываем интенцию
        intent = self.model.predict(input_vec)[0]
        return intent

    def get_response(self, user_input):
        """Получает ответ на основе намерения"""
        intent = self.predict_intent(user_input)
        # Если уверенность низкая (можно добавить порог вероятности), но в базовой версии используем fallback по-умному
        # Если интенция не найдена или это фолбэк
        if intent not in self.intents:
            intent = "fallback"
        
        return self.intents[intent]["response"]

# Точка входа в программу
def main():
    print("="*50)
    print("🤖 Бот-помощник Университета 'Синергия' (версия для практики)")
    print("Напишите 'пока' или 'выход' для завершения работы.")
    print("="*50)
    
    bot = UniversityBot()
    
    while True:
        user_input = input("\n👉 Вы: ")
        
        if user_input.lower() in ['выход', 'exit', 'quit', 'закрыть']:
            print("👋 Бот: До свидания! Рады были помочь.")
            break
            
        if not user_input.strip():
            continue
            
        response = bot.get_response(user_input)
        print(f"🤖 Бот: {response}")

if __name__ == "__main__":
    main()
