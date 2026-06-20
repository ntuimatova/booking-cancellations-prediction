# Booking Cancellations Prediction

Бинарная классификация отмены бронирования отелей с помощью полносвязной нейронной сети.

Задача: по признакам бронирования предсказать, будет ли оно отменено (`is_canceled`: 0 - нет, 1 - да).

### Данные

- **Источник:** [Hotel booking demand (Kaggle)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Файл:** `hotel_bookings.csv`
- **Размер:** 119 390 бронирований
- **Признаки:** тип отеля, канал бронирования, lead time, тип депозита, количество гостей, цена, дата заезда и др.
- **Целевая переменная:** `is_canceled`

### Структура проекта

```
booking-cancellations-prediction/
├── hotel_booking_cancellations_prediction.ipynb
├── hotel_bookings.csv
└── README.md
```

### Запуск
```bash
cd booking-cancellations-prediction
python -m venv venv
venv\Scripts\activate          # Windows
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow jupyter
jupyter notebook hotel_booking_cancellations_prediction.ipynb
```

### Подход
1) EDA: описательная статистика, анализ факторов отмены
2) Подготовка данных: кодирование категориальных признаков, масштабирование
3) Модель: полносвязная нейронная сеть (256 → 128 → 64 → 1), ReLU, sigmoid на выходе, BatchNormalization и Dropout
4) Оценка: accuracy на тестовой выборке

### Результаты
- Test set: Accuracy - **83%**
- EDA показал, что на отмену сильнее всего влияют lead_time (время от бронирования до заезда) и deposit_type (тип депозита). Дальнейший рост точности ограничен случайным характером отмен - человеческим фактором и внешними обстоятельствами, не отражёнными в данных.
- Модель может использоваться для оценки риска отмены при планировании загрузки отелей.

### Стек
Python, pandas, scikit-learn, TensorFlow/Keras, matplotlib, seaborn, Jupyter

### Автор
**Туйматова Надежда**

- GitHub: https://github.com/ntuimatova
- Email: n.tuimatova@yandex.ru
