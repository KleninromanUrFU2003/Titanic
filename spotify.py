import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif

# 1. Загрузка данных
df = pd.read_csv('/kaggle/input/spotify/dataset (2).csv')

# Предварительная обработка данных (замена NaN, проверка типов)
print("Первые 5 строк датасета:")
print(df.head())
print("\nОписание данных:")
print(df.info())

# Разделение признаков и целевой переменной
target_column = 'popularity'
y = df[target_column]
X = df.drop(columns=[target_column])

# Обработка пропусков
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')
X.fillna(X.median(numeric_only=True), inplace=True)

# 2. EDA (исследовательский анализ)
sns.set(style="whitegrid")

# 2.1 Распределение целевой переменной
plt.figure(figsize=(8, 4))
sns.countplot(y=y, palette='viridis')
plt.title('Распределение целевой переменной')
plt.show()

# Вывод: смотрим на дисбаланс классов
print("\nРаспределение целевой переменной:")
print(y.value_counts())

# 2.2 Корреляция числовых переменных
plt.figure(figsize=(12, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Корреляция признаков')
plt.show()

# Вывод: Выбираем признаки с высокой корреляцией (по модулю > 0.7) и решаем об их удалении

# 2.3 Анализ выбросов
for col in X.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=X, x=col, color='salmon')
    plt.title(f'Анализ выбросов для {col}')
    plt.show()

# Вывод: Проверяем необходимость обработки выбросов (например, логарифмирование или Winsorization)

# 3. Feature Engineering
# Пример: создание нового признака, произведение двух числовых колонок
if len(X.select_dtypes(include=[np.number]).columns) >= 2:
    X['new_feature'] = X.iloc[:, 0] * X.iloc[:, 1]
    print("Создан новый признак: произведение двух первых числовых колонок")

# 3.1 Корреляция новых признаков с таргетом
new_corr = pd.Series(mutual_info_classif(X, y, discrete_features=False), index=X.columns)
print("Корреляция признаков с таргетом:")
print(new_corr.sort_values(ascending=False))

# 4. Feature Importances (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 5))
plt.title('Важность признаков (Random Forest)')
plt.show()

# 5. Эксперименты с моделями машинного обучения
# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5.1 Линейная модель
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_logreg))

# 5.2 Модель на основе деревьев
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# 5.3 Градиентный бустинг
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Report:")
print(classification_report(y_test, y_pred_xgb))

# 5.4 Нейронная сеть
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
print("MLP Report:")
print(classification_report(y_test, y_pred_mlp))

# 6. Кросс-валидация для лучшей модели
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(xgb, X, y, cv=kf, scoring='accuracy')
print("Средняя точность модели XGBoost на кросс-валидации:", scores.mean())
