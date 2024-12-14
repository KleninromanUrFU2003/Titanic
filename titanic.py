# Импорт библиотек для работы с данными, построения графиков и модели
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Загрузка данных Titanic
titanic = pd.read_csv('/kaggle/input/titanic/Titanic.csv')  # Загрузка данных

# Проверка на пропуски
print("Количество пропусков в данных:")
print(titanic.isnull().sum())  # Выводим количество пропусков в каждом столбце

# Заполнение пропусков
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)  # Возраст заполняем медианой
titanic['Fare'].fillna(titanic['Fare'].median(), inplace=True)  # Тариф тоже медианой
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)  # Порт посадки модой

# Анализ количества выживших и невыживших
print("Распределение выживших и погибших:")
print(titanic['Survived'].value_counts())  # Считаем выживших и погибших

# График выживших и погибших
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=titanic, palette='Set1')
plt.title('Количество выживших и погибших')
plt.xlabel('Выживание (0 - не выжил, 1 - выжил)')
plt.ylabel('Количество')
plt.show()

# График возраста для выживших и погибших
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='Age', hue='Survived', multiple='stack', palette='Set2', kde=False)
plt.title('Распределение возраста среди выживших и погибших')
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.show()

# График выживания по полу
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', hue='Survived', data=titanic, palette='Set3')
plt.title('Распределение по полу среди выживших и погибших')
plt.xlabel('Пол')
plt.ylabel('Количество')
plt.legend(title='Выживание', labels=['Не выжил', 'Выжил'])
plt.show()

# График выживания по классам
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=titanic, palette='Set1')
plt.title('Распределение по классам среди выживших и погибших')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.legend(title='Выживание', labels=['Не выжил', 'Выжил'])
plt.show()

# Создание новых признаков
# Семья (SibSp - число братьев, супругов и Parch - дети и родители)
titanic['Family_Size'] = titanic['SibSp'] + titanic['Parch'] + 1  # Размер семьи

# Признак "Один или нет"
titanic['Is_Alone'] = (titanic['Family_Size'] == 1).astype(int)  # Признак "был ли пассажир один"

# Преобразование категорий
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Family_Size', 'Is_Alone']  # Выбор признаков
X = pd.get_dummies(titanic[features], drop_first=True)  # Преобразуем категориальные данные
y = titanic['Survived']  # Целевая переменная

# Проверка на NaN
X.fillna(0, inplace=True)  # Убираем пропуски, если остались

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Разделение данных

# Масштабирование
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Масштабируем тренировочные данные
X_test = scaler.transform(X_test)  # Масштабируем тестовые данные

# Создание модели Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Создаем модель случайного леса
model.fit(X_train, y_train)  # Обучаем модель

# Предсказания
y_pred = model.predict(X_test)  # Делаем предсказания

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)  # Точность модели
conf_matrix = confusion_matrix(y_test, y_pred)  # Матрица ошибок
class_report = classification_report(y_test, y_pred)  # Отчет по классификации

print(f"Точность модели: {accuracy}")  # Вывод точности
print("Матрица ошибок:")
print(conf_matrix)  # Печать матрицы ошибок
print("Отчет по классификации:")
print(class_report)  # Печать отчета по классификации

# Важность признаков
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
print("Важность признаков:")
print(feature_importances)  # Печать важности каждого признака
