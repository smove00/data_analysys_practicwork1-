# practical_work_1_bank_churn.py
"""
Практическое задание №1: Прогнозирование оттока клиентов банка
Полный код для анализа оттока клиентов банка
"""

# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Настройка визуализации
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

print("=== ПРАКТИЧЕСКОЕ ЗАДАНИЕ №1: АНАЛИЗ ОТТОКА КЛИЕНТОВ БАНКА ===\n")

# =============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =============================================================================

print("1. ЗАГРУЗКА ДАННЫХ")
print("-" * 50)

# Загрузка данных
df = pd.read_csv('Churn_Modelling.csv')

print(f"Размер датасета: {df.shape}")
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}\n")

# Первичный осмотр данных
print("Первые 5 строк данных:")
print(df.head())

print("\nИнформация о данных:")
print(df.info())

print("\nСтатистическое описание числовых признаков:")
print(df.describe())

# =============================================================================
# 2. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)
# =============================================================================

print("\n2. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)")
print("-" * 50)

# Анализ целевой переменной
print("Распределение целевой переменной (Exited):")
print(df['Exited'].value_counts())
print(f"Доля ушедших клиентов: {df['Exited'].mean():.2%}")

# Визуализация распределения целевой переменной
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Exited')
plt.title('Распределение клиентов по факту оттока')
plt.xlabel('Отток (0 - остался, 1 - ушел)')
plt.ylabel('Количество клиентов')

plt.subplot(1, 2, 2)
df['Exited'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
plt.title('Доля оттока клиентов')

plt.tight_layout()
plt.show()

# Анализ категориальных признаков
print("\nАнализ категориальных признаков:")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# География
sns.countplot(data=df, x='Geography', hue='Exited', ax=axes[0, 0])
axes[0, 0].set_title('Отток по странам')
axes[0, 0].tick_params(axis='x', rotation=45)

# Пол
sns.countplot(data=df, x='Gender', hue='Exited', ax=axes[0, 1])
axes[0, 1].set_title('Отток по полу')

# Количество продуктов
sns.countplot(data=df, x='NumOfProducts', hue='Exited', ax=axes[1, 0])
axes[1, 0].set_title('Отток по количеству продуктов')

# Наличие кредитной карты
sns.countplot(data=df, x='HasCrCard', hue='Exited', ax=axes[1, 1])
axes[1, 1].set_title('Отток по наличию кредитной карты')

plt.tight_layout()
plt.show()

# Анализ числовых признаков
print("\nАнализ числовых признаков:")

numerical_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, feature in enumerate(numerical_features):
    row, col = i // 2, i % 2
    sns.boxplot(data=df, x='Exited', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'Распределение {feature}')

plt.tight_layout()
plt.show()

# Распределение возраста с разбивкой по оттоку
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(data=df, x='Age', hue='Exited', kde=True, bins=30)
plt.title('Распределение возраста')

plt.subplot(1, 2, 2)
sns.histplot(data=df, x='Balance', hue='Exited', kde=True, bins=30)
plt.title('Распределение баланса')

plt.tight_layout()
plt.show()

# Анализ срока сотрудничества
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Tenure', hue='Exited')
plt.title('Отток по сроку сотрудничества (Tenure)')
plt.xlabel('Срок сотрудничества (лет)')
plt.ylabel('Количество клиентов')
plt.show()

# =============================================================================
# 3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# =============================================================================

print("\n3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("-" * 50)

# Выделяем числовые признаки для корреляционного анализа
numeric_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                  'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']

correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', linewidths=0.5)
plt.title('Матрица корреляций числовых признаков', fontsize=16)
plt.tight_layout()
plt.show()

# Анализ корреляций с целевой переменной
print("Корреляция признаков с целевой переменной (Exited):")
correlation_with_target = correlation_matrix['Exited'].sort_values(ascending=False)
print(correlation_with_target)

# Визуализация топ корреляций
plt.figure(figsize=(10, 6))
top_correlations = correlation_with_target.drop('Exited').head(10)
sns.barplot(x=top_correlations.values, y=top_correlations.index)
plt.title('Топ-10 признаков по корреляции с оттоком')
plt.xlabel('Коэффициент корреляции')
plt.tight_layout()
plt.show()

# =============================================================================
# 4. ПРЕДОБРАБОТКА ДАННЫХ
# =============================================================================

print("\n4. ПРЕДОБРАБОТКА ДАННЫХ")
print("-" * 50)

# Создаем копию данных для предобработки
df_processed = df.copy()

print("Размер данных до предобработки:", df_processed.shape)

# Удаление неинформативных столбцов
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
df_processed = df_processed.drop(columns_to_drop, axis=1)

print("Размер данных после удаления столбцов:", df_processed.shape)

# Проверка на пропущенные значения
print("\nПропущенные значения:")
print(df_processed.isnull().sum())

# Кодирование категориальных переменных
print("\nКодирование категориальных переменных...")

# One-Hot Encoding для Geography
df_encoded = pd.get_dummies(df_processed, columns=['Geography'], drop_first=True)

# Label Encoding для Gender
le = LabelEncoder()
df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'])

print("Категориальные переменные закодированы:")
print(f"Gender mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# =============================================================================
# 5. КОНСТРУИРОВАНИЕ ПРИЗНАКОВ
# =============================================================================

print("\n5. КОНСТРУИРОВАНИЕ ПРИЗНАКОВ")
print("-" * 50)

# Создание новых признаков
print("Создание новых признаков...")

# Возрастные группы
df_encoded['AgeGroup'] = pd.cut(df_encoded['Age'], 
                               bins=[0, 30, 45, 60, 100],
                               labels=['Young', 'Middle', 'Senior', 'Elderly'])

# Бинарные признаки
df_encoded['HighBalance'] = (df_encoded['Balance'] > df_encoded['Balance'].median()).astype(int)
df_encoded['LoyalCustomer'] = (df_encoded['Tenure'] > 5).astype(int)
df_encoded['HighCreditScore'] = (df_encoded['CreditScore'] > 700).astype(int)

# Взаимодействие признаков
df_encoded['BalancePerProduct'] = df_encoded['Balance'] / (df_encoded['NumOfProducts'] + 1)
df_encoded['ActiveWithProducts'] = (df_encoded['IsActiveMember'] == 1) & (df_encoded['NumOfProducts'] > 1)

# One-Hot Encoding для возрастных групп
df_final = pd.get_dummies(df_encoded, columns=['AgeGroup'], drop_first=True)

print(f"Размер данных после создания признаков: {df_final.shape}")
print(f"Количество признаков: {df_final.shape[1]}")

# =============================================================================
# 6. ОТБОР ПРИЗНАКОВ
# =============================================================================

print("\n6. ОТБОР ПРИЗНАКОВ")
print("-" * 50)

# Анализ важности признаков
final_correlation = df_final.corr()['Exited'].sort_values(ascending=False)

print("Топ-15 наиболее значимых признаков:")
print(final_correlation.head(15))

# Визуализация важности признаков
plt.figure(figsize=(12, 8))
top_features = final_correlation.drop('Exited').head(15)
sns.barplot(x=top_features.values, y=top_features.index)
plt.title('Топ-15 признаков по корреляции с оттоком', fontsize=14)
plt.xlabel('Коэффициент корреляции')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Выбор наиболее значимых признаков для моделирования
selected_features = [
    'Age', 'IsActiveMember', 'Balance', 'Geography_Germany', 
    'NumOfProducts', 'Gender', 'HighBalance', 'LoyalCustomer',
    'AgeGroup_Middle', 'AgeGroup_Senior', 'AgeGroup_Elderly',
    'Exited'
]

df_selected = df_final[selected_features]

print(f"Финальный размер датасета: {df_selected.shape}")
print(f"Отобранные признаки: {len(selected_features) - 1}")  # исключаем целевую переменную

# =============================================================================
# 7. АНАЛИЗ И ВЫВОДЫ
# =============================================================================

print("\n7. АНАЛИЗ И ВЫВОДЫ")
print("-" * 50)

# Ключевые метрики
total_customers = len(df)
churned_customers = df['Exited'].sum()
churn_rate = churned_customers / total_customers

print(f"Общее количество клиентов: {total_customers}")
print(f"Количество ушедших клиентов: {churned_customers}")
print(f"Уровень оттока: {churn_rate:.2%}")

# Анализ по странам
country_analysis = df.groupby('Geography')['Exited'].agg(['count', 'sum', 'mean'])
country_analysis['churn_rate'] = country_analysis['mean']
country_analysis = country_analysis.sort_values('churn_rate', ascending=False)

print("\nАнализ оттока по странам:")
print(country_analysis[['count', 'sum', 'churn_rate']])

# Анализ по возрасту
age_analysis = df.groupby(pd.cut(df['Age'], bins=[0, 30, 45, 60, 100]))['Exited'].mean()
print("\nУровень оттока по возрастным группам:")
print(age_analysis)

# Анализ по активности
activity_analysis = df.groupby('IsActiveMember')['Exited'].mean()
print("\nУровень оттока по активности:")
print(activity_analysis)

# =============================================================================
# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

print("\n8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("-" * 50)

# Сохранение обработанных данных
df_selected.to_csv('bank_churn_processed.csv', index=False)
df_final.to_csv('bank_churn_full_processed.csv', index=False)

print("Обработанные данные сохранены в файлы:")
print("- bank_churn_processed.csv (отобранные признаки)")
print("- bank_churn_full_processed.csv (все признаки)")

# Сохранение ключевых статистик
summary_stats = {
    'total_customers': total_customers,
    'churned_customers': churned_customers,
    'churn_rate': churn_rate,
    'final_features_count': len(selected_features) - 1,
    'top_features': list(top_features.index[:5])
}

print("\nКлючевые статистики анализа:")
for key, value in summary_stats.items():
    print(f"{key}: {value}")

print("\n=== АНАЛИЗ ЗАВЕРШЕН ===")
print("Данные готовы для построения моделей машинного обучения!")
