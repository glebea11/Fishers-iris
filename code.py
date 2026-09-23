import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
X = iris.data[iris.target != 2, :2] # Тут берем только первые два признака т.к. у нас бинарная классификация 
y = iris.target[iris.target != 2] # Метки: 0 — setosa, 1 — versicolor
# Разделяем данные на обучающую и тестовую выборки(тестовую выборку лучше взять 30%, т.к. в нашем датасете 150 объектов)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# Создание самой модели
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Настройка метрик
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")
print("\nМатрица ошибок:")
print(confusion_matrix(y_test, y_pred))
print("\nОтчёт по классификации:")
print(classification_report(y_test, y_pred))
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolor='k', s=100)
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
)

grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)
# Визуализация границы решения и карты вероятностей
plt.figure(figsize=(8, 6))

contourf = plt.contourf(xx, yy, probs, levels=50, cmap="coolwarm", alpha=0.6)

decision_boundary = plt.contour(xx, yy, probs, levels=[0.5], colors='k', linewidths=2)

# тут точки тестовой выборки
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolor='k', s=100, vmin=0, vmax=1)

plt.show()
