import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# Helpers
def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{name}")
    print(f"R²   : {r2:.5f}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

# 1) Load + quick checks
 
df = pd.read_csv('IZStepProjectPython/ikea_clean.csv')

cols = ['price', 'depth', 'width', 'height', 'category', 'other_colors', 'designer_norm']
df = df[cols].copy()

print("Shape:", df.shape)
print(df.head(3))
print("\nMissing values:\n", df.isna().sum())

print("\nPrice sanity:")
print("  price < 0 :", (df['price'] < 0).sum())
print("  price ==0 :", (df['price'] == 0).sum())
print("  price NaN :", df['price'].isna().sum())

# 2) Minimal cleaning price must be known and positive
df = df[df['price'].notna() & (df['price'] > 0)].reset_index(drop=True)

# "stub" sizes: <= 2 -> NaN (will be imputed in pipeline)
dims = ['depth', 'width', 'height']
for c in dims:
    df.loc[df[c].notna() & (df[c] <= 2), c] = np.nan

print("\nAfter cleaning shape:", df.shape)
print("Missing values after cleaning:\n", df.isna().sum())

# 3) Split

numeric_features = ['depth', 'width', 'height']
categorical_features = ['category', 'designer_norm', 'other_colors']

X = df[numeric_features + categorical_features].copy()
y = df['price'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain:", X_train.shape, "Test:", X_test.shape)

# 4) Preprocessing, - Ridge: needs scaling, - RF: scaling not needed
 
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer_ridge = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

numeric_transformer_rf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

preprocess_ridge = ColumnTransformer(transformers=[
    ('num', numeric_transformer_ridge, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

preprocess_rf = ColumnTransformer(transformers=[
    ('num', numeric_transformer_rf, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

# 5) Baseline: Ridge (log target)
 
ridge_pipe = Pipeline(steps=[
    ('preprocess', preprocess_ridge),
    ('model', Ridge())
])

ridge_model = TransformedTargetRegressor(
    regressor=ridge_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

ridge_model.fit(X_train, y_train)
pred_ridge = ridge_model.predict(X_test)
evaluate("Baseline: Ridge (log target)", y_test, pred_ridge)


# 6) RandomForest baseline (log target)
 
rf_pipe = Pipeline(steps=[
    ('preprocess', preprocess_rf),
    ('model', RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model = TransformedTargetRegressor(
    regressor=rf_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)
evaluate("RF baseline (log target)", y_test, pred_rf)

# 7) Cross-validation for RF baseline

cv_scores = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print("\nCV R² per fold:", np.round(cv_scores, 5))
print("Mean CV R²:", cv_scores.mean().round(5))
print("Std  CV R²:", cv_scores.std().round(5))


# 8) GridSearchCV for RF

param_grid = {
    'regressor__model__n_estimators': [300, 600],
    'regressor__model__max_depth': [None, 15, 30],
    'regressor__model__min_samples_leaf': [1, 2, 4],
}

grid = GridSearchCV(
    rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest params:", grid.best_params_)
print("Best CV R²:", round(grid.best_score_, 5))

best_rf = grid.best_estimator_
pred_best = best_rf.predict(X_test)
evaluate("Best RF from GridSearch (log target)", y_test, pred_best)

# 9) Permutation importance (on best_rf)
 
result = permutation_importance(
    best_rf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring='r2'
)

imp_best = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("\nPermutation importance for BEST_RF (drop in R²):")
print(imp_best.round(5))

# 10) Add engineered feature: volume (FINAL model)

X_train2 = X_train.copy()
X_test2  = X_test.copy()

dims = ['depth', 'width', 'height']

med = X_train2[dims].median()

X_train2['volume'] = X_train2[dims].fillna(med).prod(axis=1)
X_test2['volume']  = X_test2[dims].fillna(med).prod(axis=1)

numeric_features2 = ['depth', 'width', 'height', 'volume']

preprocess_rf2 = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features2),
    ('cat', categorical_transformer, categorical_features),
])

rf_pipe2 = Pipeline(steps=[
    ('preprocess', preprocess_rf2),
    ('model', RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1
    ))
])

final_model = TransformedTargetRegressor(
    regressor=rf_pipe2,
    func=np.log1p,
    inverse_func=np.expm1
)

final_model.fit(X_train2, y_train)
final_pred = final_model.predict(X_test2)
evaluate("FINAL: RF + volume (log target)", y_test, final_pred)


result_final = permutation_importance(
    final_model,
    X_test2,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring='r2'
)

imp_final = pd.Series(result_final.importances_mean, index=X_test2.columns).sort_values(ascending=False)
print("\nPermutation importance for FINAL_MODEL (drop in R²):")
print(imp_final.round(5))

# 11) Diagnostics for FINAL model
 
plt.figure(figsize=(6, 6))
plt.scatter(y_test, final_pred, alpha=0.4)
mn = min(y_test.min(), final_pred.min())
mx = max(y_test.max(), final_pred.max())
plt.plot([mn, mx], [mn, mx], linestyle='--')
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual vs Predicted (FINAL RF + volume)")
plt.tight_layout()
plt.show()

residuals = y_test - final_pred
plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=30)
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Count")
plt.title("Residuals histogram (FINAL RF + volume)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
imp_final.sort_values().plot(kind='barh')
plt.title("Permutation importance (FINAL: drop in R²)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df['price'], bins=40)
plt.title("Price distribution (raw)")
plt.xlabel("price")
plt.ylabel("count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.hist(np.log1p(df['price']), bins=40)
plt.title("Price distribution (log1p)")
plt.xlabel("log1p(price)")
plt.ylabel("count")
plt.tight_layout()
plt.show()


'''
Висновки щодо моделювання ціни IKEA
Було побудовано модель прогнозування ціни меблів на основі ознак: розміри (depth/width/height), категорія (category), дизайнер (designer_norm) та наявність інших кольорів (other_colors). Перед навчанням виконано мінімальну очистку даних: залишено лише додатні ціни, а підозріло малі розміри (≤2) замінено на NaN з подальшим заповненням у пайплайні медіаною. Для коректного навчання й оцінювання застосовано pipeline з імпутацією та one-hot кодуванням категоріальних ознак, а таргет перетворено через log1p(price), щоб зменшити вплив “довгого хвоста” дорогих товарів.
Як базову модель використано Ridge Regression, яка показала низьку якість: R² = 0.069, MAE ≈ 572, RMSE ≈ 1373. Це свідчить, що залежність ціни від ознак є нелінійною, і лінійна модель не здатна її адекватно відтворити.

Далі застосовано RandomForestRegressor, який суттєво покращив результати: R² ≈ 0.808, MAE ≈ 349, RMSE ≈ 623. Перевірка стабільності через 5-fold cross-validation дала Mean CV R² ≈ 0.776 зі стандартним відхиленням ≈ 0.031, що підтверджує стабільність моделі та відсутність сильного перенавчання. Налаштування гіперпараметрів через GridSearchCV дало майже такі самі результати, отже базові параметри Random Forest вже були близькі до оптимальних.

Як невелике покращення було додано інженерну ознаку volume = depth × width × height, що підняло якість до фінального рівня: R² = 0.810, MAE ≈ 346.5, RMSE ≈ 619.9. Пермутаційна важливість ознак показала, що найбільший внесок у прогноз має ширина (width), далі — категорія товару, дизайнер, а також глибина та висота. Ознака other_colors має найменший вплив, тобто сама по собі наявність кольорових варіантів слабко впливає на точність прогнозу у порівнянні з розмірами та категоріями.

Підсумок: найкращою моделлю в рамках цієї роботи є Random Forest з логарифмуванням таргета та додатковою ознакою volume. Модель демонструє хорошу якість прогнозу (R² ≈ 0.81) і стабільність за CV, а також має інтерпретовані ключові фактори (насамперед width, category та designer_norm), що узгоджується зі здоровим глуздом для ціноутворення меблів.
'''
