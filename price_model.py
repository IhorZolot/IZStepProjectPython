import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

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


df = df[df['price'].notna() & (df['price'] > 0)].reset_index(drop=True)

dims = ['depth', 'width', 'height']
for c in dims:
    df.loc[df[c].notna() & (df[c] <= 2), c] = np.nan

print("After cleaning shape:", df.shape)
print("Missing values after cleaning:\n", df.isna().sum())

numeric_features = ['depth', 'width', 'height']
categorical_features = ['category', 'designer_norm', 'other_colors']

X = df[numeric_features + categorical_features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape, "Test:", X_test.shape)


def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{name}")
    print(f"R²   : {r2:.5f}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

ridge_pipe = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', Ridge())
])

model = TransformedTargetRegressor(
    regressor=ridge_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
evaluate("Baseline: Ridge (log target)", y_test, pred)

from sklearn.ensemble import RandomForestRegressor

rf_pipe = Pipeline(steps=[
    ('preprocess', preprocess),
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
evaluate("Step 5: RandomForest (log target)", y_test, pred_rf)


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    rf_model,        
    X_train,
    y_train,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print("CV R² per fold:", np.round(cv_scores, 5))
print("Mean CV R²:", cv_scores.mean().round(5))
print("Std  CV R²:", cv_scores.std().round(5))


from sklearn.model_selection import GridSearchCV

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

print("Best params:", grid.best_params_)
print("Best CV R²:", round(grid.best_score_, 5))

best_rf = grid.best_estimator_
pred_best = best_rf.predict(X_test)
evaluate("Step 7: Best RF from GridSearch (log target)", y_test, pred_best)


import matplotlib.pyplot as plt

# Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, pred_rf, alpha=0.4)
mn = min(y_test.min(), pred_rf.min())
mx = max(y_test.max(), pred_rf.max())
plt.plot([mn, mx], [mn, mx], linestyle='--')
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual vs Predicted (RF)")
plt.tight_layout()
plt.show()

# Residuals
residuals = y_test - pred_rf
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=30)
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Count")
plt.title("Residuals histogram (RF)")
plt.tight_layout()
plt.show()

from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np

best_rf = best_rf  # у тебе вже є grid.best_estimator_ (TransformedTargetRegressor)

result = permutation_importance(
    best_rf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring='r2'
)

imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("\nPermutation importance (drop in R²):")
print(imp.round(5))


X_train2 = X_train.copy()
X_test2 = X_test.copy()

# volume (NaN збережеться, а потім його заповнить імпутер)
X_train2['volume'] = X_train2['depth'] * X_train2['width'] * X_train2['height']
X_test2['volume']  = X_test2['depth']  * X_test2['width']  * X_test2['height']

numeric_features2 = ['depth', 'width', 'height', 'volume']
categorical_features = ['category', 'designer_norm', 'other_colors']


numeric_transformer2 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # для RF не критично, але лишаємо як було
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess2 = ColumnTransformer(transformers=[
    ('num', numeric_transformer2, numeric_features2),
    ('cat', categorical_transformer, categorical_features)
])

rf_pipe2 = Pipeline(steps=[
    ('preprocess', preprocess2),
    ('model', RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model2 = TransformedTargetRegressor(
    regressor=rf_pipe2,
    func=np.log1p,
    inverse_func=np.expm1
)

rf_model2.fit(X_train2, y_train)
pred2 = rf_model2.predict(X_test2)
evaluate("Step 10: RF + volume (log target)", y_test, pred2)


'''
Було побудовано модель прогнозування ціни меблів IKEA на основі розмірів (depth/width/height), категорії, дизайнера та наявності варіантів кольору. Як базову модель використано Ridge регресію з логарифмуванням цілі (log1p(price)), яка показала низьку якість (R²=0.069). Далі застосовано RandomForestRegressor у пайплайні з імпутацією пропусків та one-hot кодуванням категоріальних ознак. Модель Random Forest суттєво покращила результати: R²=0.809, MAE≈348, RMSE≈621. Крос-валідація (cv=5) дала середнє R²≈0.774 ± 0.032, що свідчить про стабільність. Тюнінг параметрів через GridSearchCV не дав суттєвого приросту, підтверджуючи близькість базових налаштувань до оптимальних. Пермутаційна важливість ознак показала, що найбільший вплив має ширина (width), а також категорія та дизайнер. Додавання інженерної ознаки volume (добуток розмірів) дало невелике додаткове покращення (R²=0.810, RMSE≈620).

Щоб ти міг швидко закрити звіт, ось що варто зафіксувати “фінальним”:
Baseline (Ridge, log target): R² = 0.069, MAE ≈ 571.8, RMSE ≈ 1372.8
Final (RandomForest, log target + volume): R² = 0.810, MAE ≈ 346.5, RMSE ≈ 620.1
CV (5-fold) для RF: mean R² ≈ 0.774, std ≈ 0.032
Permutation importance: width (найсильніша), далі category, designer_norm, depth, height, other_colors.

'''