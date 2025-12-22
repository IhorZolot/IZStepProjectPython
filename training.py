import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load cleaned data
df = pd.read_csv('IZStepProject/ikea_clean.csv')

cols = ['price', 'depth', 'width', 'height', 'category', 'other_colors', 'designer_norm']
df = df[cols].copy()
print(df.shape)

# 1. Alternative features based on medians
dims = ['depth', 'width', 'height']

group_medians = df.groupby('category')[dims].transform('median')
df[dims] = df[dims].fillna(group_medians)
print(df[dims].isna().sum())

# 2. Pipeline with raw features (X, Y)

numeric_features = ['depth', 'width', 'height']
categorical_features = ['category', 'designer_norm', 'other_colors']

X = df[numeric_features + categorical_features].copy()
Y = df['price'].copy()
# Y = df['price_ln'] = np.log1p(df['price'])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

col_prepr = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features),
    ]
)

model_pipeline = Pipeline(steps=[
    ('col_prep', col_prepr),
    ('dtr', DecisionTreeRegressor(max_depth=10, random_state=42)),
])

model_pipeline.fit(X_train, Y_train)
y_pred = model_pipeline.predict(X_test)

print('\nDecisionTreeRegressor with raw features')
print('R^2  : {:.5f}'.format(r2_score(Y_test, y_pred)))
print('MAE  : {:.5f}'.format(mean_absolute_error(Y_test, y_pred)))
print('RMSE : {:.5f}'.format(
    np.sqrt(mean_squared_error(Y_test, y_pred))
))

# 3. Alternative feature set + model comparison

train_tmp = X_train.copy()
train_tmp['price'] = Y_train.values   

median_by_cat  = train_tmp.groupby('category')['price'].median()
median_by_dsgn = train_tmp.groupby('designer_norm')['price'].median()
global_median_price = Y_train.median()

for X_part in (X_train, X_test):
    X_part['category_median_price'] = X_part['category'].map(median_by_cat)
    X_part['designer_median_price'] = (
        X_part['designer_norm']
          .map(median_by_dsgn)
          .fillna(global_median_price)
    )
    
# for X_part in (X_train, X_test):
#     X_part['volume'] = (
#         X_part['depth'] * X_part['width'] * X_part['height']
#     )

#     X_part['category_median_price'] = X_part['category'].map(median_by_cat)
#     X_part['designer_median_price'] = (
#         X_part['designer_norm']
#           .map(median_by_dsgn)
#           .fillna(global_median_price)
#     )

numeric_features_alt = [
    'depth', 'width', 'height',
    'category_median_price', 'designer_median_price'
]
       
X_train_alt = X_train[numeric_features_alt].copy()
X_test_alt  = X_test[numeric_features_alt].copy()
    
def getBestRegressor(X_train, X_test, y_train, y_test):
    models = [
        LinearRegression(),
        LassoCV(),
        RidgeCV(),
        SVR(kernel='linear'),
        KNeighborsRegressor(n_neighbors=16),
        DecisionTreeRegressor(max_depth=10, random_state=42),
        RandomForestRegressor(random_state=42),
        GradientBoostingRegressor(),
    ]

    rows = []

    for model in models:
        name = model.__class__.__name__
        print(name)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append({
            'Model': name,
            'R^2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        })

    df_models = pd.DataFrame(rows).set_index('Model')
    return df_models
  
test1 = getBestRegressor(X_train_alt, X_test_alt, Y_train, Y_test)
print('\nModel comparison on engineered features:')
print(test1.sort_values(by='R^2', ascending=False).round(5))

# 4. GridSearchCV for RandomForestRegressor

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [10, 50, 100],
}
# param_grid = {
#     'n_estimators': [100, 300, 500],
#     'max_depth': [10, 20, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2'],
# }

forest_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

forest_grid.fit(X_train_alt, Y_train)

print('\nGridSearchCV for RandomForestRegressor')
print('Best Estimator :', forest_grid.best_estimator_)
print('Best Score     : {:.5f}'.format(forest_grid.best_score_))

Y_best = forest_grid.predict(X_test_alt) 

print('\nTest metrics for best forest:')
print('R^2  : {:.5f}'.format(r2_score(Y_test, Y_best)))
print('MAE  : {:.5f}'.format(mean_absolute_error(Y_test, Y_best)))
print('RMSE : {:.5f}'.format(
    np.sqrt(mean_squared_error(Y_test, Y_best))
))

print('\nFeature importance:')
print('-------------------------')
for feat, importance in zip(
        X_train_alt.columns,
        forest_grid.best_estimator_.feature_importances_):
    print('{:.5f}   {}'.format(importance, feat))

#  5. Feature importance analysis
best_forest = forest_grid.best_estimator_

importances = best_forest.feature_importances_
feat_names = X_train_alt.columns

fi = pd.DataFrame({
    'feature': feat_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print('\nFeature importances (RandomForest):')
print(fi)

result = permutation_importance(
    best_forest,
    X_test_alt,
    Y_test,
    n_repeats=10,
    random_state=42,
    scoring='r2'
)

perm_importances = result.importances_mean
fi_perm = pd.DataFrame({
    'feature': X_test_alt.columns,
    'importance': perm_importances
}).sort_values('importance', ascending=False)

print('\nPermutation importances (drop in R²):')
print(fi_perm)

# 6. Cross-validation (cross_val_score) for best model

best_forest = forest_grid.best_estimator_

cv_scores = cross_val_score(
    best_forest,
    X_train_alt,
    Y_train,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print('\nCross-validation (cv=5) for best RandomForest:')
print('R^2 per fold :', np.round(cv_scores, 5))
print('Mean R^2     : {:.5f}'.format(cv_scores.mean()))
print('Std R^2      : {:.5f}'.format(cv_scores.std()))

# 7. Visualization 
# 1. Heatmap of correlations of numerical features

num_cols = ['price', 'depth', 'width', 'height']
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

ax.set_xticks(range(len(num_cols)))
ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha='right')
ax.set_yticklabels(num_cols)

for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                ha='center', va='center', color='white')

fig.colorbar(im, ax=ax, label='Correlation')
ax.set_title('Correlation heatmap of numeric features', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()

#  2. Barplot R² by models (baseline)

model_info = test1.copy()  

model_info_sorted = model_info.sort_values(by='R^2', ascending=False)

models = model_info_sorted.index.tolist()
r2_values = model_info_sorted['R^2'].values

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(models, r2_values)
ax.set_ylabel('R² score', fontsize=12, fontweight='bold')
ax.set_title('Model comparison by R² (baseline)', fontsize=14, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
fig.tight_layout()
plt.show()

# 3. Barplot RMSE by model

rmse_values = model_info_sorted['RMSE'].values

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(models, rmse_values)
ax.set_ylabel('RMSE')
ax.set_title('Model comparison by RMSE (baseline)', fontsize=14, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
fig.tight_layout()
plt.show()

#  4.Scatter “Actual vs Predicted” for the best model

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(Y_test, Y_best, alpha=0.5)
min_val = min(Y_test.min(), Y_best.min())
max_val = max(Y_test.max(), Y_best.max())
ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')  

ax.set_xlabel('Actual price', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted price', fontsize=12, fontweight='bold')
ax.set_title('Actual vs Predicted (best RandomForest)', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()

# 5. Histogram residuals

residuals = Y_test - Y_best

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=30)
ax.set_xlabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Histogram of residuals (best RandomForest)', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()

#  6. Barplot grouped feature importance

best_forest = forest_grid.best_estimator_
feat_names = X_train_alt.columns
importances = best_forest.feature_importances_

fi = pd.DataFrame({
    'feature': feat_names,
    'importance': importances
})

groups = {
    'dims': ['depth', 'width', 'height'],
    'category_median_price': ['category_median_price'],
    'designer_median_price': ['designer_median_price'],
}

group_rows = []
for group_name, feats in groups.items():
    group_importance = fi.loc[fi['feature'].isin(feats), 'importance'].sum()
    group_rows.append({'group': group_name, 'importance': group_importance})

fi_grouped = pd.DataFrame(group_rows).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(fi_grouped['group'], fi_grouped['importance'])
ax.set_ylabel('Total importance', fontsize=12, fontweight='bold')
ax.set_title('Grouped feature importance (RandomForest)', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()

#  7. Boxplot R² з cross_val_score
best_forest = forest_grid.best_estimator_

cv_scores = cross_val_score(
    best_forest,
    X_train_alt,
    Y_train,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print('CV R² scores:', np.round(cv_scores, 5))
print('Mean R²:', cv_scores.mean())
print('Std R² :', cv_scores.std())

fig, ax = plt.subplots(figsize=(4, 6))
ax.boxplot(cv_scores, vert=True)
ax.set_xticklabels(['RandomForest'])
ax.set_ylabel('R² score', fontsize=12, fontweight='bold')
ax.set_title('Cross-validation R² (cv=5)', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()

#  8. Barplot feature importance

fi_sorted = fi.sort_values('importance', ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(fi_sorted['feature'], fi_sorted['importance'])
ax.set_ylabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature importances (RandomForestRegressor)', fontsize=14, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

fig.tight_layout()
plt.show()


'''
У цій частині роботи було побудовано та порівняно декілька моделей регресії для прогнозування ціни товарів IKEA. На першому етапі як базову модель було використано DecisionTreeRegressor, що навчався на «сирих» ознаках (геометричні розміри, категорія, наявність варіантів кольору, нормалізоване ім’я дизайнера). Базова модель досягла якості R² ≈ 0,68, середньої абсолютної помилки (MAE) близько 499 та RMSE близько 811, що задає відправну точку для подальших покращень.
Далі було виконано інженерію ознак: для кожної категорії та кожного дизайнера обчислено медіанну ціну (category_median_price та designer_median_price), які додано до набору даних як додаткові предиктори. На цьому розширеному просторі ознак було протестовано низку моделей (лінійну регресію, Ridge, Lasso, SVR, KNN, дерево рішень, Random Forest та Gradient Boosting). Найкращий результат показав RandomForestRegressor: R² ≈ 0,78, MAE ≈ 388 та RMSE ≈ 664. Це означає суттєве зниження як середньої помилки, так і великих відхилень порівняно з базовою моделлю. Крос-валідація (5 фолдів) для найкращого лісу дала середній R² ≈ 0,79 зі стандартним відхиленням близько 0,025, що свідчить про стабільність моделі та відсутність сильного перенавчання.
Аналіз важливості ознак показав, що ключовими факторами, які впливають на ціну, є ширина товару (найвагоміша фіча як за стандартною, так і за пермутаційною важливістю) та медіанна ціна по дизайнеру (designer_median_price). Глибина та висота мають помірний вплив, тоді як медіанна ціна по категорії (category_median_price) додає відносно небагато інформації. Таким чином, поєднання геометричних характеристик із агрегованими ціновими характеристиками дизайнера дозволяє суттєво підвищити якість прогнозу, а модель Random Forest із підібраними гіперпараметрами можна вважати найуспішнішим підходом у рамках цієї роботи.
'''