import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.stats import mannwhitneyu, spearmanr, kruskal, ttest_ind, gaussian_kde

# 1. Load cleaned data
df = pd.read_csv('IZStepProjectPython/ikea_clean.csv')

mask_pos = df['price'].notna() & (df['price'] >= 0)
df = df.loc[mask_pos].reset_index(drop=True)
df['price_ln'] = np.log1p(df['price'])
print(df[['price', 'price_ln']].head(10))

# Hypothesis 1. Products with color options are more expensive

valid_values = ['Yes', 'No']
mask_colors = df['other_colors'].isin(valid_values)

df_colors = df.loc[mask_colors, ['name', 'price', 'price_ln', 'other_colors']].copy()

print("")
print(df_colors['other_colors'].value_counts())
print(df_colors.head())

group_stats = (
    df_colors
      .groupby('other_colors')['price']
      .agg(
          n_items='size',
          median_price='median',
          mean_price='mean',
          q1=lambda s: s.quantile(0.25),
          q3=lambda s: s.quantile(0.75)
      )
      .round(2)
)
print("")
print(group_stats)

group_stats_ln = (
    df_colors
      .groupby('other_colors')['price_ln']
      .agg(
          n_items='size',
          median_price_ln='median',
          mean_price_ln='mean'
      )
      .round(2)
)
print("")
print(group_stats_ln)

prices_yes = df_colors.loc[df_colors['other_colors'] == 'Yes', 'price']
prices_no  = df_colors.loc[df_colors['other_colors'] == 'No',  'price']

prices_yes_ln = df_colors.loc[df_colors['other_colors'] == 'Yes', 'price_ln']
prices_no_ln  = df_colors.loc[df_colors['other_colors'] == 'No',  'price_ln']
print("")
print(len(prices_yes), len(prices_no))

# Test Mann–Whitney U 

stat, p_value = mannwhitneyu(
    prices_yes,
    prices_no,
    alternative='greater' 
)
alpha = 0.05 
print("")
if p_value < alpha:
    print(f'p-value = {p_value:.4f} < {alpha} reject H0')
    print('Conclusion: products with color variations are statistically more expensive than those without.')
else:
    print(f'p-value = {p_value:.4f} ≥ {alpha} fail to reject H0')
    print('Conclusion: there is not enough evidence to claim that products with color variations are more expensive.')

# Test Shapiro–Wilk

sample_yes = prices_yes_ln.sample(min(500, len(prices_yes_ln)), random_state=0)
sample_no  = prices_no_ln.sample(min(500, len(prices_no_ln)), random_state=0)

sh_yes = stats.shapiro(sample_yes)
sh_no  = stats.shapiro(sample_no)

alpha = 0.05
print("")
print('Yes group: W =', sh_yes.statistic, ', p =', sh_yes.pvalue)
if sh_yes.pvalue < alpha:
    print('Reject H0 for Yes: price_ln is NOT normal.')
else:
    print('Fail to reject H0 for Yes: price_ln can be considered normal.')
print('No group: W =', sh_no.statistic, ', p =', sh_no.pvalue)
if sh_no.pvalue < alpha:
    print('Reject H0 for No: price_ln is NOT normal.')
else:
    print('Fail to reject H0 for No: price_ln can be considered normal.')

# Test t-test
res_t = ttest_ind(
    prices_yes_ln,
    prices_no_ln,
    equal_var=False  
)

alpha = 0.05  
print("")
if res_t.pvalue < alpha:
    print(f"p-value = {res_t.pvalue:.4f} < {alpha} reject H0")
    print("Conclusion: mean log price differs between products with and without color variations.")
else:
    print(f"p-value = {res_t.pvalue:.4f} ≥ {alpha} fail to reject H0")
    print("Conclusion: there is no statistically significant difference in mean log price between the groups.")
    
# Test Bootstrap
def bootstrap_diff_means(x, y, n_boot=10_000, func=np.mean, random_state=0):
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    y = np.asarray(y)

    obs = func(x) - func(y) 

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        x_s = rng.choice(x, size=len(x), replace=True)
        y_s = rng.choice(y, size=len(y), replace=True)
        diffs[i] = func(x_s) - func(y_s)

    p_value = (np.abs(diffs) >= abs(obs)).mean()
    return obs, p_value, diffs

obs_raw, p_raw, diffs_raw = bootstrap_diff_means(prices_yes, prices_no)
print("")
print('Bootstrap (price):')
print('  diff_mean =', obs_raw)
print('  p-value   =', p_raw)

obs_ln, p_ln, diffs_ln = bootstrap_diff_means(prices_yes_ln, prices_no_ln)
print("")
print('Bootstrap (log price):')
print('  diff_mean =', obs_ln)
print('  p-value   =', p_ln)

# Visualizations

plt.figure(figsize=(10, 6))
plt.hist(prices_no_ln,  bins=40, alpha=0.6, density=True, label='No color variants')
plt.hist(prices_yes_ln, bins=40, alpha=0.6, density=True, label='With color variants')

xs = np.linspace(
    min(prices_no_ln.min(), prices_yes_ln.min()),
    max(prices_no_ln.max(), prices_yes_ln.max()),
    300
)
kde_no  = gaussian_kde(prices_no_ln)
kde_yes = gaussian_kde(prices_yes_ln)

plt.plot(xs, kde_no(xs),  label='No color variants (KDE)')
plt.plot(xs, kde_yes(xs), label='With color variants (KDE)')
plt.title('Distribution of log(price) by color variants')
plt.xlabel('log(1 + price)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

def ecdf(data):
    data = np.asarray(data.dropna())
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y
x_no,  y_no  = ecdf(prices_no_ln)
x_yes, y_yes = ecdf(prices_yes_ln)

plt.figure(figsize=(10, 6))
plt.step(x_no,  y_no,  where='post', label='No color variants')
plt.step(x_yes, y_yes, where='post', label='With color variants')
plt.title('ECDF of log(price) by color variants', fontsize=14, fontweight='bold')
plt.xlabel('log(1 + price)', fontsize=12, fontweight='bold')
plt.ylabel('ECDF', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.hist(diffs_raw, bins=50, density=True, alpha=0.7)
plt.axvline(0,       linestyle='--', label='0 (no difference)')
plt.axvline(obs_raw, linestyle='-',  label=f'Observed diff = {obs_raw:.2f}')
plt.title('Bootstrap distribution of mean(price_yes) - mean(price_no)', fontsize=14, fontweight='bold')
plt.xlabel('Difference in means (price)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.hist(diffs_ln, bins=50, density=True, alpha=0.7)
plt.axvline(0,      linestyle='--', label='0 (no difference)')
plt.axvline(obs_ln, linestyle='-',  label=f'Observed diff = {obs_ln:.3f}')
plt.title('Bootstrap distribution of mean(log price): Yes - No', fontsize=14, fontweight='bold')
plt.xlabel('Difference in means (log price)', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

'''
Гіпотеза 1. Товари з варіантами кольорів дорожчі, ніж товари без варіантів кольорів.
Було порівняно дві групи: товари з other_colors = "Yes" (n = 1081)та з other_colors = "No" (n = 1417). Медіанна ціна в групі без варіантів кольору становить 449, тоді як у групі з варіантами кольору – 620 (≈ на 38% вище). Середня ціна також вища для товарів з варіантами кольорів(1228 проти 919, приблизно на 34% вище).
Непараметричний тест Mann–Whitney U показав p-value ≈ 1.66·10⁻⁹, що значно менше за рівень значущості α = 0.05,
тому нульову гіпотезу про однаковість розподілів цін було відхилено. Додатково було застосовано t-тест для логарифмованих цін (log(1 + price)), який також показав статистично значущу різницю
Отже, результати аналізу підтримують гіпотезу про те, що товари з варіантами кольорів дорожчі за товари без них.
'''

# Hypothesis 2. The larger the volume of the product, the higher the price

dims = ['depth', 'height', 'width']
for col in dims:
  df[col] = pd.to_numeric(df[col], errors='coerce')

for col in dims:
  df.loc[df[col] <= 0, col] = np.nan

group_means = df.groupby('category')[dims].transform('mean')
df[dims] = df[dims].fillna(group_means)
df_vol = df.dropna(subset=dims).copy()

df_vol['volume'] = (
    df_vol['depth'] *
    df_vol['height'] *
    df_vol['width']
)

df_vol = df_vol[df_vol['volume'] > 0].copy()
df_vol['volume_ln'] = np.log(df_vol['volume'])

print('')
print(df_vol[['price', 'price_ln', 'volume', 'volume_ln']].head())
print(df_vol[dims + ['volume']].describe().round(2))

# Test 1: Spearman correlation

rho_raw, p_raw = spearmanr(df_vol['price'], df_vol['volume'])
rho_ln,  p_ln  = spearmanr(df_vol['price_ln'], df_vol['volume_ln'])

print('')
print('Spearman (price vs volume):    rho =', rho_raw, 'p =', p_raw)
print('Spearman (log price vs log V): rho =', rho_ln,  'p =', p_ln)
    
# Test 2: Permutation test for correlation (or for β in linear regression)

def permutation_corr(x, y, n_perm=10_000, random_state=0):
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    y = np.asarray(y)

    obs_r, _ = stats.pearsonr(x, y)

    perm_rs = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        r_perm, _ = stats.pearsonr(x, y_perm)
        perm_rs[i] = r_perm

    p_value = (np.abs(perm_rs) >= abs(obs_r)).mean()
    return obs_r, p_value, perm_rs

obs_r, p_perm, perm_rs = permutation_corr(
    df_vol['volume_ln'].values,
    df_vol['price_ln'].values
)
print('')
print('Permutation test for Pearson r (log-log):')
print('  r       =', obs_r)
print('  p-value =', p_perm)

# Test 3: Volume partitioning + Kruskal–Wallis

df_vol['volume_quartile'] = pd.qcut(
    df_vol['volume'], 4,
    labels=['Q1 (smallest)', 'Q2', 'Q3', 'Q4 (largest)']
)
print('')
print(df_vol['volume_quartile'].value_counts())

group_price_stats = (
    df_vol
      .groupby('volume_quartile', observed=True)['price']
      .agg(n_items='size', median_price='median', mean_price='mean')
      .round(2)
)
print(group_price_stats)

groups = [
    g['price'].values
    for _, g in df_vol.groupby('volume_quartile', observed=True)
]
stat_kw, p_kw = kruskal(*groups)

print('Kruskal–Wallis (price ~ volume_quartile):')
print('  statistic =', stat_kw)
print('  p-value   =', p_kw)

# Visualizations

plt.figure(figsize=(8, 6))
plt.scatter(df_vol['volume_ln'], df_vol['price_ln'], alpha=0.3, s=10)
plt.xlabel('log(volume)')
plt.ylabel('log(price)')
plt.title('log(price) vs log(volume)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.hexbin(df_vol['volume_ln'], df_vol['price_ln'], gridsize=40)
plt.xlabel('log(volume)')
plt.ylabel('log(price)')
plt.title('log(price) vs log(volume) (hexbin)')
plt.colorbar(label='count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(perm_rs, bins=40, density=True, alpha=0.7)
plt.axvline(obs_r, color='red', linestyle='--', label=f'Observed r = {obs_r:.3f}')
plt.axvline(-obs_r, color='red', linestyle='--', alpha=0.5)

plt.title('Permutation distribution of Pearson r (log(volume), log(price))')
plt.xlabel('r')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

'''
Гіпотеза 2. Ми перевіряємо, чи пов’язана ціна товару з його об’ємом. 
На очищених даних було обчислено Spearman-кореляцію між ціною та об’ємом, а також між логарифмами цих змінних. Коефіцієнт кореляції Spearman склав ρ ≈ 0.64, що свідчить про сильний додатній монотонний зв’язок: товари більшого об’єму, як правило, дорожчі. Додатково проведено пермутаційний тест для коефіцієнта Пірсона між log(price) і log(volume): r ≈ 0.62, p < 0.0001 (жодна з 10 000 перестановок не дала такої ж або сильнішої кореляції). Це підтверджує, що спостережуваний зв’язок навряд чи є випадковим. Для більш інтерпретованого аналізу об’єм було поділено на квартилі, після чого застосовано тест Крускала–Валліса. Медіанна ціна зростає від 145 грн у найменшому квартилі об’єму (Q1) до 1670 грн у найбільшому (Q4). Тест Крускала–Валліса  показав статистично значущі відмінності між квартилями. Візуалізації (розсіювання та hexbin на лог-шкалі) демонструють чіткий зростаючий тренд.
Отже, гіпотеза про те, що більший об’єм товару пов’язаний з вищою ціною, підтверджується.
'''
# Hypothesis 3. The price depends on the designer

df = df.dropna(subset=['designer_norm'])
designer_counts = df['designer_norm'].value_counts()
top_designers = designer_counts[designer_counts >= 20].index

df_des = df[df['designer_norm'].isin(top_designers)].copy()
print('')
print('Number of selected designers:', df_des['designer_norm'].nunique())
print(df_des['designer_norm'].value_counts().head(10))

# Test 1. Kruskal–Wallis on logarithmized price (nonparametric)

groups_ln = [
    g['price_ln'].values
    for _, g in df_des.groupby('designer_norm')
]
# Kruskal–Wallis H-test 
res_kw = stats.kruskal(*groups_ln)

alpha = 0.05
print('')
if res_kw.pvalue < alpha:
    print(f"p-value = {res_kw.pvalue:.4f} < {alpha} → reject H0")
    print("Conclusion: the distribution of log(price) differs across designers.")
else:
    print(f"p-value = {res_kw.pvalue:.4f} ≥ {alpha} → fail to reject H0")
    print("Conclusion: there is no statistically significant evidence that log(price) differs by designer.")

# Test 2. One-way ANOVA on log(price) (parametric)

groups_ln = [
    g['price_ln'].values
    for _, g in df_des.groupby('designer_norm')
]

res_lev = stats.levene(*groups_ln)
print('Levene test for equal variances (price_ln by designer):')
print('  stat   =', res_lev.statistic)
print('  p-val  =', res_lev.pvalue)

res_anova = stats.f_oneway(*groups_ln)

alpha = 0.05
print("\nLevene:")
if res_lev.pvalue < alpha:
    print(f"p = {res_lev.pvalue:.4f} < {alpha} → reject H0 (variances differ)")
else:
    print(f"p = {res_lev.pvalue:.4f} ≥ {alpha} → fail to reject H0 (variances can be treated as equal)")

print("\nANOVA:")
if res_anova.pvalue < alpha:
    print(f"p = {res_anova.pvalue:.4f} < {alpha} → reject H0")
    print("Conclusion: mean log(price) differs across designers.")
else:
    print(f"p = {res_anova.pvalue:.4f} ≥ {alpha} → fail to reject H0")
    print("Conclusion: no statistically significant difference in mean log(price) between designers.")

# Test 3. Permutation test for F-statistics (logarithm effect)
rng = default_rng(0)

def anova_f_for_column(df_sub, col):
    groups = [g[col].values for _, g in df_sub.groupby('designer_norm')]
    res = stats.f_oneway(*groups)
    return res.statistic

def permutation_anova_pvalue(df_sub, col, n_perm=5000, random_state=0):
    rng = default_rng(random_state)

    f_obs = anova_f_for_column(df_sub, col)

    f_perm = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = df_sub['designer_norm'].sample(
            frac=1.0,
            replace=False,
            random_state=rng.integers(0, 1_000_000)
        ).values

        df_perm = df_sub.copy()
        df_perm['designer_perm'] = shuffled

        groups = [
            g[col].values
            for _, g in df_perm.groupby('designer_perm')
        ]
        res = stats.f_oneway(*groups)
        f_perm[i] = res.statistic

    p_value = (f_perm >= f_obs).mean()
    return f_obs, p_value, f_perm

# 3.1. Permutation test on raw price
f_raw, p_raw, f_perm_raw = permutation_anova_pvalue(df_des, 'price', n_perm=5000)
print('Permutation ANOVA (price ~ designer):')
print('  F-observed =', f_raw)
print('  p-value    =', p_raw)

# 3.2. Permutation test on logarithmic price
f_ln, p_ln, f_perm_ln = permutation_anova_pvalue(df_des, 'price_ln', n_perm=5000)
print('Permutation ANOVA (log price ~ designer):')
print('  F-observed =', f_ln)
print('  p-value    =', p_ln)

# Visualization 
# 1: boxplot log-prices by designers (sorted)
designer_order = (
    df_des.groupby('designer_norm')['price_ln']
          .median()
          .sort_values(ascending=False)
          .index
)
data_ln_by_designer = [
    df_des.loc[df_des['designer_norm'] == d, 'price_ln'].values
    for d in designer_order
]

plt.figure(figsize=(12, 6))
plt.boxplot(data_ln_by_designer, tick_labels=designer_order, showfliers=False)
plt.xticks(rotation=30, ha='right')
plt.ylabel('log(1 + price)', fontsize=12, fontweight='bold')
plt.title('Distribution of log(price) by designer (n ≥ 20)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Visualization 2: Bar chart of median price by designer

median_price_by_designer = (
    df_des.groupby('designer_norm')['price']
          .median()
          .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))
plt.bar(median_price_by_designer.index, median_price_by_designer.values)
plt.xticks(rotation=30, ha='right')
plt.ylabel('Median price', fontsize=12, fontweight='bold')
plt.title('Median price by designer (n ≥ 20)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Visualization 3 — permutation distribution F for log(price)

plt.figure(figsize=(8, 6))
plt.hist(f_perm_ln, bins=50, density=True, alpha=0.7)
plt.axvline(f_ln, linestyle='--', label=f'Observed F = {f_ln:.2f}')
plt.xlabel('F-statistic', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')
plt.title('Permutation distribution of F (log price ~ designer)', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

'''
Гіпотеза 3. Ціна товару залежить від дизайнера.
Було відібрано 20 дизайнерів, які мають не менше 20 товарів у каталозі IKEA. Для цих дизайнерів було проаналізовано розподіл логарифмованих цін log(1 + price). Непараметричний тест Kruskal–Wallis показав H = 436.25, p ≈ 1.23·10⁻⁸⁰,
що свідчить про статистично значущі відмінності між дизайнерами за рівнем цін. Параметрична однофакторна ANOVA для log(1 + price) також показала значущі відмінності (F = 30.96, p ≈ 1.16·10⁻⁹⁴), хоча тест Лівена виявив нерівність дисперсій між групами. Пермутаційний тест для F-статистики (5000 пермутацій) дав p < 0.0002 як для сирої ціни, так і для логарифмованої, що підтверджує стійкість результатів ANOVA без жорстких припущень про нормальність та гомогенність дисперсій.
Boxplot та діаграма медіанних цін за дизайнерами показують, що частина дизайнерів формує суттєво дорожчі лінійки товарів, тоді як інші в середньому дешевші. Таким чином, дані однозначно підтверджують гіпотезу про залежність ціни від дизайнера.
'''