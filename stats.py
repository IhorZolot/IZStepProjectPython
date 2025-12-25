import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load cleaned data
df = pd.read_csv('IZStepProjectPython/ikea_clean.csv')

num_cols = ['price', 'depth', 'height', 'width']
cat_cols = ['name', 'category', 'other_colors', 'designer_norm', 'short_description']

df.info()
print(df.shape)
print(df.head(10))
print(df.tail(10))
print(df[num_cols].describe().round(2))

def all_cat_cols(df: pd.DataFrame, cat_cols: list[str]) -> None:
    for col in cat_cols:
        print(f'Column: {col}')
        print('unique values:', df[col].nunique(dropna=False))
        print('Top 10:')
        print(df[col].value_counts(dropna=False).head(10))
        print('-' * 50)
    return None
        
all_cat_cols(df, cat_cols)

# 1. Visualization of the number of products by category

category_counts = df['category'].value_counts()

fig, ax = plt.subplots(figsize=(12,6))
category_counts.plot(kind='bar', ax=ax)

plt.title('Number of Products by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontweight='bold')
plt.ylabel('Number of Products', fontweight='bold')
plt.xticks(rotation=30, ha='right')

ax.bar_label(ax.containers[0], fontsize=9, padding=1.5, fmt='%.0f')

plt.tight_layout()
plt.show()

# 2. Visualization of median price by category

price_by_category = (
    df.groupby('category', as_index=False)['price']
    .median()
    .sort_values('price', ascending=False)
)

fig, ax = plt.subplots(figsize=(12,6))
price_by_category.plot(kind='bar', ax=ax, x='category', y='price', legend=False)

ax.set_title('Median price by category', fontsize=14, fontweight='bold')
ax.set_xlabel('Category', fontweight='bold')
ax.set_ylabel('Median price', fontweight='bold')
plt.xticks(rotation=30, ha='right')
ax.bar_label(ax.containers[0], fontsize=9, padding=1.5, fmt='%.0f')

plt.tight_layout()
plt.show()

# # 3.  Correlation matrix for numerical features 

df_num = df[num_cols].copy()
corr = df_num.corr(method='pearson')
print(corr.round(2)) 

fig, ax = plt.subplots(figsize=(6, 5))

cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)

ax.set_xticks(np.arange(len(num_cols)))
ax.set_yticks(np.arange(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha='left')
ax.set_yticklabels(num_cols)

ax.set_title('Correlation: price vs dimensions', pad=20, fontsize=14, fontweight='bold')

for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        value = corr.iloc[i, j]
        ax.text(j, i, f'{value:.2f}', va='center', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# # 4. Visualization of the number of products by designer

designer_counts = df['designer_norm'].value_counts().head(15)

fig, ax = plt.subplots(figsize=(12,6))
designer_counts.plot(kind='bar', ax=ax, legend=False)

ax.set_title('Top 15 designers by number of products', fontsize=14, fontweight='bold')
ax.set_xlabel('Designer', fontweight='bold')
ax.set_ylabel('Number of products', fontweight='bold')
plt.xticks(rotation=30, ha='right')
ax.bar_label(ax.containers[0], fontsize=9, padding=1.5, fmt='%.0f')
plt.tight_layout()
plt.show()

# # 5. Count of items with / without color variants and Median price of items with / without color variants

color_stats = (
    df.groupby('other_colors')['price']
      .agg(n_items='size', median_price='median')
      .round(2)
      .reset_index()
)
print(color_stats)
print(df.groupby('other_colors')['price'].describe().round(2))

label_map = {
    'No':  'Without color variants',
    'Yes': 'With color variants',
}
color_stats['label'] = color_stats['other_colors'].map(label_map)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].pie(
    color_stats['n_items'],
    labels=color_stats['label'],
    autopct='%.1f%%',
    startangle=90,
    textprops={'fontweight': 'bold'}
)
ax[0].set_title(
    'Share of items with / without color variants',
    fontsize=14,
    fontweight='bold'
)

prices_no = df.loc[df['other_colors'] == 'No',  'price'].dropna()
prices_yes = df.loc[df['other_colors'] == 'Yes', 'price'].dropna()
labels = [label_map['No'], label_map['Yes']]

ax[1].boxplot(
    [prices_no, prices_yes],
    tick_labels=labels,
    showfliers=False 
)

ax[1].set_title(
    'Price distribution by color variants',
    fontsize=14,
    fontweight='bold'
)
ax[1].set_xlabel('Color variants', fontweight='bold')
ax[1].set_ylabel('Price', fontweight='bold')

plt.tight_layout()
plt.show()

# # 6. Number of items sold / not sold online, median price of items sold / not sold online.

df_online = df.dropna(subset=['price', 'sellable_online']).copy()

label_map = {
    False: 'Not sold online',
    True:  'Sold online',
}

online_stats = (
    df_online.groupby('sellable_online')['price']
             .agg(n_items='size', median_price='median')
             .round(2)
             .reset_index()
)
online_stats['label'] = online_stats['sellable_online'].map(label_map)

print(online_stats)
print(df_online.groupby('sellable_online')['price'].describe().round(2))

prices_off = df_online.loc[df_online['sellable_online'] == False, 'price'].dropna()
prices_on  = df_online.loc[df_online['sellable_online'] == True,  'price'].dropna()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].pie(
    online_stats['n_items'],
    labels=online_stats['label'],
    autopct='%.1f%%',
    startangle=90
)
ax[0].set_title(
    'Share of items: sold / not sold online',
    fontsize=14,
    fontweight='bold'
)

ax[1].boxplot(
    [prices_off, prices_on],
    labels=[label_map[False], label_map[True]],
    showfliers=False
)
ax[1].set_title(
    'Price distribution: sold / not sold online',
    fontsize=14,
    fontweight='bold'
)
ax[1].set_xlabel('Sellable online', fontweight='bold')
ax[1].set_ylabel('Price', fontweight='bold')

med_off = prices_off.median()
med_on  = prices_on.median()
ax[1].text(1, med_off, f'Median: {med_off:.0f}', va='bottom', ha='center', fontsize=9)
ax[1].text(2, med_on,  f'Median: {med_on:.0f}',  va='bottom', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# # 7. Which categories sell the most online?

df_cat = df.dropna(subset=['category', 'sellable_online']).copy()

online_counts = (
    df_cat[df_cat['sellable_online'] == True]
      .groupby('category')
      .size()
      .sort_values(ascending=False)
      .rename('n_online')
)

TOP_N = 10
top_online = online_counts.head(TOP_N)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

top_online.sort_values(ascending=True).plot(kind='barh', ax=ax)

ax.set_xlabel('Number of online items', fontweight='bold')
ax.set_ylabel('Category', fontweight='bold')
ax.set_title('Top 10 categories by number of items sold online', fontsize=14, fontweight='bold')

bars = ax.containers[0]
ax.bar_label(bars, fmt='%.0f', padding=2, fontsize=9)

plt.tight_layout()
plt.show()

# # 8. Median price by designer

median_price_by_designer = (
    df.groupby('designer_norm')['price']
      .median()
      .sort_values(ascending=False)
)
print(median_price_by_designer.head(10).round(2))

stats_by_designer = (
    df.groupby('designer_norm')['price']
      .agg(median_price='median', n_items='size')
      .sort_values('median_price', ascending=False)
)
print(stats_by_designer.head(10))

stats_filtered = stats_by_designer.query('n_items >= 20')
print(stats_filtered.head(20))

TOP_N = 20
top_designers = stats_filtered.head(TOP_N)

fig, ax = plt.subplots(figsize=(10, 8))

bars = ax.barh(top_designers.index, top_designers['median_price'])

ax.set_xlabel('Median price', fontweight='bold')
ax.set_ylabel('Designer', fontweight='bold')
ax.set_title('Top designers by median price (min 20 items)', fontsize=14, fontweight='bold')

ax.bar_label(bars, fmt='%.0f', padding=2, fontsize=8)

plt.tight_layout()
plt.show()

'''
Висновки: 
Отже на першому етапі аналізу даних про продукти IKEA ми подивилися на перші 10 та останні 10 продуктів, а також подивились медіанні цифри. Після цього ми можемо зробити наступні декілька висновків:

1. Найбільше товарів у категорії “Bookcases & shelving units” (432 позиції), далі “Chairs” (392) та “Tables & desks” (365). Натомість найменше товарів у категоріях “Sideboards, buffets & console tables” (6) та “Room dividers” (9). Асортимент зосереджений на масових категоріях для щоденного використання, тоді як спеціалізовані меблі покриваються мінімально, скоріше “для галочки”, ніж як головний напрям бізнесу. 

2. Тут ми вже дивимось не на кількість, а на медіанну ціну в категорії. Найдорожчі категорії за медіанною ціною – це Wardrobes (~1608), Sofas & armchairs (~955) та Beds (~920). Тобто шафи, дивани й ліжка – найбільш «дорогі» типи меблів. Найдешевші категорії – це Children’s furniture (~175), Bookcases & shelving units(~234), Nursery furniture (~305), Café furniture (~397). Це може означати, що найбільший внесок у виручку за штуку йде від великих предметів спальні/вітальні (шафи, дивани, ліжка), а дитячі й частина зберігання позиціонуються як більш бюджетні сегменти. Також помітно, що Bookcases & shelving units, де найбільше товарів за кількістю, мають відносно низьку медіанну ціну — тобто це масовий, але недорогий сегмент.

3. Найсильніші зв’язки – з шириною (0.71) та глибиною (0.65). Тобто чим ширший і глибший предмет, тим, як правило, дорожчий. З висотою зв’язок слабкий (0.25). Висота сама по собі майже не пояснює ціну – важливіше, скільки місця меблі займають по площі, а не по висоті. Між самими розмірами є лише помірні кореляції, а depth–height навіть трохи від’ємна (-0.10) – тобто високі предмети не обов’язково глибокі. Кореляція між ціною та розмірами товарів є слабкою, що свідчить про те, що ціна не обов'язково залежить від фізичних розмірів. 

4. Абсолютний лідер – “IKEA of Sweden” з 567 товарами. Це в кілька разів більше, ніж у будь-якого окремого дизайнера. Далі йдуть окремі дизайнери: Ehlén Johansson (131), Francis Cayouette (118), Ola Wihlborg (100), потім ще кілька авторів і пар-колаборацій із ~90–30 товарами. Отже IKEA значну частину асортименту розробляє внутрішньою командою (“IKEA of Sweden”), а зовнішні/іменні дизайнери доповнюють лінійку окремими серіями. Тобто бренд спирається не стільки на «зіркових» авторів, скільки на власний дизайн-відділ з масовим виробництвом.

5. Структура асортименту за кольорами ~56.7% товарів не мають додаткових кольорових варіантів. ~43.3% – мають кілька кольорів. Тобто більшість позицій продаються в одному кольорі, але частка мультиколірних теж велика – майже половина. Ціна vs наявність кольорових варіантів із boxplot видно, що: Медіанна ціна в групі “With color variants” вища, ніж у товарів без варіантів. Розкид цін теж більший: є як відносно недорогі, так і дуже дорогі товари з кількома кольорами. Отже, товари з кількома кольоровими варіантами загалом коштують дорожче й представлені в ширшому діапазоні цін, тоді як однокольорові – більш масовий і, в середньому, дешевший сегмент.

6. Лівий графік — структура асортименту. Приблизно 99% а точніше майже весь асортимент товарів продаються онлайн, і лише ~1% — тільки офлайн.  Правий графік — відносна медіанна ціна. Якщо порівняти медіанні ціни, то на товари, що продаються онлайн, припадає ≈65% сумарної медіанної ціни, а на тільки офлайн – близько ~35%. Це означає, що медіанна ціна онлайн-товарів майже вдвічі вища, ніж у товарів, які не продаються онлайн.

7. Тут видно «онлайн-зріз» того, що ми вже бачили по всьому асортименту: Найбільше товарів онлайн у категоріях Bookcases & shelving units (431), Chairs (389), Tables & desks (364), Sofas & armchairs (337). Тобто основні категорії зберігання й сидіння максимально представлені в онлайн-каналі. Далі йдуть Outdoor furniture, Beds, Cabinets & cupboards, Wardrobes, Chests of drawers, TV & media furniture – теж масові категорії для дому. Оскільки числа майже збігаються з загальною кількістю товарів у цих категоріях, можна сказати, що майже весь ключовий асортимент цих меблів доступний онлайн – саме на них IKEA робить основний онлайн-фокус.

8. Щоб уникнути перекосів через поодинокі дорогі товари, ми аналізуємо лише тих дизайнерів, у яких у вибірці є щонайменше 20 товарів(хоча можна і 10 і 5, я зупинився на 20). Для дизайнерів, представлених 1–2 позиціями, медіанна ціна не є репрезентативною і може бути штучно завищеною. Найдорожчі колекції - це колаборації: Ehlén Johansson / IKEA of Sweden (~2525) та E Lilja Löwenhielm / K Malmvall (~1834). Тобто спільні серії з відомими дизайнерами позиціонуються як преміум. Самі по собі Ehlén Johansson (~1260), Ebba Strandmark / IKEA of Sweden (~1110), IKEA of Sweden / Ola Wihlborg (~1076) теж вище за середній рівень. Натомість “IKEA of Sweden” окремо має невисоку медіанну ціну (~205), хоча за кількістю товарів є абсолютним лідером. Отже, масовий асортимент робить внутрішня команда IKEA з більш низькими цінами, а преміум-сегмент формують спільні колекції з окремими дизайнерами, де медіанна ціна у кілька разів вища.

'''