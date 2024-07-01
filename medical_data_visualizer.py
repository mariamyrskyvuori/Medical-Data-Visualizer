import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Lue data
df = pd.read_csv('medical_examination.csv')

# Laske BMI ja luo 'overweight' sarake
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# Normalisoi data
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# Kategorisen datan visualisointi
def draw_cat_plot():
    categorical_features = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']

    # Muunna data long-formaattiin
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=categorical_features, var_name='variable', value_name='value')


    # Luo catplot
    g = sns.catplot(
        data=df_cat,
        x='variable',
        hue='value',
        col='cardio',
        kind='count',
        height=5,
        aspect=1,
        sharey=False
    )

    # Aseta ylabel jokaiselle akselille

    for ax in g.axes.flat:
        ax.set_ylabel('total')


    # Tallenna kuva
    g.fig.savefig('catplot.png')
    return g.fig


# Lämpökartan luominen
def draw_heat_map():
    # Puhdista data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
            (df['height'] >= df['height'].quantile(0.025)) &
            (df['height'] <= df['height'].quantile(0.975)) &
            (df['weight'] >= df['weight'].quantile(0.025)) &
            (df['weight'] <= df['weight'].quantile(0.975))]

    # Laske korrelaatiomatriisi
    corr = df_heat.corr()

    # Generoi maski yläkolmiolle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Aseta matplotlib-figuuri ja akselit
    fig, ax = plt.subplots(figsize=(12, 8))

    # Piirrä lämpökartta

    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5}, linewidths=.5, ax=ax, vmin=-0.16, vmax=0.32)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-0.08, 0, 0.08, 0.16, 0.24])
    colorbar.set_ticklabels(['-0.08', '0.0', '0.08', '0.16', '0.24'])


    # Tallenna kuva
    fig.savefig('heatmap.png')
    return fig
