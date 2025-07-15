import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import streamlit as st
import os

def gradient_fill(ax, x, y1, y2, color1, color2, **kwargs):
    z = np.empty((100, 1, 4), dtype=float)
    rgb1 = np.array(mcolors.to_rgba(color1))
    rgb2 = np.array(mcolors.to_rgba(color2))
    for i in range(100):
        z[i, 0, :] = (i * rgb2 + (99 - i) * rgb1) / 99
        z[i, 0, 3] = 0.3
    x = np.array([x[0], x[-1]])
    ax.imshow(z, aspect='auto', extent=[x.min(), x.max(), y1.min(), y2.max()], origin='lower', **kwargs)

def add_image(ax, image_path, zoom=0.25, xy=(0.9, -0.1)):
    if os.path.exists(image_path):
        image = plt.imread(image_path)
        imagebox = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(imagebox, xy, frameon=False, xycoords='axes fraction')
        ax.add_artist(ab)

st.title("Análise de Textura")

entrada_modelos = st.text_input("Digite os modelos separados por vírgula:")
entrada_perda_agua = st.text_input("Digite as perdas de água separadas por vírgula:")
entrada_crocancia = st.text_input("Digite as crocâncias médias separadas por vírgula:")

if entrada_modelos and entrada_perda_agua and entrada_crocancia:
    try:
        modelos = [m.strip() for m in entrada_modelos.split(',')]
        perda_agua = list(map(float, entrada_perda_agua.split(',')))
        crocancia_med = list(map(float, entrada_crocancia.split(',')))

        if len(modelos) == len(perda_agua) == len(crocancia_med):
            df = pd.DataFrame({
                'Modelo': modelos,
                'Perda água': perda_agua,
                'Crocância med': crocancia_med
            })

            for produto in df['Modelo'].unique():
                df_produto = df[df['Modelo'] == produto]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df_produto['Perda água'], df_produto['Crocância med'], color='b', label='Dados')

                ax.set_title(f'{produto}')
                ax.set_xlabel('Water loss (%)')
                ax.set_ylabel('Headness (N)')
                ax.set_xticks([i for i in range(0, 71, 2)])
                ax.set_yticks(range(0, 26, 1))
                ax.set_xlim(0, 70)
                ax.set_ylim(0, 25)
                ax.axhline(0, color='gray', linewidth=0.2, zorder=0)
                ax.axvline(0, color='gray', linewidth=0.2, zorder=0)
                ax.grid(color='gray', linestyle='-', linewidth=0.2, zorder=0)

                mean_x = df_produto['Perda água'].mean()
                mean_y = df_produto['Crocância med'].mean()
                ax.scatter(mean_x, mean_y, color='r', marker='o', s=100, label='Ponto de Encontro', zorder=5)

                gradient_fill(ax, [20, 60], np.array([5]), np.array([24.8]), 'lightgreen', 'lightcoral')

                ax.plot([20, 60], [5, 5], color='red', zorder=3)
                ax.plot([20, 20], [0.5, 24.8], color='red', zorder=3)
                ax.plot([60, 60], [0.5, 24.8], color='red', zorder=3)
                ax.plot([20, 60], [10, 10], color='gray', zorder=3)

                ax.text(10, 13, 'Uncooked', color='black', ha='center', va='center', fontsize=20, zorder=15)
                ax.text(65, 13, 'Dry', color='black', ha='center', va='center', fontsize=20, zorder=15)
                ax.text(40, 2, 'Indefinite', color='black', ha='center', va='center', fontsize=20, zorder=15)

                ax.legend()
                add_image(ax, 'foto_1.png')
                st.pyplot(fig)
        else:
            st.error("As listas devem ter o mesmo número de elementos.")
    except ValueError:
        st.error("Por favor, insira apenas números válidos nas perdas de água e crocâncias.")
else:
    st.info("Preencha todos os campos para gerar o gráfico.")
