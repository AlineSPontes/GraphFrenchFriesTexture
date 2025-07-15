import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrow, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import streamlit as st

# Função para criar um gradiente de cores mais claras e transparentes
def gradient_fill(ax, x, y1, y2, color1, color2, **kwargs):
    z = np.empty((100, 1, 4), dtype=float)
    rgb1 = np.array(mcolors.to_rgba(color1))
    rgb2 = np.array(mcolors.to_rgba(color2))
    for i in range(100):
        z[i, 0, :] = (i * rgb2 + (99 - i) * rgb1) / 99
        z[i, 0, 3] = 0.3  # Ajustar a transparência aqui
    x = np.array([x[0], x[-1]])
    im = ax.imshow(z, aspect='auto', extent=[x.min(), x.max(), y1.min(), y2.max()], origin='lower', **kwargs)
    return im

# Função para criar uma caixa com degradê
def gradient_box(ax, x, y, width, height, color1, color2, **kwargs):
    z = np.empty((1, 100, 4), dtype=float)
    rgb1 = np.array(mcolors.to_rgba(color1))
    rgb2 = np.array(mcolors.to_rgba(color2))
    for i in range(100):
        z[0, i, :] = (i * rgb2 + (99 - i) * rgb1) / 99
    im = ax.imshow(z, aspect='auto', extent=[x, x + width, y, y + height], origin='lower', **kwargs)
    return im

# Função para adicionar uma imagem ao gráfico
def add_image(ax, image_path, zoom=0.2, xy=(0.85, 0.15)):
    image = plt.imread(image_path)
    imagebox = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy, frameon=False, xycoords='axes fraction')
    ax.add_artist(ab)

# Solicitar os dados do usuário
st.title("Análise de Textura")
modelos = st.text_input("Digite os modelos separados por vírgula:").split(',')
perda_agua = list(map(float, st.text_input("Digite as perdas de água separadas por vírgula:").split(',')))
crocancia_med = list(map(float, st.text_input("Digite as crocâncias médias separadas por vírgula:").split(',')))

# Criar um DataFrame com os dados fornecidos
data = {
    'Modelo': modelos,
    'Perda água': perda_agua,
    'Crocância med': crocancia_med
}
df = pd.DataFrame(data)

# Obter a lista de produtos únicos
produtos = df['Modelo'].unique()

# Criar gráficos por produto
for produto in produtos:
    df_produto = df[df['Modelo'] == produto]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_produto['Perda água'], df_produto['Crocância med'], color='b', label='Dados')

    # Adicionar título e rótulos aos eixos
    ax.set_title(f'{produto}')
    ax.set_xlabel('Water loss (%)')
    ax.set_ylabel('Headness (N)')

    # Definir intervalos dos eixos
    ax.set_xticks([i for i in range(0, 71, 2)])
    ax.set_yticks(range(0, 26, 1))

    # Definir limites dos eixos
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 25)

    # Adicionar linhas dos eixos x e y mais fracas e atrás dos outros elementos
    ax.axhline(0, color='gray', linewidth=0.2, zorder=0)
    ax.axvline(0, color='gray', linewidth=0.2, zorder=0)
    ax.grid(color='gray', linestyle='-', linewidth=0.2, zorder=0)

    # Marcar um ponto no ponto de encontro (média dos valores)
    mean_perda_agua = df_produto['Perda água'].mean()
    mean_crocancia_med = df_produto['Crocância med'].mean()
    ax.scatter(mean_perda_agua, mean_crocancia_med, color='r', marker='o', s=100, label='Ponto de Encontro', zorder=5)

    # Preencher com um degradê do verde para o laranja (mais claro e transparente)
    gradient_fill(ax, [20, 60], np.array([5]), np.array([24.8]), 'lightgreen', 'lightcoral')

    # Adicionar linhas vermelhas
    ax.plot([20, 60], [5, 5], color='red', zorder=3)
    ax.plot([20, 20], [0.5, 24.8], color='red', zorder=3)
    ax.plot([60, 60], [0.5, 24.8], color='red', zorder=3)

    # Adicionar linha cinza
    ax.plot([20, 60], [10, 10], color='gray', zorder=3)

    # Adicionar palavras no eixo X, alinhadas com a linha do eixo y = 13
    ax.text(10, 13, 'Uncooked', color='black', ha='center', va='center', fontsize=20, zorder=15)
    ax.text(65, 13, 'Dry', color='black', ha='center', va='center', fontsize=20, zorder=15)
    ax.text(40, 2, 'Indefinite', color='black', ha='center', va='center', fontsize=20, zorder=15)

    # Adicionar legenda
    ax.legend()

    # Adicionar imagem ao gráfico
    add_image(ax, 'foto_1.png', zoom=0.25, xy=(0.9, -0.1))

    # Exibir o gráfico
    st.pyplot(fig)