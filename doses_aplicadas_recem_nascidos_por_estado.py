import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# 1. Ler os dados
# ============================

nascidos_vivos = pd.read_csv("nascidos_vivos.csv", index_col=0, sep=';')
doses_aplicadas = pd.read_csv("doses_aplicadas_30dias.csv", index_col=0, sep=';')

# Converte todas as colunas para numérico (int ou float), forçando erros a NaN
nascidos_vivos = nascidos_vivos.apply(pd.to_numeric, errors='coerce')
doses_aplicadas = doses_aplicadas.apply(pd.to_numeric, errors='coerce')

# ============================
# 2. Normalizar nomes de colunas e índices
# ============================

# Remove espaços e garante que as colunas sejam strings
nascidos_vivos.columns = nascidos_vivos.columns.astype(str).str.strip()
doses_aplicadas.columns = doses_aplicadas.columns.astype(str).str.strip()

# Normaliza o índice (nomes dos estados)
nascidos_vivos.index = nascidos_vivos.index.str.strip().str.upper()
doses_aplicadas.index = doses_aplicadas.index.str.strip().str.upper()

# ============================
# 3. Calcular cobertura vacinal (%)
# ============================

cobertura = (doses_aplicadas / nascidos_vivos) * 100
cobertura = cobertura.round(0) #arredonda sem casas decimais

# Limitar entre 0 e 100
cobertura = cobertura.clip(0, 100)

# ============================
# 4. Visualizar os dados
# ============================

print(cobertura.head())

# ============================
# 5. Gerar mapa de calor
# ============================

plt.figure(figsize=(12, 6))

# Azul = valores altos, vermelho = valores baixos
cmap = sns.color_palette("RdYlBu", as_cmap=True)
cmap.set_bad(color="black") # valores ausentes em preto

cobertura.index = cobertura.index.str.title() # Primeira letra maiúscula para cada palavra

sns.heatmap(cobertura, annot=True, fmt="0.0f", cmap="YlGnBu", cbar_kws={'label': 'COBERTURA VACINAL (%)'})
plt.title("Cobertura Vacinal da Hepatite B (idade: <30 dias) no Brasil e regiões (2013-2023)", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("ANO", fontsize=14)
plt.ylabel("BRASIL", fontsize=14)

# Aumentar tamanho das legendas dos ticks
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()

# ============================
