import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- Etapa 1: Leitura e Pré-processamento de Dados ---
# O arquivo deve conter as colunas: ano, regiao_geografica, taxa_incidencia, taxa_mortalidade
df = pd.read_csv('incidencia_regioes_2024.csv', sep=';', decimal=',')
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df.columns = df.columns.str.lower().str.strip()

# Renomear a coluna de iteração para garantir consistência
COLUNA_ITINERANTE = 'regiao_geografica' 
df = df.rename(columns={'regiao': COLUNA_ITINERANTE}) # Ajuste o nome da coluna aqui se for diferente

for col in ['taxa_incidencia', 'taxa_mortalidade']:
    df[col] = df[col].astype(float)

# --- Etapa 2: Treinar e Prever para cada Região Geográfica ---
regioes_geograficas = df[COLUNA_ITINERANTE].unique()
resultados_por_regiao = []
metricas_por_regiao = []

for regiao in regioes_geograficas:
    print(f"\n--- Análise para a região: {regiao} ---")

    # 1. Filtrar os dados para a região atual
    df_regiao = df[df[COLUNA_ITINERANTE] == regiao].copy()

    if len(df_regiao) < 2:
        print(f"Dados insuficientes para a região {regiao}. Pulando.")
        continue

    # Separar os dados de treino (até 2023)
    df_treino = df_regiao[df_regiao['ano'] < 2024]

    # Definir variáveis preditoras e alvo (usando 'ano' como preditor)
    X = df_treino[['ano']]

    # Dividir em treino e teste para avaliar o modelo
    X_train, X_test, y_train, y_test = train_test_split(X, df_treino['taxa_incidencia'],
                                                        test_size=0.2, random_state=42)

    # --- MODELO DE INCIDÊNCIA ---
    print("Avaliação do Modelo de Incidência:")
    modelo_incidencia = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_incidencia.fit(X_train, y_train)
    y_pred_incidencia = modelo_incidencia.predict(X_test)

    mae_inc = mean_absolute_error(y_test, y_pred_incidencia)
    rmse_inc = np.sqrt(mean_squared_error(y_test, y_pred_incidencia))
    r2_inc = r2_score(y_test, y_pred_incidencia)

    print(f'MAE: {mae_inc:.4f}')
    print(f'RMSE: {rmse_inc:.4f}')
    print(f'R²: {r2_inc:.4f}')

    # Armazenar métricas para incidência
    metricas_por_regiao.append({
        COLUNA_ITINERANTE: regiao,
        'modelo': 'incidencia',
        'MAE': mae_inc,
        'RMSE': rmse_inc,
        'R²': r2_inc
    })

    # --- MODELO DE MORTALIDADE ---
    print("\nAvaliação do Modelo de Mortalidade:")
    y_mortalidade_train = df_treino.loc[y_train.index, 'taxa_mortalidade']
    y_mortalidade_test = df_treino.loc[y_test.index, 'taxa_mortalidade']

    modelo_mortalidade = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_mortalidade.fit(X_train, y_mortalidade_train)
    y_pred_mortalidade = modelo_mortalidade.predict(X_test)

    mae_mort = mean_absolute_error(y_mortalidade_test, y_pred_mortalidade)
    rmse_mort = np.sqrt(mean_squared_error(y_mortalidade_test, y_pred_mortalidade))
    r2_mort = r2_score(y_mortalidade_test, y_pred_mortalidade)

    print(f'MAE: {mae_mort:.4f}')
    print(f'RMSE: {rmse_mort:.4f}')
    print(f'R²: {r2_mort:.4f}')

    # Armazenar métricas para mortalidade
    metricas_por_regiao.append({
        COLUNA_ITINERANTE: regiao,
        'modelo': 'mortalidade',
        'MAE': mae_mort,
        'RMSE': rmse_mort,
        'R²': r2_mort
    })

    # 3. Fazer as previsões para 2024
    dados_2024 = pd.DataFrame({'ano': [2024]})

    incidencia_estimada = modelo_incidencia.predict(dados_2024)[0]
    mortalidade_estimada = modelo_mortalidade.predict(dados_2024)[0]

    # 4. Armazenar os resultados
    resultados_por_regiao.append({
        'ano': 2024,
        COLUNA_ITINERANTE: regiao,
        'incidencia_estimada': incidencia_estimada,
        'mortalidade_estimada': mortalidade_estimada
    })

# --- Etapa 3: Exibir e Salvar os resultados finais ---
df_previsoes_2024 = pd.DataFrame(resultados_por_regiao)
df_metricas = pd.DataFrame(metricas_por_regiao)

print("\n--- Previsões para 2024 por Região Geográfica ---")
print(df_previsoes_2024)

print("\n--- Métricas de Avaliação dos Modelos ---")
print(df_metricas)

# --- Etapa 4: Salvar para Excel ---
with pd.ExcelWriter('resultados_random_forest_regiao.xlsx') as writer:
    df_previsoes_2024.to_excel(writer, sheet_name='Previsoes_2024', index=False)
    df_metricas.to_excel(writer, sheet_name='Metricas_Avaliacao', index=False)

print("\nDados salvos com sucesso no arquivo 'resultados_random_forest_regiao.xlsx'.")