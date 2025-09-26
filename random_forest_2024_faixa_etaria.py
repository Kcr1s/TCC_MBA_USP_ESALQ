import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- Etapa 1: Leitura e Pré-processamento de Dados ---
df = pd.read_csv('dados_random_forest.csv', sep=';', decimal=',')
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df.columns = df.columns.str.lower().str.strip()

for col in ['taxa_incidencia', 'taxa_mortalidade']:
    df[col] = df[col].astype(float)

# --- Etapa 2: Treinar e Prever para cada Grupo Etário ---
faixas_etarias = df['faixa_etaria'].unique()
resultados_por_faixa = []

# Iniciar a lista 'metricas_por_faixa' 
metricas_por_faixa = []

for faixa in faixas_etarias:
    print(f"\n--- Análise para o grupo etário: {faixa} ---")

    # 1. Filtrar os dados para a faixa etária atual
    df_faixa = df[df['faixa_etaria'] == faixa].copy()

    if len(df_faixa) < 2:
        print(f"Dados insuficientes para a faixa {faixa}. Pulando.")
        continue

    # Separar os dados de treino (até 2023)
    df_treino = df_faixa[df_faixa['ano'] < 2024]

    # Definir variáveis preditoras e alvo
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
    metricas_por_faixa.append({
        'faixa_etaria': faixa,
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
    metricas_por_faixa.append({
        'faixa_etaria': faixa,
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
    resultados_por_faixa.append({
        'ano': 2024,
        'faixa_etaria': faixa,
        'incidencia_estimada': incidencia_estimada,
        'mortalidade_estimada': mortalidade_estimada
    })

# --- Etapa 3: Exibir e Salvar os resultados finais ---
df_previsoes_2024 = pd.DataFrame(resultados_por_faixa)
df_metricas = pd.DataFrame(metricas_por_faixa)

print("\n--- Previsões para 2024 por Grupo Etário ---")
print(df_previsoes_2024)

print("\n--- Métricas de Avaliação dos Modelos ---")
print(df_metricas)

# --- Etapa 4: Salvar para Excel ---
with pd.ExcelWriter('resultados_random_forest.xlsx') as writer:
    df_previsoes_2024.to_excel(writer, sheet_name='Previsoes_2024', index=False)
    df_metricas.to_excel(writer, sheet_name='Metricas_Avaliacao', index=False)

print("\nDados salvos com sucesso no arquivo 'resultados_random_forest.xlsx'.")
