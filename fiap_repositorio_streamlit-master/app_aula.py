#!pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
from pycaret.classification import *
from sklearn.pipeline import Pipeline



def plot_shap_summary(shap_values, Xtest_preprocessed, feature_names):
    plt.figure()
    shap.summary_plot(shap_values, Xtest_preprocessed, feature_names=feature_names, plot_type="bar", show=False)
    st.pyplot(plt.gcf())
    plt.clf()
    
def show_analysis_1(base, treshold):
    #st.write("Análise 1")
    plot_stacked_bar_chart(base, treshold)
    

# def show_analysis_2(ypred):
#     st.write("Análise 2")

# def show_analysis_3(ypred):
#     st.write("Análise 3")

# def show_analysis_4(ypred):
#     st.write("Análise 4")

# def show_analysis_5(ypred):
#     st.write("Análise 5")
    
def plot_stacked_bar_chart(base, treshold):
    # Crie um DataFrame com base nas previsões e no limiar
    base['Group'] = np.where(base['Score_True'] > treshold, 'Propensos', 'Não Propensos')

    # Selecione apenas as colunas das campanhas e o grupo
    campaigns = base[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Group']]
    
    # Renomear as colunas
    for col in campaigns.columns:
        if "Accepted" in col:
            novo_nome = col.replace("Accepted", "")
            campaigns = campaigns.rename(columns={col: novo_nome})

    # Calcule a quantidade de compras em cada campanha para cada grupo
    campaign_counts = campaigns.groupby('Group').sum().reset_index()

    # Calcule a proporção de cada grupo em cada campanha
    campaign_proportions = campaign_counts.set_index('Group').T
    campaign_proportions['Total'] = campaign_proportions['Propensos'] + campaign_proportions['Não Propensos']
    campaign_proportions['Propensos (%)'] = campaign_proportions['Propensos'] / campaign_proportions['Total'] * 100
    campaign_proportions['Não Propensos (%)'] = campaign_proportions['Não Propensos'] / campaign_proportions['Total'] * 100

    # Crie o gráfico de barras
        # Crie o gráfico de barras empilhadas
    plt.figure(figsize=(6, 4))

    # Plote as barras empilhadas
    plt.bar(campaign_proportions.index, campaign_proportions['Propensos (%)'], label='Propensos', color='lightgreen')
    plt.bar(campaign_proportions.index, campaign_proportions['Não Propensos (%)'], bottom=campaign_proportions['Propensos (%)'], label='Não Propensos', color='lightsalmon')

    plt.title('Proporção de Propensos Por Participação em Campanhas')
    plt.ylabel('Proporção de Compras (%)')
    plt.xlabel('')

    # Remover as linhas de fundo do gráfico
    sns.despine(left=True, bottom=True)

    # Ajustar a legenda do gráfico
    plt.legend(title='Grupos', loc='upper left', bbox_to_anchor=(1, 1))

    # Exibir o gráfico no Streamlit
    st.pyplot(plt.gcf())
    plt.clf() 
    
def plot_age_histogram(base, treshold):
    # Crie um DataFrame com base nas previsões e no limiar
    base['Group'] = np.where(base['Score_True'] > treshold, 'Propensos', 'Não Propensos')
    
    # Selecione apenas as colunas 'Age', 'Score_True' e 'Group'
    age_data = base[['Age', 'Score_True', 'Group']]

    # Crie dois DataFrames separados para os grupos 'Propensos' e 'Não Propensos'
    propensos = age_data[age_data['Group'] == 'Propensos']
    nao_propensos = age_data[age_data['Group'] == 'Não Propensos']

    # Determine o número de bins e os intervalos de idade
    num_bins = 20
    bin_edges = np.linspace(age_data['Age'].min(), age_data['Age'].max(), num_bins + 1)

    # Calcule as frequências de idade para cada grupo e normalize-as
    propensos_freq, _ = np.histogram(propensos['Age'], bins=bin_edges)
    nao_propensos_freq, _ = np.histogram(nao_propensos['Age'], bins=bin_edges)

    propensos_perc = propensos_freq / len(propensos) * 100
    nao_propensos_perc = nao_propensos_freq / len(nao_propensos) * 100

    # Crie o histograma de idade em porcentagem para cada grupo
    plt.figure(figsize=(5, 3))
    plt.bar(bin_edges[:-1], propensos_perc, width=np.diff(bin_edges), alpha=0.8, color='lightgreen', label='Propensos', align='edge')
    plt.bar(bin_edges[:-1], nao_propensos_perc, width=np.diff(bin_edges), alpha=0.6, color='lightsalmon', label='Não Propensos', align='edge')

    # Configurar rótulos e estilos do gráfico
    plt.title('Histograma de Idades por Grupo (em Porcentagem)')
    plt.xlabel('Idade')
    plt.ylabel('Porcentagem')
    plt.legend(title='Grupos', loc='upper left', bbox_to_anchor=(1, 1))

    # Remover as linhas de fundo do gráfico
    sns.despine(left=True, bottom=True)

    # Exibir o gráfico no Streamlit
    st.pyplot(plt.gcf())
    plt.clf()

def plot_income_histogram(base, treshold):
    # Crie um DataFrame com base nas previsões e no limiar
    base['Group'] = np.where(base['Score_True'] > treshold, 'Propensos', 'Não Propensos')
    
    # Selecione apenas as colunas 'Age', 'Score_True' e 'Group'
    income_data = base[['Income', 'Score_True', 'Group']]

    # Crie dois DataFrames separados para os grupos 'Propensos' e 'Não Propensos'
    propensos = income_data[income_data['Group'] == 'Propensos']
    nao_propensos = income_data[income_data['Group'] == 'Não Propensos']

    # Determine o número de bins e os intervalos de idade
    num_bins = 20
    bin_edges = np.linspace(income_data['Income'].min(), income_data['Income'].max(), num_bins + 1)

    # Calcule as frequências de idade para cada grupo e normalize-as
    propensos_freq, _ = np.histogram(propensos['Income'], bins=bin_edges)
    nao_propensos_freq, _ = np.histogram(nao_propensos['Income'], bins=bin_edges)

    propensos_perc = propensos_freq / len(propensos) * 100
    nao_propensos_perc = nao_propensos_freq / len(nao_propensos) * 100

    # Crie o histograma de idade em porcentagem para cada grupo
    plt.figure(figsize=(5, 3))
    plt.bar(bin_edges[:-1], propensos_perc, width=np.diff(bin_edges), alpha=0.8, color='lightgreen', label='Propensos', align='edge')
    plt.bar(bin_edges[:-1], nao_propensos_perc, width=np.diff(bin_edges), alpha=0.6, color='lightsalmon', label='Não Propensos', align='edge')
    
    # Configurar rótulos e estilos do gráfico
    plt.title('Histograma por Grupo (em Porcentagem)')
    plt.xlabel('Renda')
    plt.ylabel('Porcentagem')
    plt.legend(title='Grupos', loc='upper left', bbox_to_anchor=(1, 1))

    # Remover as linhas de fundo do gráfico
    sns.despine(left=True, bottom=True)

    # Exibir o gráfico no Streamlit
    st.pyplot(plt.gcf())
    plt.clf()

def plot_histogram(base, treshold,coluna,rotulo):
    # Crie um DataFrame com base nas previsões e no limiar
    base['Group'] = np.where(base['Score_True'] > treshold, 'Propensos', 'Não Propensos')
    
    # Selecione apenas as colunas 'Age', 'Score_True' e 'Group'
    income_data = base[[coluna, 'Score_True', 'Group']]

    # Crie dois DataFrames separados para os grupos 'Propensos' e 'Não Propensos'
    propensos = income_data[income_data['Group'] == 'Propensos']
    nao_propensos = income_data[income_data['Group'] == 'Não Propensos']

    # Determine o número de bins e os intervalos de idade
    num_bins = 20
    bin_edges = np.linspace(income_data[coluna].min(), income_data[coluna].max(), num_bins + 1)

    # Calcule as frequências de idade para cada grupo e normalize-as
    propensos_freq, _ = np.histogram(propensos[coluna], bins=bin_edges)
    nao_propensos_freq, _ = np.histogram(nao_propensos[coluna], bins=bin_edges)

    propensos_perc = propensos_freq / len(propensos) * 100
    nao_propensos_perc = nao_propensos_freq / len(nao_propensos) * 100

    # Crie o histograma de idade em porcentagem para cada grupo
    plt.figure(figsize=(5, 3))
    plt.bar(bin_edges[:-1], nao_propensos_perc, width=np.diff(bin_edges), alpha=0.7, color='lightsalmon', label='Não Propensos', align='edge')
    plt.bar(bin_edges[:-1], propensos_perc, width=np.diff(bin_edges), alpha=0.7, color='lightgreen', label='Propensos', align='edge')

    
    
    # Configurar rótulos e estilos do gráfico
    plt.title('Histograma por Grupo (em Porcentagem)')
    plt.xlabel(rotulo)
    plt.ylabel('Porcentagem')
    plt.legend(title='Grupos', loc='upper left', bbox_to_anchor=(1, 1))

    # Remover as linhas de fundo do gráfico
    sns.despine(left=True, bottom=True)

    # Exibir o gráfico no Streamlit
    st.pyplot(plt.gcf())
    plt.clf()
    
    
    
def display_big_numbers(base):
    # Calcule a média de idade para cada grupo
    avg_age_propensos = base[base['Group'] == 'Propensos']['Age'].mean()
    avg_age_nao_propensos = base[base['Group'] == 'Não Propensos']['Age'].mean()
    
    st.markdown(f"### Média de Idade")
    # Crie duas colunas para exibir os big numbers
    col1, col2 = st.columns(2)

    # Exiba a média de idade dos Propensos na coluna 1
    
    with col1:
        st.markdown(f"### Propensos")
        st.markdown(f"<h2 style='text-align: left; color: lightgreen;'>{avg_age_propensos:.2f}</h2>", unsafe_allow_html=True)

    # Exiba a média de idade dos Não Propensos na coluna 2
    with col2:
        st.markdown(f"### Não Propensos")
        st.markdown(f"<h2 style='text-align: left; color: lightsalmon;'>{avg_age_nao_propensos:.2f}</h2>", unsafe_allow_html=True)

def display_big_numbers_income(base):
    # Calcule a média de idade para cada grupo
    avg_age_propensos = base[base['Group'] == 'Propensos']['Income'].mean()
    avg_age_nao_propensos = base[base['Group'] == 'Não Propensos']['Income'].mean()
    
    st.markdown(f"### Média de Renda")
    # Crie duas colunas para exibir os big numbers
    col1, col2 = st.columns(2)

    # Exiba a média de idade dos Propensos na coluna 1
    
    with col1:
        st.markdown(f"### Propensos")
        st.markdown(f"<h2 style='text-align: left; color: lightgreen;'>{avg_age_propensos:.2f}</h2>", unsafe_allow_html=True)

    # Exiba a média de idade dos Não Propensos na coluna 2
    with col2:
        st.markdown(f"### Não Propensos")
        st.markdown(f"<h2 style='text-align: left; color: lightsalmon;'>{avg_age_nao_propensos:.2f}</h2>", unsafe_allow_html=True)
        
# Adicione isso após a parte "Visualizar Predições:"
# with st.expander("Visualizar Análises:", expanded=False):
#     analysis_options = [
#         "Analisar",
#         "Não analisar",
# #         "Análise 2",
# #         "Análise 3",
# #         "Análise 4",
# #         "Análise 5",
#     ]
#     selected_analysis = st.selectbox("Escolha uma análise:", analysis_options)

#     if selected_analysis == "Analisar":
#         show_analysis_1(ypred,treshold)
#         plot_age_histogram(ypred,treshold)

def plot_stacked_bar_chart(base, treshold):
    # Crie um DataFrame com base nas previsões e no limiar
    base['Group'] = np.where(base['Score_True'] > treshold, 'Propensos', 'Não Propensos')

    # Selecione apenas as colunas das campanhas e o grupo
    campaigns = base[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Group']]
    
    # Renomear as colunas
    for col in campaigns.columns:
        if "Accepted" in col:
            novo_nome = col.replace("Accepted", "")
            campaigns = campaigns.rename(columns={col: novo_nome})

    # Calcule a quantidade de compras em cada campanha para cada grupo
    campaign_counts = campaigns.groupby('Group').sum().reset_index()

    # Calcule a proporção de cada grupo em cada campanha
    campaign_proportions = campaign_counts.set_index('Group').T
    campaign_proportions['Total'] = campaign_proportions['Propensos'] + campaign_proportions['Não Propensos']
    campaign_proportions['Propensos (%)'] = campaign_proportions['Propensos'] / campaign_proportions['Total'] * 100
    campaign_proportions['Não Propensos (%)'] = campaign_proportions['Não Propensos'] / campaign_proportions['Total'] * 100

    # Crie o gráfico de barras
        # Crie o gráfico de barras empilhadas
    plt.figure(figsize=(6, 4))

    # Plote as barras empilhadas
    plt.bar(campaign_proportions.index, campaign_proportions['Propensos (%)'], label='Propensos', color='lightgreen')
    plt.bar(campaign_proportions.index, campaign_proportions['Não Propensos (%)'], bottom=campaign_proportions['Propensos (%)'], label='Não Propensos', color='lightsalmon')

    plt.title('Proporção de Compras por Grupo e Campanha')
    plt.ylabel('Proporção de Compras (%)')
    plt.xlabel('')

    # Remover as linhas de fundo do gráfico
    sns.despine(left=True, bottom=True)

    # Ajustar a legenda do gráfico
    plt.legend(title='Grupos', loc='upper left', bbox_to_anchor=(1, 1))

    # Exibir o gráfico no Streamlit
    st.pyplot(plt.gcf())
    plt.clf() 
    
def plot_stacked_bar_chart_education(base, treshold):
    # Crie um DataFrame com base nas previsões e no limiar
    base['Group'] = np.where(base['Score_True'] > treshold, 'Propensos', 'Não Propensos')

    # Selecione apenas as colunas 'Education' e 'Group'
    education_data = base[['Education', 'Group']]

    # Agrupe os dados por 'Education' e 'Group' e calcule a contagem em cada grupo
    education_counts = education_data.groupby(['Education', 'Group']).size().reset_index(name='Count')

    # Calcule a proporção de cada grupo em cada categoria de 'Education'
    total_counts = education_counts.groupby('Education')['Count'].transform('sum')
    education_counts['Proportion'] = education_counts['Count'] / total_counts * 100

    # Crie um DataFrame com as proporções de cada grupo em cada categoria de 'Education'
    stacked_data = education_counts.pivot_table(index='Education', columns='Group', values='Proportion', fill_value=0).reset_index()

    # Crie o gráfico de barras empilhadas
    plt.figure(figsize=(6, 4))

    # Plote as barras empilhadas
    plt.bar(stacked_data['Education'], stacked_data['Propensos'], label='Propensos', color='lightgreen')
    plt.bar(stacked_data['Education'], stacked_data['Não Propensos'], bottom=stacked_data['Propensos'], label='Não Propensos', color='lightsalmon')

    plt.title('Proporção de Grupos por Educação')
    plt.ylabel('Proporção (%)')
    plt.xlabel('Educação')

    # Remover as linhas de fundo do gráfico
    sns.despine(left=True, bottom=True)

    # Ajustar a legenda do gráfico
    plt.legend(title='Grupos', loc='upper left', bbox_to_anchor=(1, 1))

    # Exibir o gráfico no Streamlit
    st.pyplot(plt.gcf())
    plt.clf()
    
def plot_stacked_bar_chart_marital(base, treshold):
    # Crie um DataFrame com base nas previsões e no limiar
    base['Group'] = np.where(base['Score_True'] > treshold, 'Propensos', 'Não Propensos')

    # Selecione apenas as colunas 'Marital_Status' e 'Group'
    education_data = base[['Marital_Status', 'Group']]

    # Agrupe os dados por 'Education' e 'Group' e calcule a contagem em cada grupo
    education_counts = education_data.groupby(['Marital_Status', 'Group']).size().reset_index(name='Count')

    # Calcule a proporção de cada grupo em cada categoria de 'Education'
    total_counts = education_counts.groupby('Marital_Status')['Count'].transform('sum')
    education_counts['Proportion'] = education_counts['Count'] / total_counts * 100

    # Crie um DataFrame com as proporções de cada grupo em cada categoria de 'Education'
    stacked_data = education_counts.pivot_table(index='Marital_Status', columns='Group', values='Proportion', fill_value=0).reset_index()

    # Crie o gráfico de barras empilhadas
    plt.figure(figsize=(6, 4))

    # Plote as barras empilhadas
    plt.bar(stacked_data['Marital_Status'], stacked_data['Propensos'], label='Propensos', color='lightgreen')
    plt.bar(stacked_data['Marital_Status'], stacked_data['Não Propensos'], bottom=stacked_data['Propensos'], label='Não Propensos', color='lightsalmon')

    plt.title('Proporção de Grupos por Educação')
    plt.ylabel('Proporção (%)')
    plt.xlabel('Educação')

    # Remover as linhas de fundo do gráfico
    sns.despine(left=True, bottom=True)

    # Ajustar a legenda do gráfico
    plt.legend(title='Grupos', loc='upper left', bbox_to_anchor=(1, 1))

    # Exibir o gráfico no Streamlit
    st.pyplot(plt.gcf())
    plt.clf()

def plot_stacked_bar_chart_kidhome(base, treshold):
    # Crie um DataFrame com base nas previsões e no limiar
    base['Group'] = np.where(base['Score_True'] > treshold, 'Propensos', 'Não Propensos')

    # Selecione apenas as colunas 'Marital_Status' e 'Group'
    education_data = base[['Kidhome', 'Group']]
    
    education_data['Kidhome'] = education_data['Kidhome'].astype(str)

    # Agrupe os dados por 'Education' e 'Group' e calcule a contagem em cada grupo
    education_counts = education_data.groupby(['Kidhome', 'Group']).size().reset_index(name='Count')

    # Calcule a proporção de cada grupo em cada categoria de 'Education'
    total_counts = education_counts.groupby('Kidhome')['Count'].transform('sum')
    education_counts['Proportion'] = education_counts['Count'] / total_counts * 100

    # Crie um DataFrame com as proporções de cada grupo em cada categoria de 'Education'
    stacked_data = education_counts.pivot_table(index='Kidhome', columns='Group', values='Proportion', fill_value=0).reset_index()

    # Crie o gráfico de barras empilhadas
    plt.figure(figsize=(6, 4))

    # Plote as barras empilhadas
    plt.bar(stacked_data['Kidhome'], stacked_data['Propensos'], label='Propensos', color='lightgreen')
    plt.bar(stacked_data['Kidhome'], stacked_data['Não Propensos'], bottom=stacked_data['Propensos'], label='Não Propensos', color='lightsalmon')

    plt.title('Proporção de Grupos por Quantidade de Filhos')
    plt.ylabel('Proporção (%)')
    plt.xlabel('Filhos')

    # Remover as linhas de fundo do gráfico
    sns.despine(left=True, bottom=True)

    # Ajustar a legenda do gráfico
    plt.legend(title='Grupos', loc='upper left', bbox_to_anchor=(1, 1))

    # Exibir o gráfico no Streamlit
    st.pyplot(plt.gcf())
    plt.clf()    

def plot_income_boxplot(base, treshold):
    # Crie um DataFrame com base nas previsões e no limiar
    base['Group'] = np.where(base['Score_True'] > treshold, 'Propensos', 'Não Propensos')
    
    # Selecione apenas as colunas 'Income' e 'Group'
    income_data = base[['Income', 'Group']]

    # Crie o gráfico de boxplot
    plt.figure(figsize=(8, 5))

    # Plote o boxplot para cada grupo com cores diferentes
    sns.boxplot(x='Group', y='Income', data=income_data, palette={'Propensos': 'lightgreen', 'Não Propensos': 'lightsalmon'})

    # Configurar rótulos e estilos do gráfico
    plt.title('Boxplot de Renda por Grupo')
    plt.xlabel('Grupo')
    plt.ylabel('Renda')

    # Remover as linhas de fundo do gráfico
    sns.despine(left=True, bottom=True)

    # Exibir o gráfico no Streamlit
    st.pyplot(plt.gcf())
    plt.clf()

    
def plot_age_boxplot(base, treshold):
    # Crie um DataFrame com base nas previsões e no limiar
    base['Group'] = np.where(base['Score_True'] > treshold, 'Propensos', 'Não Propensos')
    
    # Selecione apenas as colunas 'Income' e 'Group'
    income_data = base[['Age', 'Group']]

    # Crie o gráfico de boxplot
    plt.figure(figsize=(8, 5))

    # Plote o boxplot para cada grupo com cores diferentes
    sns.boxplot(x='Group', y='Age', data=income_data, palette={'Propensos': 'lightgreen', 'Não Propensos': 'lightsalmon'})

    # Configurar rótulos e estilos do gráfico
    plt.title('Boxplot de Idade por Grupo')
    plt.xlabel('Grupo')
    plt.ylabel('Idade')

    # Remover as linhas de fundo do gráfico
    sns.despine(left=True, bottom=True)

    # Exibir o gráfico no Streamlit
    st.pyplot(plt.gcf())
    plt.clf()

def plot_stacked_bar_chart2(base, treshold,coluna):
    # Crie um DataFrame com base nas previsões e no limiar
    base['Group'] = np.where(base['Score_True'] > treshold, 'Propensos', 'Não Propensos')

    # Selecione apenas as colunas 'Marital_Status' e 'Group'
    education_data = base[[coluna, 'Group']]

    # Agrupe os dados por 'Education' e 'Group' e calcule a contagem em cada grupo
    education_counts = education_data.groupby([coluna, 'Group']).size().reset_index(name='Count')

    # Calcule a proporção de cada grupo em cada categoria de 'Education'
    total_counts = education_counts.groupby(coluna)['Count'].transform('sum')
    education_counts['Proportion'] = education_counts['Count'] / total_counts * 100

    # Crie um DataFrame com as proporções de cada grupo em cada categoria de 'Education'
    stacked_data = education_counts.pivot_table(index=coluna, columns='Group', values='Proportion', fill_value=0).reset_index()

    # Crie o gráfico de barras empilhadas
    plt.figure(figsize=(6, 4))

    # Plote as barras empilhadas
    plt.bar(stacked_data[coluna], stacked_data['Propensos'], label='Propensos', color='lightgreen')
    plt.bar(stacked_data[coluna], stacked_data['Não Propensos'], bottom=stacked_data['Propensos'], label='Não Propensos', color='lightsalmon')

    plt.title('Proporção de Grupos')
    plt.ylabel('Proporção (%)')
    plt.xlabel(coluna)

    # Remover as linhas de fundo do gráfico
    sns.despine(left=True, bottom=True)

    # Ajustar a legenda do gráfico
    plt.legend(title='Grupos', loc='upper left', bbox_to_anchor=(1, 1))

    # Exibir o gráfico no Streamlit
    st.pyplot(plt.gcf())
    plt.clf()
    

st.set_page_config( page_title = 'Simulador - Case Ifood',
                    page_icon = './images/logo_fiap.png',
                    layout = 'wide',
                    initial_sidebar_state = 'expanded')

st.title('Simulador - Conversão de Vendas')

with st.expander('Descrição do App', expanded = False):
    var_test = 5
    # st.write(var_test)
    # st.markdown(var_test)
    st.write('O objetivo principal deste app é prever quais clientes tem mais propensão a comprarem o produto da campanha de acordo com algumas variáveis.')

with st.sidebar:
    c1, c2 = st.columns(2)
    c1.image('./images/logo_fiap.png', width = 100)
    c2.write('')
    c2.subheader('Auto ML - Fiap [v2]')

    # database = st.selectbox('Fonte dos dados de entrada (X):', ('CSV', 'Online'))
    database = st.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'))

    if database == 'CSV':
        st.info('Upload do CSV')
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')

def get_preprocessed_columns(X):
    pipeline_preprocess = Pipeline(steps=list(mdl_lgbm.named_steps.items())[:-1])
    X_preprocessed = pipeline_preprocess.transform(X)
    return X_preprocessed

#Tela principal
if database == 'CSV':
    if file:
        #carregamento do CSV
        Xtest = pd.read_csv(file)
        #feature_names = Xtest.columns

        #carregamento / instanciamento do modelo pkl
        mdl_lgbm = load_model('./pickle_lgbm_pycaret')
        
        # Acesse o modelo LGBMClassifier dentro do objeto 'Pipeline'
        lgbm_model = mdl_lgbm.named_steps['trained_model']

        # Remova a etapa final do pipeline ('trained_model')
        #preprocessing_pipeline = mdl_lgbm[:-1]

        # Aplique o pipeline de pré-processamento ao conjunto de dados de entrada
        #Xtest_preprocessed = preprocessing_pipeline.transform(Xtest)
        #Xtest_preprocessed = get_preprocessed_columns(Xtest)
        
        # Crie o objeto explainer com o modelo LGBMClassifier
        #explainer = shap.Explainer(lgbm_model)

        # Calcule os valores SHAP
        #shap_values = explainer(Xtest_preprocessed)


        #predict do modelo
        ypred = predict_model(mdl_lgbm, data = Xtest, raw_score = True)
        ypred['Teenhome'] = ypred['Teenhome'].astype(str)

        with st.expander('Visualizar CSV carregado:', expanded = False):
            c1, _ = st.columns([2,4])
            qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:', 
                                    min_value = 5, 
                                    max_value = Xtest.shape[0], 
                                    step = 10,
                                    value = 5)
            st.dataframe(Xtest.head(qtd_linhas))

        with st.expander('Visualizar Predições:', expanded = True):
            c1, _, c2, c3 = st.columns([2,.5,1,1])
            treshold = c1.slider('Treshold (ponto de corte para considerar predição como True)',
                                min_value = 0.0,
                                max_value = 1.0,
                                step = .1,
                                value = .5)
            qtd_true = ypred.loc[ypred['Score_True'] > treshold].shape[0]

            c2.metric('Qtd clientes True', value = qtd_true)
            c3.metric('Qtd clientes False', value = len(ypred) - qtd_true)

            def color_pred(val):
                color = 'olive' if val > treshold else 'orangered'
                return f'background-color: {color}'

            tipo_view = st.radio('', ('Completo', 'Apenas predições'))
            if tipo_view == 'Completo':
                df_view = ypred.copy()
            else:
                df_view = pd.DataFrame(ypred.iloc[:,-1].copy())

            st.dataframe(df_view.style.applymap(color_pred, subset = ['Score_True']))

            csv = df_view.to_csv(sep = ';', decimal = ',', index = True)
            st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
            st.download_button(label = 'Download CSV',
                            data = csv,
                            file_name = 'Predicoes.csv',
                            mime = 'text/csv')
            
        with st.expander("Visualizar Análises:", expanded=False):
            analysis_options = [
                "Analisar",
                "Não analisar",
            ]
            selected_analysis = st.selectbox("Escolha uma análise:", analysis_options)

            if selected_analysis == "Analisar":
                # Crie duas colunas para exibir os gráficos lado a lado
                col1, col2 = st.columns(2)

                # Exiba o primeiro gráfico na coluna 1
                with col1:
                    show_analysis_1(ypred, treshold)
                    plot_stacked_bar_chart_education(ypred, treshold)
                    plot_age_boxplot(ypred, treshold)
                    plot_histogram(ypred, treshold,'Age','Age')
                    plot_histogram(ypred, treshold,'MntFishProducts','MntFishProducts')
                    plot_histogram(ypred, treshold,'MntGoldProds','MntGoldProds')
                    plot_histogram(ypred, treshold,'MntSweetProducts','MntSweetProducts')
                    plot_stacked_bar_chart2(ypred, treshold, 'NumCatalogPurchases')
                    plot_stacked_bar_chart2(ypred, treshold, 'NumStorePurchases')
                    plot_stacked_bar_chart2(ypred, treshold, 'NumWebVisitsMonth')
                    plot_histogram(ypred, treshold,'Recency','Recency')
        #             plot_shap_summary(shap_values, Xtest_preprocessed, feature_names)




                # Exiba o segundo gráfico na coluna 2
                with col2:
                    plot_stacked_bar_chart_kidhome(ypred, treshold)
                    plot_stacked_bar_chart_marital(ypred, treshold)
                    plot_income_boxplot(ypred, treshold)
                    plot_histogram(ypred, treshold,'Income','Income')
                    plot_histogram(ypred, treshold,'MntFruits','MntFruits')
                    plot_histogram(ypred, treshold,'MntMeatProducts','MntMeatProducts')
                    plot_histogram(ypred, treshold,'MntWines','MntWines')
                    plot_stacked_bar_chart2(ypred, treshold, 'NumDealsPurchases')
                    plot_stacked_bar_chart2(ypred, treshold, 'NumWebPurchases')
                    plot_stacked_bar_chart2(ypred, treshold, 'Teenhome')
                    display_big_numbers(ypred)
                    display_big_numbers_income(ypred)


                # Exiba os big numbers com a média de idade dos grupos


        #     elif selected_analysis == "Análise 2":
        #         show_analysis_2(ypred)
        #     elif selected_analysis == "Análise 3":
        #         show_analysis_3(ypred)
        #     elif selected_analysis == "Análise 4":
        #         show_analysis_4(ypred)
        #     elif selected_analysis == "Análise 5":
        #         show_analysis_5(ypred)

            

    else:
        st.warning('Arquivo CSV não foi carregado')
        # st.info('Arquivo CSV não foi carregado')
        # st.error('Arquivo CSV não foi carregado')
        # st.success('Arquivo CSV não foi carregado')

elif database == 'Online':
    st.subheader("Simulação de entrada de dados")
    
    with st.sidebar:
        st.write("Preencha as informações do cliente:")
        col1, col2 = st.columns(2)
        
        # Exemplo de campo numérico
        AcceptedCmp1 = col1.number_input("AcceptedCmp1", min_value=0, max_value=1)
        AcceptedCmp2 = col1.number_input("AcceptedCmp2", min_value=0, max_value=1)
        AcceptedCmp3 = col1.number_input("AcceptedCmp3", min_value=0, max_value=1)
        AcceptedCmp4 = col1.number_input("AcceptedCmp4", min_value=0, max_value=1)
        AcceptedCmp5 = col1.number_input("AcceptedCmp5", min_value=0, max_value=1)
        Age = col1.number_input("Age", min_value=5, max_value=100)
        Complain = col1.number_input("Complain", min_value=0, max_value=1)
        Education = col1.selectbox("Education", options=['Graduation','Master','Basic','2n Cycle','PhD'])
        Income = col1.number_input("Income", min_value=0, max_value=200000)
        Kidhome = col1.number_input("Kidhome", min_value=0, max_value=5)
        Marital_Status = col1.selectbox("Marital_Status", options=['Together','Married','Widow','Divorced','Single'])
        MntFishProducts = col1.number_input("MntFishProducts", min_value=0, max_value=1500)
        MntFruits = col1.number_input("MntFruits", min_value=0, max_value=1500)
        MntGoldProds = col1.number_input("MntGoldProds", min_value=0, max_value=1500)
        MntMeatProducts = col1.number_input("MntMeatProducts", min_value=0, max_value=1500)
        MntSweetProducts = col1.number_input("MntSweetProducts ", min_value=0, max_value=1500)
        MntWines = col1.number_input("MntWines", min_value=0, max_value=1500)
        NumCatalogPurchases = col1.number_input("NumCatalogPurchases", min_value=0, max_value=20)
        NumDealsPurchases = col1.number_input("NumDealsPurchases", min_value=0, max_value=20)
        NumStorePurchases = col1.number_input("NumStorePurchases", min_value=0, max_value=20)
        NumWebPurchases = col1.number_input("NumWebPurchase", min_value=0, max_value=20)
        NumWebVisitsMonth = col1.number_input("NumWebVisitsMonth", min_value=0, max_value=20)
        Recency = col1.number_input("Recency", min_value=0, max_value=200)
        Teenhome = col1.number_input("Teenhome", min_value=0, max_value=1)
        Time_Customer = col1.number_input("Time_Customer", min_value=0, max_value=10000)
        
        
        # Crie um DataFrame com os valores inseridos pelo usuário
        Xtest = pd.DataFrame(data=[[AcceptedCmp1,AcceptedCmp2,AcceptedCmp3,AcceptedCmp4,AcceptedCmp5,Age,Complain,Education,Income,Kidhome,Marital_Status,MntFishProducts,MntFruits,MntGoldProds,MntMeatProducts,MntSweetProducts,MntWines,NumCatalogPurchases,NumDealsPurchases,NumStorePurchases,NumWebPurchases,NumWebVisitsMonth,Recency,Teenhome,Time_Customer]], columns=['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Age','Complain','Education','Income','Kidhome','Marital_Status','MntFishProducts','MntFruits','MntGoldProds','MntMeatProducts','MntSweetProducts','MntWines','NumCatalogPurchases','NumDealsPurchases','NumStorePurchases','NumWebPurchases','NumWebVisitsMonth','Recency','Teenhome','Time_Customer'])

        #carregamento / instanciamento do modelo pkl
        mdl_lgbm = load_model('./pickle_lgbm_pycaret')
        
        # Acesse o modelo LGBMClassifier dentro do objeto 'Pipeline'
        #lgbm_model = mdl_lgbm.named_steps['trained_model']

        #predict do modelo
        ypred = predict_model(mdl_lgbm, data = Xtest, raw_score = True)
        #ypred['Teenhome'] = ypred['Teenhome'].astype(str)

    with st.expander('Visualizar CSV carregado:', expanded = False):
        c1, _ = st.columns([2,4])
        qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:', 
                                min_value = 5, 
                                max_value = Xtest.shape[0], 
                                step = 10,
                                value = 5)
        st.dataframe(Xtest.head(qtd_linhas))

    with st.expander('Visualizar Predições:', expanded = True):
        c1, _, c2, c3 = st.columns([2,.5,1,1])
        treshold = c1.slider('Treshold (ponto de corte para considerar predição como True)',
                            min_value = 0.0,
                            max_value = 1.0,
                            step = .1,
                            value = .5)
        qtd_true = ypred.loc[ypred['Score_True'] > treshold].shape[0]

        c2.metric('Qtd clientes True', value = qtd_true)
        c3.metric('Qtd clientes False', value = len(ypred) - qtd_true)

        def color_pred(val):
            color = 'olive' if val > treshold else 'orangered'
            return f'background-color: {color}'

        tipo_view = st.radio('', ('Completo', 'Apenas predições'))
        if tipo_view == 'Completo':
            df_view = ypred.copy()
        else:
            df_view = pd.DataFrame(ypred.iloc[:,-1].copy())

        st.dataframe(df_view.style.applymap(color_pred, subset = ['Score_True']))

        csv = df_view.to_csv(sep = ';', decimal = ',', index = True)
        st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
        st.download_button(label = 'Download CSV',
                        data = csv,
                        file_name = 'Predicoes.csv',
                        mime = 'text/csv')
    # Aplique o pipeline de pré-processamento ao conjunto de dados de entrada do usuário
    #user_input_preprocessed = preprocessing_pipeline.transform(user_input)

    # Faça a previsão
    #user_prediction = predict_model(mdl_lgbm, data=user_input, raw_score=True)

    # Exiba a previsão
    #st.subheader("Previsão")
    #st.write("Probabilidade de conversão:", user_prediction['Score_True'].values[0])

else:
    st.error('Esta opção será desenvolvida no Entregável 1 da disciplina')



    

        
