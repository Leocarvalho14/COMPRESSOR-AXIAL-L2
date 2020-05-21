from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#CRIA DF COM DADOS NORMALIZADOS
def Normaliza_dados(df_, modo='minmax'):
    df = df_.copy()
    if modo == 'minmax':  
        scaler = MinMaxScaler() #chama método de normalização
    if modo == 'standart':
        scaler = StandardScaler()
    ft = df.select_dtypes(include=['float64']).copy() #seleciona somente coluna de dados do tipo "float64"
    df[ft.columns] = scaler.fit_transform(ft).copy()
    return df


#CALCULA MÉDIA MÓVEL 
def Media_Movel(df_on,janela):
    """
    Função que calcula média móvel de todas as variáveis "float64" do DF.
    
    df_on = DF de Entrada
    janela = janela em numero de linhas para cálculo da média móvel
    """
    df_on_rol = df_on.copy()
    df_on_rol[df_on.select_dtypes(include=['float64']).columns] = df_on[df_on.select_dtypes(include=['float64']).columns].rolling(janela).mean()
    df_on_rol.fillna(0, inplace=True)
    return df_on_rol



#PLOT TODAS AS VARIÁEVIS INDIVIDUALMENTE
def plot_todas_variaveis(df, _figsize=(10,20)):
    """
    Plota gráfico individual para cada uma das variáveis tipo    "float64" do  DataFrame.
    """
    i = len(df.select_dtypes(include=['float64']).columns)
    fig,ax = plt.subplots(i, figsize=_figsize) 
    ii = 0;
    for n in df.columns:
        if df[n].dtype != 'float64':
            continue 
        ax[ii].plot(df['DATA'],df[n])
        #ax[ii].ylabel(n)
        ax[ii].legend(loc='lower right')
        ii=ii+1
        
       
#FUNÇÃO DE PLOTAR GRÁFICO E HISTOGRAMA + LIMITES DE 2SIGMA

def plota_todas_variaveis_resumo(data):
    """
    Plota gráfico contendo TIME SERIES e HISTOGRAMA das colunas tipo     'float64' do dataframe (data) 
      *Insere linhas de 2 sigma para ajudar na análse de outliers
      *Criado em 02/10/2019
    """
    i = len(data.select_dtypes(include=['float64']).columns) #i = quantidade de colunas com dados reais

    fig, axes = plt.subplots(nrows=i, ncols=2, figsize=(15,i*5),gridspec_kw={'width_ratios':[3,1]})#, constrained_layout=True) #declara figura
    _ = fig.suptitle('Análise Série Temporal') #insere título na figura

    ii = 0;
    for n in data.columns: #intera função entre todas as colunas = 'float64'
        if data[n].dtype != 'float64':
            continue
        mean = data[n].mean(); sigma = data[n].std() #calcula média e desvio padrão
        upperline = mean+2*sigma #define linha vermelha superior
        lowerline = mean-2*sigma #define linha vermelha inferior
        _ = axes[ii,0].plot(data[n], drawstyle='steps') #plota time series
        _ = axes[ii,0].set(title='Time Series', ylabel=n, xlabel='Tempo')
        _ = axes[ii,0].axhline(y=upperline, linestyle='--', color='red') #insere linhas de 2 sigma
        _ = axes[ii,0].axhline(y=lowerline, linestyle='--', color='red')
        _ = axes[ii,1].set(title='Histogram', xlabel='Qtd. de Pontos') 
        _ = axes[ii,1].hist(data[n],orientation="horizontal",rwidth=1,edgecolor='black', linewidth=1.2) #plota histograma
        _ = axes[ii,1].axhline(y=upperline, linestyle='--', color='red')
        _ = axes[ii,1].axhline(y=lowerline, linestyle='--', color='red')
        ii = ii+1
    

        
#HISTOGRAMA DE TODOS OS DADOS CARACTERIZADOS COM REGRAS EM FORMA DE LISTAS
def plot_kde(df_plot,condicoes): 
    """
    Plota 1D KDE individual para cada uma das variáveis tipo "float64" do DataFrame com regras em forma de listas.
    """
    i = len(df_plot.select_dtypes(include=['float64']).columns)
    fig,ax = plt.subplots(i, figsize=(15,i*3)) 
    ii = 0;
    for n in df_plot.columns:
        if df_plot[n].dtype != 'float64':
            continue 
        for k in condicoes:
            sns.kdeplot(df_plot[k[0]][n], label=k[1], ax=ax[ii])
        
        ax[ii].set_ylabel(n)
        ii=ii+1
        
#PLOTA HEATMAP da correção entre todas as variáveis
def plot_corr(df, data_inic,data_final):
    #plt.close(fig)
    df_plot = df#[df['STATUS']=="FLARE"]  #df_on[(df_on['DATA'] > data_inic) & (df_on['DATA'] < data_final)]
    sum_corr = abs(df_plot.corr()).sum().sort_values(ascending=True).index.values
    a = df_plot.corr()
    #[sum_corr]
    mask = np.zeros_like(a)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(15,10))
    sns.heatmap(a,linewidth=0.5,square=False,annot=True,fmt='.1f',cmap='coolwarm', mask=mask)
    print('Quantidade de Linhas:', len(df) )
    print('Data Inicial:', data_inic )
    print('Data Final:', data_final )
    
#PLOTA VALORES NO TEMPO COM INDICAÇÃO DE PROBLEMA

import matplotlib.patches as mpatches
def Plot_Valores_Tempo(df,variabel_plot,label_colun,label):
    df_lb = df.copy()

    gridsize = (3, 2)
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid(gridsize, (2, 0), colspan=2, rowspan=1)

    df_lb.plot(y=variabel_plot, x='DATA', ax=ax1,linewidth=1);
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_ylabel('Dados')
    ax1.set_title('Dados BOIL OFF ')

    ax2.fill_between(df['DATA'].values, 0, 1, where=df[label_colun]==label,
                facecolor='green', alpha=0.5)#, transform=trans)
    ax2.set_ylabel("Label")

#MONTA SCATTERPLOT PARA COMPARAÇÃO DE VARIÁVEIS

def Scatter_Plot(df,colunas):
    dados = df[colunas]#[df['STATUS']=="FLARE"]
    g = sns.PairGrid(dados)
    g = g.map_upper(sns.scatterplot)
    g = g.map_diag(plt.hist, bins = 30, edgecolor = 'k')
    g = g.map_lower(sns.kdeplot)
    g.fig.set_size_inches(15,15)
    
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpatches


#ANALISA PERIODOS DE MÁQUINA COM PROBLEMA

class Data_Ponto_Mudanca:
    
#Função Cria Mudança de Variaveis
    def __init__(self,coluna_data, coluna_analise, i=10):
        lb_DM = LabelEncoder()
        df_col_copy = pd.DataFrame({'DATA': coluna_data})
        df_col_copy['COLUNA_ANALISE'] = coluna_analise
        df_col_copy['COLUNA_ANALISE_ENC'] = lb_DM.fit_transform(df_col_copy['COLUNA_ANALISE'])
        df_col_copy['AUX_MUDANCA'] = df_col_copy['COLUNA_ANALISE_ENC'].diff()
        df_col_copy['MUDANCA'] = np.where((df_col_copy['AUX_MUDANCA'] != 0) & (df_col_copy['AUX_MUDANCA'].isnull() != 1),1,0) 
        self.df = df_col_copy
        self.i = i
    
    def Ponto_Mudanca(self): 
        return self.df.MUDANCA
    
    def Indic_Mudanca(self): 
        return self.df['AUX_MUDANCA']
    
    def Data_Mudanca(self):
        DataMudanca = self.df.loc[self.df['MUDANCA'] == 1]
        DataMudanca_df = pd.DataFrame({'Data Troca Grade': DataMudanca['DATA'].values})
        return DataMudanca_df
    
    def Label_Mudanca(self):
        self.df['LABEL'] = self.df['MUDANCA'].copy()
        self.df['LABEL'].loc[self.df['MUDANCA'] == 1] = 'FALHA'
        self.df['LABEL'].loc[self.df['MUDANCA'] == 0] = 'NORMAL'
        for n in list(self.df.MUDANCA[self.df['MUDANCA'] == 1].index):
            self.df['LABEL'].iloc[n-self.i:n] = "PRE-FALHA"
        Label_Mudanca_df = pd.DataFrame({'LABEL_MUDANCA': self.df['LABEL'].values})


        return Label_Mudanca_df

def Calc_periodos_problema(df_on):
    DPM = Data_Ponto_Mudanca(df_on['DATA'],df_on["STATUS"])
 #DPM.Data_Mudanca()

    df_on['Indic_Mudanca'] = DPM.Indic_Mudanca()#.loc[2:]
    df_on['Indic_Mudanca'].iloc[0] = 0
    df_on['Indic_Mudanca'].iloc[-1] = 1


    Data_Saidas = df_on['DATA'].loc[df_on['Indic_Mudanca'] == 1]
    Data_Entradas = df_on['DATA'].loc[df_on['Indic_Mudanca'] == -1]
    Datas_Entradas_Saidas = pd.DataFrame({'Data_Entradas':Data_Entradas.values, 'Data_Saidas':Data_Saidas.values})
    Datas_Entradas_Saidas['Intervalo'] = Datas_Entradas_Saidas.Data_Saidas - Datas_Entradas_Saidas.Data_Entradas

    Qtd_Saidas = df_on['Indic_Mudanca'].loc[df_on['Indic_Mudanca'] == -1].count()
    Qtd_Entradas = df_on['Indic_Mudanca'].loc[df_on['Indic_Mudanca'] == 1].count()

    return Datas_Entradas_Saidas

def Bar_Graph_Tendencia(df,modo='Ano'):
    if modo == 'Ano':
        index_=['Ano']
    if modo == 'Mes':
        index_=['Ano','Mes']
    df['Ano'] = df['Data Partida'].dt.year
    df['Mes'] = df['Data Partida'].dt.month
    df['Intervalo_horas'] = (df['Duração'] / np.timedelta64(1, 'h')).round(2)
    pvt_table = pd.pivot_table(df,index=index_,values='Intervalo_horas',aggfunc=[np.sum,len])
    pvt_table['Média 4 meses'] = pvt_table.iloc[:,0].rolling(4).mean().round(2)
    
    fig,ax = plt.subplots(figsize=(16,5))
    pvt_table.iloc[:,[2]].plot(ax=ax)
    pvt_table.iloc[:,[0,1]].plot(kind='bar', ax=ax,title='Quantidade x Duração dos PROBLEMAS no Tempo')
    
    label_ax2 = mpatches.Patch(label='Duração dos períodos em PROBLEMA [horas]')
    label_ax2_2 = mpatches.Patch(color='orange', label='Quantidade de entradas em PROBLEMA')
    
    if modo == 'Ano':
        ax.legend(handles=[label_ax2,label_ax2_2])
        
    if modo == 'Mes':
        label_ax2_3 = mpatches.Patch(color='blue', label='Média Móvel 4 mêses da Duração')
        ax.legend(handles=[label_ax2,label_ax2_2,label_ax2_3])
        
    return pvt_table

def F1_score(df_label, df_label_pred, label_ = "PRE-FALHA"):
    TP = df_label[(df_label == df_label_pred) & (df_label == label_)].count()
    FP = df_label[(df_label != df_label_pred) & (df_label_pred == label_)].count()
    FN = df_label[(df_label != df_label_pred) & (df_label == label_)].count()
    Precision = TP/(TP+FP) # number of correct positive results divided by the number of positive results predicted by the classifier.
    Recall = TP / (TP + FN) # number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive).
    F1 = 2*(1/(1/Precision + 1/Recall))
    
    print('TP:',TP); print('FP:',FP); print('FN:',FN); print('Label:',label_)
    print()
    
    print('Precision: {:.2f} - Quantos % dos Alertas são Verdadeiros?' .format(Precision))
    print('Recall: {:.2f} - Quantos % dos eventos foram detectados?' .format(Recall))
    print('F1_score: {:.2f} - Resumo do recall e precision' .format(F1))
    return

#CALCULAR PCA C/ 2 COMPONENTES E ANALISA REPRESENTATIVIDADE NA VARIÂNCIA
def calcula_pca(df,coluna_label='STATUS'):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df[df.select_dtypes(include=['float64']).columns])
    print('Representatividade da Primeira Componente:', pca.explained_variance_ratio_[0].round(2))
    print('Representatividade da Segunda Componente:', pca.explained_variance_ratio_[1].round(2))
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2']) #cria dataframe
    principalDf['label'] = df[coluna_label].copy()
    principalDf['DATA'] = df['DATA'].copy()
    return principalDf

#PLOTA PCA DE 2 COMPONENTES
def Plot_PCA(principalDf,title="PCA 2 Componentes"):
    fig, ax = plt.subplots(figsize=(10,10)); #sns.set(color_codes=False)
    g = sns.scatterplot(y='principal component 2',x="principal component 1",data=principalDf, hue=principalDf.label)
    plt.xlabel("Componente Principal 1"); plt.ylabel("Componente Principal 2");
    plt.title(title)

#PLOTA 2 GRÁFICOS KDE COMPARANDO LABELS BONS E RUINS
from scipy.stats import kde
def kde_contur(principalDf, nbins=50):
    k = kde.gaussian_kde(principalDf.drop(['DATA','label'],axis=1).T)
    xi, yi = np.mgrid[principalDf['principal component 1'].min():principalDf['principal component 1'].max():nbins*1j, principalDf['principal component 2'].min():principalDf['principal component 2'].max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    return xi, yi, zi

def Plot_KDE_Comaparativo(principalDf, coluna, label_bom, label_ruim, nbins=50):
    """"
    Plota 2 gráficos KDE comparativos entre duas classes de uma determinada coluna do DF.
    principalDF = Dataframe de análise
    coluna = coluna de análise do DF (string)
    label_bom = label de dados normais (string)
    label_ruim = label de dados ruins (string)
    """
   
    xi, yi, zi = kde_contur(principalDf[principalDf[coluna]==label_bom],nbins)
    xi_prob, yi_prob, zi_prob = kde_contur(principalDf[principalDf[coluna]==label_ruim],nbins)
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].set_title("Dados Normal")
    ax[1].set_title("Dados com Problema")
    g = sns.scatterplot(y='principal component 2',x="principal component 1",data=principalDf, ax=ax[0], hue='label')
    g2 = sns.scatterplot(y='principal component 2',x="principal component 1",data=principalDf, ax=ax[1], hue='label')
    ax[0].contour(xi, yi, zi.reshape(xi.shape) )
    ax[1].contour(xi_prob, yi_prob, zi_prob.reshape(xi_prob.shape))
    return
   

def Informacoes_Operacao(df, coluna_data, coluna_status, status_operando_="OPERANDO", status_parado_="PARADO"):
    """
    Entradas:
        1) DataFrame
        2) Nome da Coluna Data
        3) Nome da Coluna Status binário
        
    Saída:
        1) Tempo de máquina operando
        2) Tempo de máquina parada
        3) Quantidade de paradas / partidas
        4) Duração média do tempo operando sem parar
        5) Duração média da máquina parada
        6) Amostra dos tempos de máquina parada com data
        7) Coluna com tempos até parada ou tempo até partida do equipamento
        8) Dataframe com datas de paradas, partidas, duração.
        
    Tuple de Saida:
        1) df_
        2) Data Partidas
        3) Data Paradas
    """
    #cria dataframe auxiliar
    df_ = df[[coluna_data,coluna_status]]
    
    #substitui OPERANDO -> 1 , PARADO -> 1 (OBS: Primeiro valor = PARADO)
    df_[coluna_status].replace(to_replace=[status_operando_,status_parado_],
               value=[1,0], inplace=True)
    
    #força primeira linha da coluna_status para o valor 0 (PARADO)
    #df_.loc[0,coluna_status] = 0
    
    #Cria coluna que calcula a diferença entre valores da coluna_status
    df_['diff'] = df_[coluna_status].diff()
    
    #força primeira linha da coluna "diff" para o valor 1
    #df_.loc[0,'diff'] = 1
    
    #força ultima linha da coluna "diff" para o valor -1
    #df_.loc[df_.index[-1],'diff'] = 1
    
    
    #TABELA OPERANDO
    
    #seleciona data das partidas
    Datas_Partidas = pd.DataFrame({'DATA':df_.loc[df_['diff']==1][coluna_data].reset_index(drop=True)})
    Datas_Partidas.insert(1,'Condição','PARTIDA')
    
    #seleciona data das paradas
    Datas_Paradas = pd.DataFrame({'DATA':df_.loc[df_['diff']==-1][coluna_data].reset_index(drop=True)})
    Datas_Paradas.insert(1,'Condição','PARADA')
    
    #cria tabela de datas das paradas e partidas #possível melhorar usando regra IF iterando para valores != 0
    Datas_paradas_partidas = Datas_Partidas.append(Datas_Paradas).sort_values(by='DATA',ascending=True).reset_index(drop=True)
    
    #cria tabela com paradas e duração
    #ddf = teste[3]

    Tabela_Paradas_Partidas = pd.DataFrame(columns=['Data Partida','Data Parada','Duração'])

    for n in Datas_paradas_partidas[Datas_paradas_partidas['Condição']=="PARTIDA"].index:
        if n+1 < len(Datas_paradas_partidas):
            if Datas_paradas_partidas['Condição'].loc[n+1] == 'PARADA':
                delay = Datas_paradas_partidas['DATA'].loc[n+1] - Datas_paradas_partidas['DATA'].loc[n]

                Tabela_Paradas_Partidas = Tabela_Paradas_Partidas.append({'Data Partida':Datas_paradas_partidas['DATA'].loc[n],
                                                                     'Data Parada':Datas_paradas_partidas['DATA'].loc[n+1],
                                                                 'Duração':delay}, ignore_index=True)
                #print(n)

        #calcula duração das campanhas
        #Duracao_Campanha = Datas_Paradas - Datas_Partidas #diversos problemas
    
    return df_,Datas_Partidas,Datas_Paradas,Datas_paradas_partidas,Tabela_Paradas_Partidas


#Definições:
#1) Um evento é o momento de transição entre um cluster e outro
#2) Um idx_evento é um index que relaciona momentos antes(-) e depois(+) de cada evento.
#3) Uma Janela é o intervalo de dados antes de depois de cada evento.
#4) Um Periodo é o intervalo de dados entre eventos.
#5) Cada evento possui uma descrição na coluna "Desc_Eventos"

"""
Entendendo o código:
 - separa_clusters: entra com dataframe e sai com coluna com N labels INT para N clusters
 
 - coluna_tempo_antes_depois_evento: entra com DF, coluna data STR e coluna status STR e sai com:
     1) Tabela de datas e duração de cada evento
     2) Coluna com index de -X a +X para cada evento
     3) Coluna com identificação de cada periodo (entre eventos)
     4) Coluna com identficação de cada janela (antes e depois dos eventos)

 - tabela_periodos: entra com DF, DATA, coluna label e idx e sai com:
     1) Tabela de dados com datas e duração de cada janela
     2) Coluna de dados com descrição de cada janela (transição de um labal para o outro)
"""
from sklearn.cluster import KMeans

#Separa conjunto de dados em N clusters
def separa_clusters(df_bog,clusters=4):
    kmeans = KMeans(n_clusters=clusters)
    kmeans = kmeans.fit(df_bog)
    return kmeans.predict(df_bog) 


def janelas_eventos(df_aux,coluna_data,coluna_status):
    
    def tabela_entradas_saidas(df,coluna_data, coluna_status):
        df_ = df[[coluna_data,coluna_status]] #cria Df auxiliar
        df_['diff'] = df_[coluna_status].diff() #cria coluna com diferença entre valores de linhas subsequentes
        Datas_Troca = pd.DataFrame({'DATA_ENTRADA':df_.loc[df_['diff']!=0][coluna_data].reset_index(drop=True)}) #cria DF principal com datas onde a diferença é diferente de 0

        for n in Datas_Troca.index: #itera entre valores para calcular duração de cada periodo
            if n+1 < len(Datas_Troca): Datas_Troca.loc[n,'DATA_SAIDA'] = Datas_Troca.iloc[n+1,0] #popula coluna "DATA_SAIDA"

        Datas_Troca['Duração[h]'] = (Datas_Troca['DATA_SAIDA'] - Datas_Troca['DATA_ENTRADA']) / np.timedelta64(1, 'h') #adiciona coluna duração em cada período
        Datas_Troca['Cluster'] = df_.loc[df_['diff']!=0][coluna_status].reset_index(drop=True).astype('object') #adiciona qual cluster o sistema estava operando durante este periodo. (NECESSARIO VALIDAR)
        Datas_Troca['Index_do_Evento'] = df_.loc[df_['diff']!=0].index #Informação de em qual index o evento aconteceu
        Datas_Troca['Nº_do_Evento'] = Datas_Troca.index #Informação do numero do evento 


        return Datas_Troca


    df = df_aux
    Datas_Troca = tabela_entradas_saidas(df,coluna_data,coluna_status)

#     #Calcula idx_evento 
#     next_index = pd.merge_asof(df,Datas_Troca,left_on='DATA',right_on='DATA_ENTRADA',direction='forward')['Index_do_Evento'] 
#     last_index = pd.merge_asof(df,Datas_Troca,left_on='DATA',right_on='DATA_ENTRADA',direction='backward')['Index_do_Evento']

#     idx_ate_evento = -(next_index - df.index)
#     idx_desde_evento = df.index - last_index

#     coluna_idx_pre_pos_evento = np.where(-idx_ate_evento<idx_desde_evento,idx_ate_evento,idx_desde_evento) #cria coluna idx_tempo, referentes a momentos antes e depois do evento.

#     coluna_evento = pd.merge_asof(df,Datas_Troca,left_on='DATA',right_on='DATA_ENTRADA',direction='backward')['Nº_do_Evento'] #popula identificação de qual evento determinado periodo se encontra

#     coluna_janela = [0]
#     ii=0
#     for i in range(0,len(coluna_idx_pre_pos_evento)):
#         if i+1 < len(coluna_idx_pre_pos_evento):
#             if coluna_idx_pre_pos_evento[i+1] - coluna_idx_pre_pos_evento[i] != 1:
#                 ii=ii+1

#             coluna_janela.append(ii) #identificaçao da janela

   
#     df['janela'] = coluna_janela
#     Datas_Troca_janela = tabela_entradas_saidas(df,coluna_data,'janela')

#     df_idx_aux = pd.DataFrame({'coluna_idx_pre_pos_evento':coluna_idx_pre_pos_evento})
#     index_troca = df_idx_aux.loc[df_idx_aux['coluna_idx_pre_pos_evento']==0].index
#     Datas_Troca_janela['idx_Troca'] = index_troca

#     for n in range(len(index_troca)):
#         Datas_Troca_janela.loc[n,'Cluster_Antes'] = str(df.loc[index_troca[n-1],coluna_status])
#         Datas_Troca_janela.loc[n,'Cluster_Depois'] = str(df.loc[index_troca[n],coluna_status])

#     Datas_Troca_janela['Desc_Evento'] = "DE:"+ Datas_Troca_janela['Cluster_Antes'] + " PARA:" + Datas_Troca_janela['Cluster_Depois']

#     desc_janela_df = pd.merge_asof(df,Datas_Troca_janela,left_on='DATA',right_on='DATA_ENTRADA',direction='backward')['Desc_Evento']

    
    
#     Dataframe_saida = pd.DataFrame({
#         'idx_evento':coluna_idx_pre_pos_evento,
#         'ident_evento': coluna_evento,
#         'ident_janela': coluna_janela,
#         'desc_janela:': desc_janela_df
#     })
    
    return Datas_Troca#, Dataframe_saida, Datas_Troca_janela