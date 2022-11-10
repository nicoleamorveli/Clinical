#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Este es un codigo que convierte las imagenes y los data frame a datos tabulares de latex


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import re
from scipy.stats import chisquare
from scipy import stats
from scipy.stats.contingency import crosstab
from datetime import datetime

sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#Leer datos

dataprocessing=pd.read_excel('basededatoslamaga-ejemplo.xlsx')


# In[86]:


#Ver variables
#variables=dataprocessing.info()


# In[ ]:





# La primera tabla que se va a reportar es la Tabla 2 que resume los datos demograficos cuantitativos (promedio y por grupo) 

# In[31]:


#DESCRIPCION DATOS DEMOGRAFICOS
#TABLA 2
promedios=dataprocessing.groupby('Grupo ').mean().filter(regex='Edad|tA|Peso')
promedios

counts_sexo=pd.crosstab(dataprocessing['Grupo '], dataprocessing['Sexo'])
                        
tabla_2=counts_sexo.join(promedios)
#mandar a latex
print(tabla_2.to_latex(index=True, caption='Descripcion de datos demograficos', label='Tabla 2', position='Tabla 2'))  


# In[32]:


#RESULTADOS DATOS DEMGORAFICOS PRUEBA T 
#TABLA 3
#def ttest_groups(list_of_measures):
def ttest_groups(dataprocessing, grp1, grp2):
    list_of_measures = sorted(dataprocessing.filter(regex='^Edad|tALLA|Peso').columns.tolist())
    dico_v1 = {x: 's={:.5f}, p={:.5f}'.format(
              *ttest_ind(dataprocessing[dataprocessing['Grupo ']==grp1][x], dataprocessing[dataprocessing['Grupo ']==grp2][x])) for x in list_of_measures}
    
    dico_v2 = {key: list(map(str, re.sub('s=|p=', '', value).replace(' ', '').split(','))) for key, value in dico_v1.items()}

    result_df = pd.DataFrame.from_dict(dico_v2, orient='index', columns=['T-test', 'p-value'], dtype=np.float64)
    
    return result_df

results=ttest_groups(dataprocessing, 'A', 'B')
#Reportar resultados
promedios=dataprocessing.filter(regex='Edad|Pe|tA').mean()
promedio_1= pd.DataFrame((promedios),columns = ['Promedios'])
promedio_1
tabla_3=promedio_1.join(results)
print(tabla_3.to_latex(index=True, caption='Tabla de resultados prueba T datos demograficos'))  


# In[33]:


#ANALISIS CHI CUADRADA TABLA CONTIGENCIA CLASE SOCIAL 
#def bondadajuste(dataprocessing):
bondad_datos=dataprocessing['Clase social']
pchi=chisquare(bondad_datos)
chiresults= pd.DataFrame((pchi),index='Chi-square P-value'.split(),columns='Clase-social'.split())
chiresults

#Por grupos 
grupoA= chisquare(dataprocessing[dataprocessing['Grupo ']=='A'].filter(regex='^Clase'))
grupoB= chisquare(dataprocessing[dataprocessing['Grupo ']=='B'].filter(regex='^Clase'))

tabla_4= pd.DataFrame((grupoA, grupoB),index='Clase-socialA Clase-socialB'.split(),columns='Chi-square P-value'.split())

print(tabla_4.to_latex(index=True, caption='Tabla de resultados Chi cuadrada datos demograficos'))  


# In[32]:


#Descripcion datos clinicos
#tabla 5
promedios=dataprocessing.groupby('Grupo ').mean().filter(regex='Fre|Pre|Sat')
promedios.transpose()

df = pd.DataFrame((24,23),index='A B'.split(),columns='n'.split())

tabla=df.join(promedios)
tabla_5=tabla.transpose()
tabla_5.reset_index(inplace=True)
tabla_5 = tabla_5.rename(columns = {'index':'Variable'})
#Anadir hora
tabla_5.loc[tabla_5['Variable'].str.contains('1'),'Hora'] = '7:00 am'
tabla_5.loc[tabla_5['Variable'].str.contains('2'),'Hora'] = '8:00 am'
tabla_5.loc[tabla_5['Variable'].str.contains('3'),'Hora'] = '9:00 am'
tabla_5
#Latex
print(tabla_5.to_latex(index=False, caption='Descripcion datos clinicos'))  


# In[ ]:





# Analisis t-student
# La primera parte hace pruebas estadistica con la t student de datos demograficos y variables de interes
# Utiliza la diferencia de medias
# El codigo define los grupos que se van a comparar y corre la prueba
# Guarda la prueba en un dataframe(resultado T student)

# In[34]:


# PRUEBA T DATOS CLINICOS
#TABLA 6
def ttest_groups(dataprocessing, grp1, grp2):
    list_of_measures = sorted(dataprocessing.filter(regex='^Frecuencia|Presion|Sat').columns.tolist())
    dico_v1 = {x: 's={:.5f}, p={:.5f}'.format(
              *ttest_ind(dataprocessing[dataprocessing['Grupo ']==grp1][x], dataprocessing[dataprocessing['Grupo ']==grp2][x])) for x in list_of_measures}
    
    dico_v2 = {key: list(map(str, re.sub('s=|p=', '', value).replace(' ', '').split(','))) for key, value in dico_v1.items()}

    result_ttest = pd.DataFrame.from_dict(dico_v2, orient='index', columns=['T-test', 'p-value'], dtype=np.float64)
    
    return result_ttest
results=ttest_groups(dataprocessing, 'A', 'B')
#ANadir promedios
promedios=dataprocessing.filter(regex='Fre|Sat|Pre').mean()
promedio_1= pd.DataFrame((promedios),columns = ['Promedios'])
tabla_6=promedio_1.join(results)
#Reset index
tabla_6.reset_index(inplace=True)
tabla_6 = tabla_6.rename(columns = {'index':'Variable'})
#Anadir hora
tabla_6.loc[tabla_6['Variable'].str.contains('1'),'Hora'] = '7:00 am'
tabla_6.loc[tabla_6['Variable'].str.contains('2'),'Hora'] = '8:00 am'
tabla_6.loc[tabla_6['Variable'].str.contains('3'),'Hora'] = '9:00 am'
tabla_6
#Latex
print(tabla_6.to_latex(index=False, caption='Resultados prueba T para datos clinicos'))  


# Mann whitney 
# Esta segunda parte analiza variables cualitativas 
# Guarda la prueba en una dataframe resultado
# Lo mejor es usarla para variable Si/No 

# In[35]:


#MANN WHITNEY DATOS CLINICOS
#TABLA 7
def mannwhitney_groups(dataprocessing, grp1, grp2):
    list_of_measures = sorted(dataprocessing.filter(regex='^Escala').columns.tolist())
    dico_v1 = {x: 's={:.5f}, p={:.5f}'.format(
              *mannwhitneyu(dataprocessing[dataprocessing['Grupo ']==grp1][x], dataprocessing[dataprocessing['Grupo ']==grp2][x])) for x in list_of_measures}
    
    dico_v2 = {key: list(map(str, re.sub('s=|p=', '', value).replace(' ', '').split(','))) for key, value in dico_v1.items()}

    result_manntest = pd.DataFrame.from_dict(dico_v2, orient='index', columns=['T-test', 'p-value'], dtype=np.float64)
    
    return result_manntest

results=mannwhitney_groups(dataprocessing, 'A', 'B')
#Reportar resultados
promedios=dataprocessing.filter(regex='Esca').mean()
promedio_1= pd.DataFrame((promedios),columns = ['Promedios'])
tabla_7=promedio_1.join(results)
tabla_7=promedio_1.join(results)
tabla_7.reset_index(inplace=True)
tabla_7 = tabla_7.rename(columns = {'index':'Variable'})
#Anadir hora
tabla_7.loc[tabla_7['Variable'].str.contains('1'),'Hora'] = '7:00 am'
tabla_7.loc[tabla_7['Variable'].str.contains('2'),'Hora'] = '8:00 am'
tabla_7.loc[tabla_7['Variable'].str.contains('3'),'Hora'] = '9:00 am'
tabla_7
print(tabla_7.to_latex(index=False, caption='Resultados Mann-Whitney datos clinicos'))


# Bondad de ajuste
# Utiliza la prueba chi cuadrada para determinar si una muestra pertenece a una poblacion 
# Analisis demograficos de variables cualitativas (categoricas)
# 

# In[44]:


#CHI CUADRADA DATOS CLINICOS
#TABLA8
#Generar contingency table
contigency_percentage = pd.crosstab(dataprocessing['Grupo '], dataprocessing['Escala-dolor'], normalize='index')
contigency_percentage
#Chi-square test of independence.
c, p, dof, expected = chi2_contingency(contigency_percentage)
p
tabla_8= pd.DataFrame((p),index='P-value'.split(),columns='GrupovsEscalaDolor'.split())
tabla_8
print(tabla_8.to_latex(index=True, caption='Tabla de resultados Chi cuadrada datos clinicos'))  


# Chi cuadrada contigency table
# Solo para las variables significativas de u-mann 
# Utiliza la prueba chi cuadrada para evaluar dos variables categoricas a la vez con mas de dos categorias 
# Cuando el proposito del analisis no es evaluar la diferencia de dos grupos pero la relacion entre dos varaibles categoricas con mas de 2 categorias

# In[39]:





# Datos demograficos 
# Los datos cuantitativos se analizan con la prueba t student *pendiente revisar
# Los datos categoricos cualitativos se analizan con chi cuadrada bondad de ajuste
# Los datos categoricos cualitativos con solo ods opciones se analizan con Mannwhitney

# In[ ]:





# In[ ]:





# Graficas de resultados *Pendiente preguntar*
# Scatterplot recomendado para variables cuantitativas
# Barplot para variables cuantitativas
# Heatmap recomendado para chi cuadrada

# In[ ]:





# In[ ]:





# In[ ]:


#Barplot


# In[ ]:





# In[ ]:




