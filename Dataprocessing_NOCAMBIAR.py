#!/usr/bin/env python
# coding: utf-8

# In[248]:


#Este es un codigo que:
#Realiza las pruebas estadisticas
# Agregar columna con hora a los resultados
# Agregar nombre al scatterplot
# Exportar como tabla a excel 


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
from datetime import time

sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[262]:


#Leer datos

dataprocessing=pd.read_excel('basededatoslamaga-ejemplo.xlsx')
dataprocessing


# In[3]:


#Ver variables
variables=dataprocessing.info()


# In[4]:


#Estadistica descriptiva por grupos
grupos=dataprocessing.groupby('Grupo ').describe()
grupos


# In[5]:


#Ver datos mas especificos de la estadistica descriptiva
#Datos de interes
#dataframe se llama como la variable
Escaladolor=grupos['Escala-dolor']
Escaladolor


# In[22]:


#Descripcion datos demograficos
promedios=dataprocessing.groupby('Grupo ').mean().filter(regex='Edad|tA|Peso')
promedios

counts_sexo=pd.crosstab(dataprocessing['Grupo '], dataprocessing['Sexo'])
                        
counts_sexo.join(promedios)



# In[242]:


#Descripcion datos clinicos
promedios=dataprocessing.groupby('Grupo ').mean().filter(regex='Fre|Pre|Sat')
promedios.transpose()

df = pd.DataFrame((24,23),index='A B'.split(),columns='n'.split())

tabla_5=df.join(promedios)
tabla_5


# Analisis t-student
# La primera parte hace pruebas estadistica con la t student de datos demograficos y variables de interes
# Utiliza la diferencia de medias
# El codigo define los grupos que se van a comparar y corre la prueba
# Guarda la prueba en un dataframe(resultado T student)

# In[215]:


#crear columna para hora
#Esta celda generar horas para las variables 
milista_clinicas=sorted(dataprocessing.filter(regex='Fre|Sat|Pre').columns.tolist())
primera_hora= ['1']
segunda_hora=['2']
tercera_hora=['3']
for t in primera_hora:
    for i in range(len(milista_clinicas)):
        milista_clinicas[i] = milista_clinicas[i].replace(t, '-7:00 am')
milista_clinicas
for t in segunda_hora:
    for i in range(len(milista_clinicas)):
        milista_clinicas[i] = milista_clinicas[i].replace(t,'-8:00 am')
milista_clinicas
for t in tercera_hora:
    for i in range(len(milista_clinicas)):
        milista_clinicas[i] = milista_clinicas[i].replace(t,'-9:00 am')
milista_clinicas

tuple_lista = [item.split('-') for item in milista_clinicas]
res1, res2 = map(list, zip(*tuple_lista))
df = pd.DataFrame(tuple_lista, columns = ['Variable', 'Hora'])
df
#Imprimir como tabular y poner en latex


# Seleccionar hora
# primera_hora = filter(lambda a: '1' in a, milista_clinicas)
# Convert the filter object to list
# print(list(primera_hora))
# 
# horas=pd.timedelta_range(start='7:00:00', periods=4, freq='h')
# df = pd.DataFrame({'Horas': horas})
# print(df)
# dataprocessing['title'] = df['title'].apply(lambda title: title.split(':')[0])

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[237]:


#Codigo que procesa los datos clinicos con t student
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
res3=promedio_1.join(results)
res3.reset_index(inplace=True)
res3 = res3.rename(columns = {'index':'Variable'})
#Anadir hora
res3.loc[res3['Variable'].str.contains('1'),'Hora'] = '7:00 am'
res3.loc[res3['Variable'].str.contains('2'),'Hora'] = '8:00 am'
res3.loc[res3['Variable'].str.contains('3'),'Hora'] = '9:00 am'
res3


# In[231]:





# Mann whitney 
# Esta segunda parte analiza variables cualitativas 
# Guarda la prueba en una dataframe resultado
# Lo mejor es usarla para variable Si/No 

# In[240]:



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
res2=promedio_1.join(results)
res2.reset_index(inplace=True)
res2 = res2.rename(columns = {'index':'Variable'})
#Anadir hora
res2.loc[res2['Variable'].str.contains('1'),'Hora'] = '7:00 am'
res2.loc[res2['Variable'].str.contains('2'),'Hora'] = '8:00 am'
res2.loc[res2['Variable'].str.contains('3'),'Hora'] = '9:00 am'
res2


# Bondad de ajuste
# Utiliza la prueba chi cuadrada para determinar si una muestra pertenece a una poblacion 
# Analisis demograficos de variables cualitativas (categoricas)
# 

# In[736]:


#def bondadajuste(dataprocessing):
bondad_datos=dataprocessing['Clase social']
pchi=chisquare(bondad_datos)
chiresults= pd.DataFrame((pchi),index='Chi-square P-value'.split(),columns='Clase-social'.split())
chiresults

#Por grupos 
grupoA= chisquare(dataprocessing[dataprocessing['Grupo ']=='A'].filter(regex='^Clase'))
grupoB= chisquare(dataprocessing[dataprocessing['Grupo ']=='B'].filter(regex='^Clase'))

chiresults= pd.DataFrame((grupoA, grupoB),index='Clase-socialA Clase-socialB'.split(),columns='Chi-square P-value'.split())
chiresults


# Chi cuadrada contigency table
# Solo para las variables significativas de u-mann 
# Utiliza la prueba chi cuadrada para evaluar dos variables categoricas a la vez con mas de dos categorias 
# Cuando el proposito del analisis no es evaluar la diferencia de dos grupos pero la relacion entre dos varaibles categoricas con mas de 2 categorias

# In[767]:


#Generar contingency table
contigency_percentage = pd.crosstab(dataprocessing['Grupo '], dataprocessing['Escala-dolor'], normalize='index')
contigency_percentage
#Chi-square test of independence.
c, p, dof, expected = chi2_contingency(contigency)
p
chiresults= pd.DataFrame((p),index='P-value'.split(),columns='GrupovsEscalaDolor'.split())
chiresults
   


# Datos demograficos 
# Los datos cuantitativos se analizan con la prueba t student *pendiente revisar
# Los datos categoricos cualitativos se analizan con chi cuadrada bondad de ajuste
# Los datos categoricos cualitativos con solo ods opciones se analizan con Mannwhitney

# In[763]:


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
promedio_1.join(results)


# In[254]:


#sns.set_style('ticks')
#palette = sns.cubehelix_palette(light=.8, n_colors=6)
#plot=sns.lineplot(data=dataprocessing, x='', y='Fre', palette=palette).set(title='Frecuencia vs tiempo')
#sns.lineplot(data=dg1, x='Tiempo (horas)', y='A', palette=palette)
#plt.legend(title='Grupos', loc='upper left', labels=['A', 'B'])
#plt.axis


# Graficas de resultados *Pendiente preguntar*
# Scatterplot recomendado para variables cuantitativas
# Barplot para variables cuantitativas
# Heatmap recomendado para chi cuadrada

# In[329]:


#Tablas graficas
#Seleccionar variable significativa con regex
graf=dataprocessing.groupby('Grupo ').mean().filter(regex='Fre')
grafT=graf.T
grafT.reset_index(inplace=True)
grafT = grafT.rename(columns = {'index':'Frecuencia'})
grafT.loc[grafT['Frecuencia'].str.contains('1'),'Hora'] = '7:00 am'
grafT.loc[grafT['Frecuencia'].str.contains('2'),'Hora'] = '8:00 am'
grafT.loc[grafT['Frecuencia'].str.contains('3'),'Hora'] = '9:00 am'
grafT
#Grafica
#Cambiar nombre de y a 'Frecuencia'
sns.set_style('ticks')
palette = sns.cubehelix_palette(light=.8, n_colors=6)
ax=sns.lineplot(data=grafT, x='Hora', y='A', palette=palette).set(title='Frecuencia vs tiempo')
ax=sns.lineplot(data=grafT, x='Hora', y='B', palette=palette)
plt.legend(title='Grupos', loc='upper left', labels=['A', 'B'])
plt.xlabel('Hora')
plt.ylabel('Frecuencia')
plt.savefig('Figura-1.png')


# In[330]:



#Barplot por grupos 
sns.set_style('ticks')
palette = sns.cubehelix_palette(light=.8, n_colors=6)
ax=sns.barplot(data=grafT, x='Hora', y='A', palette=palette).set(title='Grupo A')
plt.xlabel('Hora')
plt.ylabel('Frecuencia')

plt.savefig('Figura-2.png')


# In[331]:


#Grupo B
sns.set_style('ticks')
palette = sns.cubehelix_palette(light=.8, n_colors=6)
ax=sns.barplot(data=grafT, x='Hora', y='B', palette=palette).set(title='Grupo B')
plt.xlabel('Hora')
plt.ylabel('Frecuencia')

plt.savefig('Figura-3.png')


# In[ ]:





# In[ ]:




