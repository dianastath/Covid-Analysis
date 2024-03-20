# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:10:21 2023

@author: Konst
"""

import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from datetime import datetime

import plotly.express as px
import pandas as pd


def load_data(data_path,data_name):
    csv_path = os.path.join(data_path, data_name)
    return pd.read_csv(csv_path)#,delimiter=',')


def processing_file(data_name,name):
    data_name = data_name.drop(['Province/State'], axis=1)       
    # KATARGHSH KOLONAS Province/State
    data_name.insert(0, 'Assos', 1)                                   
    # Prostetoyme mia nea kolona me 1
    data_name = data_name.groupby(['Country/Region'],as_index=False).sum() # Atrizoyme tis kolones se mia gia kath xora
    data_name['Lat']=data_name['Lat']/data_name['Assos']     
    # Mesos oros Lat
    data_name['Long']=data_name['Long']/data_name['Assos']   
    # Mesos oros Long
    data_name = data_name.drop('Assos', axis=1)       
    
    
    print(data_name.head())
    
    print('TO ARXEIO' ,name,'EXEI ',data_name.shape[0],'GRAMMES KAI ',data_name.shape[1],'STYLES')
    print('TO ARXEIO',name,' EXEI ', data_name.size,'STOIXEIA')

    print('------------------------------------------------------------')
    
    csv_path = os.path.join(ORIGINAL_DATA_PATH,name)
    data_name.to_csv(csv_path)
    
    
ORIGINAL_DATA_PATH = os.path.join('Covid_19')

Data_confirmed = load_data(ORIGINAL_DATA_PATH,'time_series_covid_19_confirmed.csv')
Data_deaths= load_data(ORIGINAL_DATA_PATH,'time_series_covid_19_deaths.csv')
Data_recovered=load_data(ORIGINAL_DATA_PATH,'time_series_covid_19_recovered.csv')



#print('PLHROFORIES GIA CONFIRMED')

#print(Data_confirmed.head())


#Data_confirmed.info()

#print(Data_confirmed.shape)
print('TO  ARXIKO MOY ARXEIO  EXEI ',Data_confirmed.shape[0],'GRAMMES KAI ',Data_confirmed.shape[1],'STYLES')
print('TO ARXIKO MOY ARXEIO EXEI ', Data_confirmed.size,'STOIXEIA')


list_with_countries=Data_confirmed['Country/Region'].unique()

LISTA = pd.DataFrame(list_with_countries)

csv_path = os.path.join(ORIGINAL_DATA_PATH,"Lista_me xores.csv")
LISTA.to_csv(csv_path)
##########################################

#######erotima 2 kai 3#####

list_with_column_names=list(Data_confirmed.columns[4::])
print('H LISTA ME THN PROTH GRAMMH',list_with_column_names) 

date_str1= list_with_column_names[0]
date_str2= list_with_column_names[-1]
a=datetime.strptime(date_str1,'%m/%d/%y')  #.date 
b=datetime.strptime(date_str2,'%m/%d/%y')  #.date
print('a=',a)
print('b=',b)
delta = b - a
print(f'Difference is {delta.days} days')

print('TO XRONIKO DIASTHMA EINAI ',delta)  
    
print('H LISTA ME TIS XORES EINAI ',list_with_countries)
print ('LEN',len(list_with_countries))
#Data_confirmed.groupby('Country/Region',as_index=False)  #.agg({'Country/Region' : 'first', 'Name' : ' '.join})
#confirmed_group.reset_index().to_csv('week_grouped.csv')

print('------------------------------------------------------------')

#call function processing_file

Data_confirmed=processing_file(Data_confirmed,'New_time_series_covid_19_confirmed.csv')
Data_deaths=processing_file(Data_deaths,'New_time_series_covid_19_deaths.csv')
Data_recovered=processing_file(Data_recovered,'New_time_series_covid_19_recovered.csv')



# Import data from USGS
Data_confirmed = pd.read_csv('time_series_covid_19_confirmed.csv')


# Drop rows with missing or inva[id values in the 'mag' column
Data_confirmed = Data_confirmed.dropna(subset=['Data_confirmed.columns[4]'])
Data_confirmed = Data_confirmed[Data_confirmed.Data_confirmed.columns[4] >= 0]


# Create scatter map
fig = px.scatter_geo(Data_confirmed, lat='latitude', lon='longitude', color='Data_confirmed.columns[4]',
                     #hover_name='count', #size='mag',
                     title='Earthquakes Around the World')
fig.show()