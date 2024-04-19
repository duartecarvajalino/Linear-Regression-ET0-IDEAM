# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:51:32 2023

@author: jmduarte
"""

import os
import numpy
import pandas
import math
from datetime import datetime, timedelta

#presion de saturacion de vapor (Capitulo 3, pag. 36, Eq. 11)
def e0(T):  
 return 0.6108 * numpy.exp((17.27 * T) / (237.3 + T))

def Evotranspiration(dayOfyear, lat, H, Tmax, Tmin, Tav, Rs, WV, HR):
    phi = lat / 180 * math.pi  
    const_dia = numpy.zeros(dayOfyear.shape[0])
    const_dia = 2 * math.pi * dayOfyear / 365
    ###############################################
    #PARAMETROS DE LA ECUACION DE EVOTRANSPIRACION#
    ###############################################
    #Presion de vapor de saturacion (Capitulo 3, pg. 36, Eq. 12)
    es = 0.5 * (e0(Tmax) + e0(Tmin))
    #Presion real de vapor (Capitulo 3, pg. 39, Eq. 19)
    ea = (HR / 100) * e0(Tav)
    #Radiacion neta solar de onda corta (Capitulo 3, pg. 51, Eq. 38)
    Rns = 0.77 * Rs
    #Calculo de la radiacion extraterrestre, periodos diarios (Capitulo 3, pg. 45, Eq. 21)
    Gsc = 0.082 # constante solar
    #Distancia relativa inversa Tierra-sol (Capitulo 3, pg. 46, Eq. 23)
    dr = 1 + 0.033 * numpy.cos(const_dia)
    #Declinacion solar (capitulo 3, pg. 46, Eq. 24)
    dsolar = 0.409 * numpy.sin(const_dia - 1.39)
    #Angulo de radiacion a la puesta de sol (Capitulo 3, pg. 46, Eq. 25)
    ws = numpy.arccos(-numpy.tan(phi) * numpy.tan(dsolar))
    #Radiacion extraterrestre (Capitulo 3, pg. 45, Eq. 21)
    Ra = ws * numpy.sin(phi) * numpy.sin(dsolar) + numpy.cos(phi) * numpy.cos(dsolar) * numpy.sin(ws)
    Ra = 24 * (60 / math.pi) * Gsc * dr * Ra
    #Radiacion solar en un dia despejado (capitulo 3, pg. 51, Eq. 37)
    Rso = Ra * (0.75 + 2e-5 * H) 
    #Radiacion neta de onda larga (Capitulo 3, pg. 52, Eq. 39)
    Rnl = 4.903e-9 * 0.5 * (numpy.power(Tmax+273.16,4) + numpy.power(Tmin+273.16,4))
    Rnl = Rnl * (0.34 - 0.14 * numpy.sqrt(ea))
    Rnl = Rnl * (1.35 * (Rs / Rso) -0.35)
    #Radiacion Neta (Capitulo 3, pag. 53, Eq. 40)
    Rn = Rns - Rnl
    #Pendiente de la curva de presion de vapor (Capitulo 3, pg. 36, Eq. 13)
    delta = 4098 * e0(Tav) / (Tav + 237.3)
    delta = delta / (Tav + 237.3)
    #Presion atmosferica (Anexo 3, Eq. 3-2)
    const_exp = 5.25703565
    T_K0 = Tav + 273.16  
    P = 101.3 * numpy.power(1 - 0.0065*(H / T_K0), const_exp)
    #Constante psicrometrica (Anexo 3, Eq. 3-10)
    Lambda = 2.501 - 2.361e-3 * Tav
    Gamma = 0.00163 * P / Lambda
    #Densidad de flujo del suelo (Capitulo 3, pg. 54, Eq. 42)
    G = 0.0
    ##############################################
    #EVOTRANSPIRACION (Capitulo 4, pg. 65, Eq. 6)#
    ##############################################
    num = Gamma * (900 / (Tav + 273.16)) * WV * (es - ea)
    num = num + 0.408 * delta * (Rn - G)
    den = delta + Gamma*(1 + 0.34 * WV)
    ET0 = num / den
    return ET0

def day_to_date(dayOfyear, year):
    date_format = '%Y-%m-%d'
    # create a datetime object for January 1st of the given year
    start_date = datetime(year, 1, 1)
    # add the number of days to the start date
    result_date = start_date + timedelta(days=dayOfyear-1)
    # format the date string using the specified format
    return result_date.strftime(date_format)

def createEmptydataframe(Stations,listOfyears):
    n_years = len(listOfyears)
    N = Stations.shape[0]
    ET0 = pandas.DataFrame(data = -999*numpy.ones((n_years*N,17),dtype=int), 
       columns=['Station','Latitude','Longitude','Altitude','Year',
                'January','February','March','April','May',
                'June','July','August','September','October',
                'November','December'])
    pos = 0
    for i in range(N):
        for j in range(n_years):
            ET0.loc[pos,'Station'] = Stations['Station'][i]
            ET0.loc[pos,'Latitude'] = Stations['Latitude'][i]
            ET0.loc[pos,'Longitude'] = Stations['Longitude'][i]
            ET0.loc[pos,'Altitude'] = Stations['Altitude'][i]
            ET0.loc[pos,'Year'] = listOfyears[j]
            pos += 1

    return ET0

root = 'D:/OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA/Escritorio/Corpoica/ET0 Gustavo/'
# CNE = pandas.read_excel(root + 'CNE_IDEAM.xls',
#                         index_col=0, 
#                         dtype={'CODIGO': str, 'latitud': float,
#                                'longitud': float, 'altitud': int})

# newStations = pandas.read_excel(root + 'Caracteristicas_estaciones_total.xlsx')
# newStations['COD'] = newStations['COD'].str.replace('@','')                                

# Stations = newStations[['COD','Latitude','Longitude','Altitude','Thornthwaite_class']]
# Stations.rename(columns = {'COD':'Station'}, inplace=True)
# Stations.to_excel(root + 'Stations.xlsx')

Stations = pandas.read_excel(root + 'Stations.xlsx', index_col=0, 
                             dtype={'Station':str, 'latitude': float,
                                    'longitude':float,'Altitude':float})
output = root + 'Excel/'

# latitude = Stations['Latitude'].values
# longitude = Stations['Longitude'].values
# base_url = r"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,T2M,T2M_MAX,T2M_MIN,RH2M,WS2M&community=RE&longitude={longitude}&latitude={latitude}&start=19810101&end=20201231&format=CSV"
# N = len(latitude)
 
# for i in range(N):
#     api_request_url = base_url.format(longitude=longitude[i], latitude=latitude[i])
#     filename = 'Station ' + str(i) + '.xlsx'
#     df = pandas.read_csv(api_request_url,header=14)
#     df.to_excel(output + filename)

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']

listOfyears = numpy.arange(1981,2021)
n_years = len(listOfyears)
N = Stations.shape[0]

# NASA POWER
NASA = createEmptydataframe(Stations,listOfyears)
for n in range(N):
    file = 'Station ' + str(n) + '.xlsx'
    print(str(n) + ': ' + file)
    table = pandas.read_excel(output + file,index_col=0)
    station = Stations.loc[n,'Station']
    lat = Stations.loc[n,'Latitude']
    long = Stations.loc[n,'Longitude']
    H = Stations.loc[n,'Altitude']
    table[table < -10] =  numpy.nan
    table.dropna(inplace=True)
    Year = table['YEAR'].values
    dayOfyear = table['DY'].values
    Month = table['MO'].values
    Tmax = table['T2M_MAX'].values
    Tmin = table['T2M_MIN'].values
    Tav = table['T2M'].values
    Rs = table['ALLSKY_SFC_SW_DWN'].values
    WV = table['WS2M'].values
    HR = table['RH2M'].values
    et0 = Evotranspiration(dayOfyear, lat, H, Tmax, Tmin, Tav, Rs, WV, HR)
    et0_accum = numpy.zeros((n_years,12))
    for i in range(len(Year)):
        year = Year[i]
        month = Month[i]
        y = numpy.where(listOfyears == year)[0][0]
        et0_accum[y, month-1] += et0[i]
    for i,year in enumerate(listOfyears):
        if year < 1984: continue
        for m,month in enumerate(months):
            boolean = (NASA['Station'] == station) & (NASA['Year'] == year)
            NASA.loc[boolean,month] = et0_accum[i,m]
    
NASA.to_excel(root + 'ET0_NasaPower.xlsx')

# TERRA CLIMATE
TERRA = createEmptydataframe(Stations,listOfyears)
TERRA_ = pandas.read_csv(root + 'ET_TERRACLIMATE_todas.csv')  
TERRA_['Codigo'] = TERRA_['Codigo'].str.replace('@','')
columns = TERRA_.columns
for i in range(TERRA_.shape[0]):
    station = TERRA_['Codigo'][i]
    print(str(i) + ': ' + station)
    row = Stations[Stations['Station'] == station]
    row.reset_index(drop=True, inplace=True)
    lat = row.loc[0,'Latitude']
    long = row.loc[0,'Longitude']
    H = row.loc[0,'Altitude']
    for j,year in enumerate(listOfyears):
        for m,month in enumerate(months):
            pos = j*12 + m
            boolean = (TERRA['Station'] == station) & (TERRA['Year'] == year)
            TERRA.loc[boolean,'Station'] = station
            TERRA.loc[boolean,'Latitude'] = lat
            TERRA.loc[boolean,'Longitude'] = long
            TERRA.loc[boolean,'Altitude'] = H
            et0 = TERRA_[columns[3+pos]].iloc[i]
            if numpy.isnan(et0): continue 
            TERRA.loc[boolean,month] = et0

TERRA.to_excel(root + 'ET0_TerraClimate.xlsx')

# EUMETSAT
EUMETSAT = createEmptydataframe(Stations,listOfyears)
directory = root + 'ETo_EUMETSAT/'
L = os.listdir(directory)
for k,l in enumerate(L):
    print(str(k) + ': ' + l)
    table = pandas.read_excel(directory + l)
    table['COD'] = table['COD'].str.replace('@','')
    columns = table.columns
    file_split = os.path.splitext(l)
    tokenize = file_split[0].split('_')
    year = int(tokenize[1])
    for i in range(table.shape[0]):
        station = table.loc[i,'COD']
        row = Stations[Stations['Station'] == station]
        row.reset_index(drop=True, inplace=True)
        lat = row.loc[0,'Latitude']
        long = row.loc[0,'Longitude']
        H = row.loc[0,'Altitude']
        for m,month in enumerate(months): 
            boolean = (EUMETSAT['Station'] == station) & (EUMETSAT['Year'] == year)
            if boolean.sum() == 0: break
            EUMETSAT.loc[boolean,'Station'] = station
            EUMETSAT.loc[boolean,'Latitude'] = lat
            EUMETSAT.loc[boolean,'Longitude'] = long
            EUMETSAT.loc[boolean,'Altitude'] = H
            et0 = table[columns[m+1]].iloc[i]
            EUMETSAT.loc[boolean,month] = et0

EUMETSAT.to_excel(root + 'ET0_EUMETSAT.xlsx')

# IDEAM
ET0_ = pandas.read_excel(root + 'ETo_completo_IDEAM.xlsx')  
ET0_['COD'] = ET0_['COD'].str.replace('@','')
date = ET0_['DATE']
year = pandas.DatetimeIndex(date).year
month = pandas.DatetimeIndex(date).month
ET0 = createEmptydataframe(Stations,listOfyears)
    
for i in range(ET0_.shape[0]):
    m = months[month[i]-1]
    y = year[i]
    station = ET0_.loc[i,'COD']
    boolean = (ET0['Station'] == station) & (ET0['Year'] == y) 
    et0 = ET0_.loc[i,'ETo-Month']
    ET0.loc[boolean,m] = et0

ET0.to_excel(root + 'ET0_IDEAM.xlsx')

root = 'D:/OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA/Escritorio/Corpoica/ET0 Gustavo/'
ET0 = pandas.read_excel(root + 'ET0_IDEAM.xlsx')
NASA = pandas.read_excel(root + 'ET0_NasaPower.xlsx')
TERRA = pandas.read_excel(root + 'ET0_TerraClimate.xlsx')
EUMETSAT = pandas.read_excel(root + 'ET0_EUMETSAT.xlsx')

with pandas.ExcelWriter(root + 'ET0_IDEAM_NasaPower_TerraClimate_EUMETSAT.xlsx') as writer:
    ET0.to_excel(writer, sheet_name='IDEAM',index=False)
    NASA.to_excel(writer, sheet_name='NasaPower',index=False)
    TERRA.to_excel(writer, sheet_name='TerraClimate',index=False)
    EUMETSAT.to_excel(writer, sheet_name='EUMETSAT',index=False)
