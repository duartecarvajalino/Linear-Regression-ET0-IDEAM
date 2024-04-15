# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:51:12 2023

@author: jmduarte
"""
#import os
import pandas
import numpy
from numpy.polynomial import Polynomial
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#import math
#from statsmodels import robust
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import LeaveOneOut, KFold
import matplotlib._color_data as mcd
import pickle

# from sklearn.linear_model import LinearRegression
# from sklearn import svm
# from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_percentage_error
# from sklearn.metrics import make_scorer
# from sklearn.model_selection import KFold

def RError(Ytrue,Ypred):
    norm_diff = numpy.linalg.norm(Ytrue-Ypred)
    norm_Ytrue = numpy.linalg.norm(Ytrue) 
    if norm_Ytrue>0: return norm_diff/norm_Ytrue*100
    else: return norm_diff

def NRMSError(Ytrue,Ypred):
   Range = Ytrue.max() - Ytrue.min()
   if Range > 0: 
       return mean_squared_error(Ytrue, Ypred, squared=False)/Range
   else: return mean_squared_error(Ytrue, Ypred, squared=False)

# def R2score_adj(Ytrue,Ypred,X):
#     pinv = numpy.linalg.inv(numpy.matmul(X.T,X))
#     H = numpy.diag(numpy.matmul(numpy.matmul(X,pinv),X.T))
#     num = numpy.median(abs(Ytrue - Ypred)) / numpy.median(1.0 - H)
#     den = numpy.median(abs(Ytrue - numpy.mean(Ytrue)))
#     Q  = numpy.square(num/den)
#     if Q < 1.0: return 1.0 - Q
#     else: return 0

def R2score_adj(Ytrue,Ypred):
    num = numpy.median(abs(Ytrue - Ypred))
    den = numpy.median(abs(Ytrue - numpy.mean(Ytrue)))
    Q  = numpy.square(num/den)
    if Q < 1.0: return 1.0 - Q
    else: return 0

def WAI(Ytrue,Ypred):
   Ytrue_mean = Ytrue.mean()
   num = numpy.square(Ypred-Ytrue).sum()
   num = numpy.linalg.norm(Ytrue-Ypred,ord=1)
   den = numpy.linalg.norm(Ytrue-Ytrue_mean,ord=1)
   den += numpy.linalg.norm(Ypred-Ytrue_mean,ord=1)
   return 1.0 - num/den

def bestRLMnorm(Y,X):
    R2 = numpy.zeros(5)
    rmse = numpy.zeros(5)
    nrmse = numpy.zeros(5)
    wai = numpy.zeros(5)
    RE = numpy.zeros(5)
    MAE = numpy.zeros(5)
    model = sm.RLM(Y,X,M=sm.robust.norms.HuberT()).fit(maxiter=1000)
    YPred = model.predict(X)
    Y2 = Y*numpy.sqrt(model.weights)
    YPred2 = YPred*numpy.sqrt(model.weights)
    rmse[0] = mean_squared_error(Y2,YPred2)
    nrmse[0] = NRMSError(Y2,YPred2)
    wai[0] = WAI(Y2,YPred2)
    RE[0] = RError(Y2,YPred2)
    MAE[0] = mean_absolute_error(Y2,YPred2)
    R2[0] = r2_score(Y2,YPred2)
    model = sm.RLM(Y,X,M=sm.robust.norms.RamsayE()).fit(maxiter=1000)
    YPred = model.predict(X)
    Y2 = Y*numpy.sqrt(model.weights)
    YPred2 = YPred*numpy.sqrt(model.weights)
    rmse[1] = mean_squared_error(Y2,YPred2)
    nrmse[1] = NRMSError(Y2,YPred2)
    wai[1] = WAI(Y2,YPred2)
    RE[1] = RError(Y2,YPred2)
    MAE[1] = mean_absolute_error(Y2,YPred2)
    R2[1] = r2_score(Y2,YPred2)
    model = sm.RLM(Y,X,M=sm.robust.norms.TukeyBiweight()).fit(maxiter=1000)
    YPred = model.predict(X)
    Y2 = Y*numpy.sqrt(model.weights)
    YPred2 = YPred*numpy.sqrt(model.weights)
    rmse[2] = mean_squared_error(Y2,YPred2)
    nrmse[2] = NRMSError(Y2,YPred2)
    wai[2] = WAI(Y2,YPred2)
    RE[2] = RError(Y2,YPred2)
    MAE[2] = mean_absolute_error(Y2,YPred2)
    R2[2] = r2_score(Y2,YPred2)
    model = sm.RLM(Y,X,M=sm.robust.norms.AndrewWave()).fit(maxiter=1000)
    YPred = model.predict(X)
    Y2 = Y*numpy.sqrt(model.weights)
    YPred2 = YPred*numpy.sqrt(model.weights)
    rmse[3] = mean_squared_error(Y2,YPred2)
    nrmse[3] = NRMSError(Y2,YPred2)
    wai[3] = WAI(Y2,YPred2)
    RE[3] = RError(Y2,YPred2)
    MAE[3] = mean_absolute_error(Y2,YPred2)
    R2[3] = r2_score(Y2,YPred2)
    model = sm.RLM(Y,X,M=sm.robust.norms.Hampel()).fit(maxiter=1000)
    YPred = model.predict(X)
    Y2 = Y*numpy.sqrt(model.weights)
    YPred2 = YPred*numpy.sqrt(model.weights)
    rmse[4] = mean_squared_error(Y2,YPred2)
    nrmse[4] = NRMSError(Y2,YPred2)
    wai[4] = WAI(Y2,YPred2)
    RE[4] = RError(Y2,YPred2)
    MAE[4] = mean_absolute_error(Y2,YPred2)
    R2[4] = r2_score(Y2,YPred2)
    best = numpy.argmax(R2)
    if best==0: return (rmse,nrmse,wai,RE,MAE,R2),'Huber’s t',sm.robust.norms.HuberT()
    elif best==1: return(rmse,nrmse,wai,RE,MAE,R2),'Ramsay’s Ea',sm.robust.norms.RamsayE()
    elif best==2: return (rmse,nrmse,wai,RE,MAE,R2),'Tukey’s Biweight',sm.robust.norms.TukeyBiweight()
    elif best==3: return (rmse,nrmse,wai,RE,MAE,R2),'Andrew’s Wave',sm.robust.norms.AndrewWave()
    elif best==4: return (rmse,nrmse,wai,RE,MAE,R2),'Hampel',sm.robust.norms.Hampel()

def Crossvalidation(et0,et0_x,n,W):
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    rmse = numpy.zeros(n)
    nrmse = numpy.zeros(n)
    r2score = numpy.zeros(n)
    wai = numpy.zeros(n)
    RE = numpy.zeros(n)
    MAE = numpy.zeros(n)
    for i, (train_index, test_index) in enumerate(kf.split(et0)):
        et0_train, et0_test = et0[train_index], et0[test_index]
        et0_x_train, et0_x_test = et0_x[train_index], et0_x[test_index]
        W_test = W[test_index]
        kind,norm = bestRLMnorm(et0_train,sm.add_constant(et0_x_train))
        model_train = sm.RLM(et0_train,sm.add_constant(et0_x_train),M=norm).fit(maxiter=1000)
        et0_hat = model_train.predict(sm.add_constant(et0_x_test))
        rmse[i] = mean_squared_error(et0_test*numpy.sqrt(W_test),et0_hat*numpy.sqrt(W_test))
        nrmse[i] = NRMSError(et0_test*numpy.sqrt(W_test),et0_hat*numpy.sqrt(W_test))
        r2score[i] = r2_score(et0_test*numpy.sqrt(W_test),et0_hat*numpy.sqrt(W_test))
        wai[i] = WAI(et0_test*numpy.sqrt(W_test),et0_hat*numpy.sqrt(W_test))
        RE[i] = RError(et0_test*numpy.sqrt(W_test),et0_hat*numpy.sqrt(W_test))
        MAE[i] = mean_absolute_error(et0_test*numpy.sqrt(W_test),et0_hat*numpy.sqrt(W_test))
        
    return (rmse.mean(),nrmse.mean(),r2score.mean(),wai.mean(),RE.mean(),MAE.mean())    

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']

root = 'E:/Corpoica/ET0 Gustavo/'

ET0_IDEAM = pandas.read_excel(root + 'ET0_IDEAM_NasaPower_TerraClimate_EUMETSAT.xlsx', 
                          index_col=0, sheet_name = 'IDEAM')
#ET0_NASA = pandas.read_excel(root + 'ET0_NasaPower.xlsx') 
ET0_NASA = pandas.read_excel(root + 'ET0_IDEAM_NasaPower_TerraClimate_EUMETSAT.xlsx', 
                          index_col=0, sheet_name = 'NasaPower')
ET0_TERRA = pandas.read_excel(root + 'ET0_IDEAM_NasaPower_TerraClimate_EUMETSAT.xlsx', 
                          index_col=0, sheet_name = 'TerraClimate')
ET0_EUMETSAT = pandas.read_excel(root + 'ET0_IDEAM_NasaPower_TerraClimate_EUMETSAT.xlsx', 
                          index_col=0, sheet_name = 'EUMETSAT')

ListUsingAllData_Nasa = []
#ListUsingAlldata_Cooks_Nasa = []
ListUsingAllData_Terra = []
#ListUsingAlldata_Cooks_Terra = []

P_NASA = numpy.zeros((10,2))
W_NASA = [[]]*10
P_TERRA = numpy.zeros((10,2))
W_TERRA = [[]]*10

NormErrors = numpy.zeros((12,5))

# z = (et0-et0.mean())/et0.std()
# stats.probplot(z, dist="norm", plot=plt) 
# plt.title("$ET_0$ IDEAM Q-Q plot") 
# plt.savefig(root + 'Figures/IDEAM Q-Q plot.jpg', dpi=600)
# plt.close()

##############
# NASA POWER #
##############
boolean = numpy.sum(ET0_IDEAM[months] > 0, axis=1)  > 0
data_IDEAM = ET0_IDEAM[boolean]
data_IDEAM.reset_index(drop=True, inplace=True)
et0 = data_IDEAM[months].values
et0 = et0.flatten()
boolean1 = et0 > 0
data_nasa = ET0_NASA[boolean]
data_nasa.reset_index(drop=True, inplace=True)
et0_nasa = data_nasa[months].values
et0_nasa = et0_nasa.flatten()
boolean2 = et0_nasa > 0
et0 = et0[boolean1 & boolean2]
et0_nasa = et0_nasa[boolean1 & boolean2]
# z = (et0_nasa-et0_nasa.mean())/et0_nasa.std()
# stats.probplot(z, dist="norm", plot=plt) 
# plt.title("$ET_0$ Nasa Power Q-Q plot") 
# plt.savefig(root + 'Figures/Nasa Power Q-Q plot.jpg', dpi=600)
# plt.close()

model_OLS = sm.OLS(et0,sm.add_constant(et0_nasa)).fit(maxiter=1000)
et0_hat_OLS = model_OLS.predict(sm.add_constant(et0_nasa))
rmse = mean_squared_error(et0,et0_hat_OLS)
nrmse = NRMSError(et0,et0_hat_OLS)
r2score = r2_score(et0,et0_hat_OLS)
wai = WAI(et0,et0_hat_OLS) 
RE = RError(et0,et0_hat_OLS)
MAE = mean_absolute_error(et0,et0_hat_OLS)

plt.scatter(et0_nasa,et0,color='blue',s=10,label='IDEAM')
plt.plot(et0_nasa,et0_hat_OLS,'g',linewidth=2,label='Fitted')
plt.xlabel('$ET_0$ Nasa Power ($mm\\times month^{-1}$)')
plt.ylabel('$ET_0$ IDEAM ($mm\\times month^{-1}$)')
# plt.text(130, 85, 'RMSE: ' + str(round(rmse,4)), fontsize=9,fontweight='bold')
# plt.text(130, 73, 'NRMSE: ' + str(round(nrmse,4)), fontsize=9,fontweight='bold')
# plt.text(130, 57, 'R$^2$: ' + str(round(r2score,4)), fontsize=9,fontweight='bold')
# plt.text(130, 42, 'WAI: ' + str(round(wai,4)), fontsize=9,fontweight='bold')
# plt.text(130, 27, 'RE(%): ' + str(round(RE,4)), fontsize=9,fontweight='bold')
string = 'Evapotranspiration ($ET_0$): OLS Nasa Power vs IDEAM \n'
string = string + '$ET_0^{IDEAM}=' + str(round(model_OLS.params[0],4)) + '+' + str(round(model_OLS.params[1],4)) + '\\times ET_0^{Nasa}$'
plt.title(string,fontsize=12)
plt.legend(loc='upper left', ncol=1, shadow=False, 
            fancybox=True, fontsize=9)
plt.tight_layout()
plt.savefig(root + 'Figures2/Nasa Power vs IDEAM.jpg', dpi=600)
plt.close()

(NormErrors[0,:],NormErrors[1,:],NormErrors[2,:],NormErrors[3,:],
 NormErrors[4,:],NormErrors[5,:]),kind,norm = bestRLMnorm(et0,sm.add_constant(et0_nasa))
print('Nasa Power: ' + kind)
model = sm.RLM(et0,sm.add_constant(et0_nasa),M=norm).fit(maxiter=1000)
P_NASA[0,:] = model.params
W_NASA[0] = model.weights
et0_hat = model.predict(sm.add_constant(et0_nasa))
(rmse,nrmse,r2score,wai,RE,MAE) = Crossvalidation(et0,et0_nasa,10,model.weights)

ListUsingAllData_Nasa.append((rmse,nrmse,r2score,wai,RE,MAE))

plt.scatter(et0_nasa,et0,s=model.weights*10,c=model.weights,
            label='Weighted data',cmap='jet')
plt.plot(et0_nasa,et0_hat,'g',linewidth=2,label='Fitted')
plt.xlabel('$ET_0$ Nasa Power ($mm\\times month^{-1}$)')
plt.ylabel('$ET_0$ IDEAM ($mm\\times month^{-1}$)')
# plt.text(120, 85, 'RMSE: ' + str(round(rmse,4)), fontsize=9,fontweight='bold')
# plt.text(120, 73, 'NRMSE: ' + str(round(nrmse,4)), fontsize=9,fontweight='bold')
# plt.text(120, 57, 'R$^2$: ' + str(round(r2score,4)), fontsize=9,fontweight='bold')
# plt.text(120, 43, 'WAI: ' + str(round(wai,4)), fontsize=9,fontweight='bold')
# plt.text(120, 30, 'RE(%): ' + str(round(RE,4)), fontsize=9,fontweight='bold')
string = 'Evapotranspiration ($ET_0$): Nasa Power vs IDEAM \n'
string = string + '$ET_0^{IDEAM}=' + str(round(model.params[0],4)) + '+' + str(round(model.params[1],4)) + '\\times ET_0^{Nasa}$'
plt.title(string,fontsize=12)
plt.legend(loc='upper left', ncol=1, shadow=False, 
            fancybox=True, fontsize=9)
plt.colorbar()
plt.tight_layout()
plt.savefig(root + 'Figures2/Nasa Power vs IDEAM_v2.jpg', dpi=600)
plt.close()

#################
# TERRA CLIMATE #
#################
boolean = numpy.sum(ET0_IDEAM[months] > 0, axis=1)  > 0
data_IDEAM = ET0_IDEAM[boolean]
data_IDEAM.reset_index(drop=True, inplace=True)
et0 = data_IDEAM[months].values
et0 = et0.flatten()
boolean1 = et0 > 0
data_terra = ET0_TERRA[boolean]
data_terra.reset_index(drop=True, inplace=True)
et0_terra = data_terra[months].values
et0_terra = et0_terra.flatten()
boolean2 = et0_terra > 0
et0 = et0[boolean1 & boolean2]
et0_terra = et0_terra[boolean1 & boolean2]

# z = (et0_terra-et0_terra.mean())/et0_terra.std()
# stats.probplot(z, dist="norm", plot=plt) 
# plt.title("$ET_0$ Terra Climate Q-Q plot") 
# plt.savefig(root + 'Figures/Terra Climate Q-Q plot.jpg', dpi=600)
# plt.close()

model_OLS = sm.OLS(et0,sm.add_constant(et0_terra)).fit(maxiter=1000)
et0_hat_OLS = model_OLS.predict(sm.add_constant(et0_terra))
rmse = mean_squared_error(et0,et0_hat_OLS)
nrmse = NRMSError(et0,et0_hat_OLS)
r2score = r2_score(et0,et0_hat_OLS)
wai = WAI(et0,et0_hat_OLS) 
RE = RError(et0,et0_hat_OLS)
MAE = mean_absolute_error(et0,et0_hat_OLS)

plt.scatter(et0_terra,et0,color='blue',s=10,label='IDEAM')
plt.plot(et0_terra,et0_hat_OLS,'g',linewidth=2,label='Fitted')
plt.xlabel('$ET_0$ Terra Climate ($mm\\times month^{-1}$)')
plt.ylabel('$ET_0$ IDEAM ($mm\\times month^{-1}$)')
# plt.text(175, 85, 'RMSE: ' + str(round(rmse,4)), fontsize=9,fontweight='bold')
# plt.text(175, 73, 'NRMSE: ' + str(round(nrmse,4)), fontsize=9,fontweight='bold')
# plt.text(175, 57, 'R$^2$: ' + str(round(r2score,4)), fontsize=9,fontweight='bold')
# plt.text(175, 45, 'WAI: ' + str(round(wai,4)), fontsize=9,fontweight='bold')
# plt.text(175, 30, 'RE(%): ' + str(round(RE,4)), fontsize=9,fontweight='bold')
string = 'Evapotranspiration ($ET_0$): OLS Terra Climate vs IDEAM \n'
string = string + '$ET_0^{IDEAM}=' + str(round(model_OLS.params[0],4)) + '+' + str(round(model_OLS.params[1],4)) + '\\times ET_0^{Terra}$'
plt.title(string,fontsize=12)
plt.legend(loc='upper left', ncol=1, shadow=False, 
            fancybox=True, fontsize=9)
plt.tight_layout()
plt.savefig(root + 'Figures2/Terra Climate vs IDEAM.jpg', dpi=600)
plt.close()

(NormErrors[6,:],NormErrors[7,:],NormErrors[8,:],NormErrors[9,:],
 NormErrors[10,:],NormErrors[11,:]),kind,norm = bestRLMnorm(et0,sm.add_constant(et0_terra))
print('Terra Climate: ' + kind)
model = sm.RLM(et0,sm.add_constant(et0_terra),M=norm).fit(maxiter=1000)
P_TERRA[0,:] = model.params
W_TERRA[0] = model.weights

et0_hat = model.predict(sm.add_constant(et0_terra))

(rmse,nrmse,r2score,wai,RE,MAE) = Crossvalidation(et0,et0_terra,10,model.weights)

ListUsingAllData_Terra.append((rmse,nrmse,r2score,wai,RE,MAE))

#model = sm.WLS(et0,sm.add_constant(et0_terra),weights=model.weights).fit(maxiter=1000)

plt.scatter(et0_terra,et0,s=model.weights*10,c=model.weights,
            label='Weighted data',cmap='jet')
plt.plot(et0_terra,et0_hat,'g',linewidth=2,label='Fitted')
plt.xlabel('$ET_0$ Terra Climate ($mm\\times month^{-1}$)')
plt.ylabel('$ET_0$ IDEAM ($mm\\times month^{-1})$')
# plt.text(175, 85, 'RMSE: ' + str(round(rmse,4)), fontsize=9,fontweight='bold')
# plt.text(175, 73, 'NRMSE: ' + str(round(nrmse,4)), fontsize=9,fontweight='bold')
# plt.text(175, 57, 'R$^2$: ' + str(round(r2score,4)), fontsize=9,fontweight='bold')
# plt.text(175, 45, 'WAI: ' + str(round(wai,4)), fontsize=9,fontweight='bold')
# plt.text(175, 30, 'RE(%): ' + str(round(RE,4)), fontsize=9,fontweight='bold')
string = 'Evapotranspiration ($ET_0$): IDEAM vs Terra Climate \n'
string = string + '$ET_0^{IDEAM}=' + str(round(model.params[0],4)) + '+' + str(round(model.params[1],4)) + '\\times ET_0 ^{Terra}$'
plt.title(string,fontsize=12)
plt.legend(loc='upper left', ncol=1, shadow=False, 
            fancybox=True, fontsize=9)
plt.colorbar()
plt.tight_layout()
plt.savefig(root + 'Figures2/IDEAM vs Terra climate v2.jpg', dpi=600)
plt.close()

numpy.zeros()

#Save the performance of each model's Norm
Table_Norms = pandas.DataFrame(NormErrors,index=['RMSE','NRMSE','WAI','RE(%)','MAE','R2score']*2,
                columns=['Huber','Ramsay Ea','Tukey Biweight','Andrew Wave','Hampel'])
Table_Norms.to_excel(root + 'Linear Model Norms Performance.xlsx')

############
# EUMETSAT #
############
boolean = numpy.sum(ET0_IDEAM[months] > 0, axis=1)  > 0
data_IDEAM = ET0_IDEAM[boolean]
data_IDEAM.reset_index(drop=True, inplace=True)
data_eumetsat = ET0_EUMETSAT[boolean]
data_eumetsat.reset_index(drop=True, inplace=True)
data_IDEAM = data_IDEAM[data_IDEAM['Year'] > 2004]
data_IDEAM.reset_index(drop=True, inplace=True)
data_eumetsat = data_eumetsat[data_eumetsat['Year'] > 2004]
data_eumetsat.reset_index(drop=True, inplace=True)

et0 = data_IDEAM[months].values
et0 = et0.flatten()
boolean1 = et0 > 0
et0_eumetsat = data_eumetsat[months].values
et0_eumetsat = et0_eumetsat.flatten()
boolean2 = et0_eumetsat > 0
et0 = et0[boolean1 & boolean2]
et0_eumetsat = et0_eumetsat[boolean1 & boolean2]

# z = (et0_eumetsat-et0_eumetsat.mean())/et0_eumetsat.std()
# stats.probplot(z, dist="norm", plot=plt) 
# plt.title("$ET_0$ EUMETSAT Q-Q plot") 
# plt.savefig(root + 'Figures/EUMETSAT Q-Q plot.jpg', dpi=600)
# plt.close()

model_OLS = sm.OLS(et0,sm.add_constant(et0_eumetsat)).fit(maxiter=1000)
et0_hat_OLS = model_OLS.predict(sm.add_constant(et0_eumetsat))
rmse = mean_squared_error(et0,et0_hat_OLS)
nrmse = NRMSError(et0,et0_hat_OLS)
r2score = r2_score(et0,et0_hat_OLS)
wai = WAI(et0,et0_hat_OLS) 
RE = RError(et0,et0_hat_OLS)
MAE = mean_absolute_error(et0,et0_hat_OLS)

plt.scatter(et0_eumetsat,et0,color='blue',s=10,label='IDEAM')
plt.plot(et0_eumetsat,et0_hat_OLS,'g',linewidth=2,label='Fitted')
plt.xlabel('$ET_0$ Eumetsat ($mm\\times month^{-1}$)')
plt.ylabel('$ET_0$ IDEAM ($mm\\times month^{-1}$)')
# plt.text(130, 85, 'RMSE: ' + str(round(rmse,4)), fontsize=9,fontweight='bold')
# plt.text(130, 73, 'NRMSE: ' + str(round(nrmse,4)), fontsize=9,fontweight='bold')
# plt.text(130, 60, 'R$^2$: ' + str(round(r2score,4)), fontsize=9,fontweight='bold')
# plt.text(130, 50, 'WAI: ' + str(round(wai,4)), fontsize=9,fontweight='bold')
# plt.text(130, 40, 'RE(%): ' + str(round(RE,4)), fontsize=9,fontweight='bold')
string = 'Evapotranspiration ($ET_0$): OLS Eumetsat vs IDEAM \n'
string = string + '$ET_0^{IDEAM}=' + str(round(model_OLS.params[0],4)) + '+' + str(round(model_OLS.params[1],4)) + '\\times ET_0^{EUMETSAT}$'
plt.title(string,fontsize=12)
plt.legend(loc='upper left', ncol=1, shadow=False, 
            fancybox=True, fontsize=9)
plt.tight_layout()
plt.savefig(root + 'Figures2/Eumetsat vs IDEAM.jpg', dpi=600)
plt.close()

boolean = numpy.sum(ET0_IDEAM[months] > 0, axis=1)  > 0
data_IDEAM = ET0_IDEAM[boolean]
data_IDEAM.reset_index(drop=True, inplace=True)
et0 = data_IDEAM[months].values
et0 = et0.flatten()
boolean = et0 > 0
et0 = et0[boolean]

L = len(et0)
L_nasa = len(et0_nasa)
if L_nasa < L:
    et0_nasa = numpy.append(et0_nasa,numpy.repeat(
        numpy.nan,L-L_nasa))

L_terra = len(et0_terra)
if L_terra < L:
    et0_terra = numpy.append(et0_terra,numpy.repeat(
        numpy.nan,L-L_terra))

df = pandas.DataFrame(data=numpy.vstack((et0,et0_nasa,
                    et0_terra)).T,
                      index=numpy.arange(0,L,dtype=int),
                      columns=['IDEAM','Nasa Power','Terra Climate'])  
sns.kdeplot(data=df,bw_adjust=2.0)
plt.title('Kernel Density Estimator $ET_0$')
plt.xlabel('$ET_0$ ($mm\\times month^{-1}$)',fontsize=12)
plt.savefig(root + 'Figures2/KDE.jpg', dpi=600)
plt.close()

Stations = pandas.read_excel(root + 'Stations.xlsx', index_col=0, 
                             dtype={'Station':str, 'latitude': float,
                                    'longitude':float,'Altitude':float})

######################################################################
#                          USING REGIONS                             #
######################################################################
Stations = pandas.read_excel(root + 'Stations.xlsx', index_col=0)
boolean = numpy.sum(ET0_IDEAM[months] > 0, axis=1)  > 0
data_IDEAM = ET0_IDEAM[boolean]
data_IDEAM.reset_index(drop=True, inplace=True)
data_nasa = ET0_NASA[boolean]
data_nasa.reset_index(drop=True, inplace=True)
data_terra = ET0_TERRA[boolean]
data_terra.reset_index(drop=True, inplace=True)

stations = numpy.unique(data_IDEAM['Station'].values)
#Regions = pandas.read_excel(root + 'EST_ETo_Caldas_Cluster.xlsx')
Regions = Stations[Stations['Station'].isin(stations)]
classes = Regions['Thornthwaite_class'].values.max()
#classes = Regions['Region'].values.max()
SR = []
for i in range(1,classes+1):
    SR.append(Regions[Regions['Thornthwaite_class'] == i]['Station'].values)

et0_R = []
for i in range(classes):
    ET0_IDEAM_R = data_IDEAM[data_IDEAM['Station'].isin(SR[i])]
    et0_R.append(ET0_IDEAM_R[months].values)

et0_R_nasa = []
for i in range(classes):
    ET0_NASA_R = data_nasa[data_nasa['Station'].isin(SR[i])]
    et0_R_nasa.append(ET0_NASA_R[months].values)

et0_R_terra = []
for i in range(classes):
    ET0_TERRA_R = data_terra[data_terra['Station'].isin(SR[i])]
    et0_R_terra.append(ET0_TERRA_R[months].values)
    
##############
# NASA POWER #
##############
ListUsingRegions_Nasa = [[]]*classes
for i in range(classes):
    et0 = et0_R[i].flatten()
    boolean1 = et0 > 0
    et0_nasa = et0_R_nasa[i].flatten()
    boolean2 = et0_nasa > 0
    et0 = et0[boolean1 & boolean2]
    et0_nasa = et0_nasa[boolean1 & boolean2]
    #model = sm.OLS(et0_,sm.add_constant(et0_nasa_)).fit()
    _,kind,norm = bestRLMnorm(et0,sm.add_constant(et0_nasa))
    print('Nasa Power: ' + kind)
    model = sm.RLM(et0,sm.add_constant(et0_nasa),M=norm).fit(maxiter=1000)
    P_NASA[i+1,:] = model.params
    W_NASA[i+1] = model.weights
    et0_hat = model.predict(sm.add_constant(et0_nasa))
    (rmse,nrmse,r2score,wai,RE,MAE) = Crossvalidation(et0,et0_nasa,10,model.weights)
    ListUsingRegions_Nasa[i] = (rmse,nrmse,r2score,wai,RE,MAE)
    x = et0_nasa.min() + (et0_nasa.max()-et0_nasa.min())*0.7
    plt.scatter(et0_nasa,et0,s=model.weights*10,c=model.weights,
            label='Weighted data',cmap='jet')
    plt.plot(et0_nasa,et0_hat,'g',linewidth=2,label='Fitted')
    plt.xlabel('$ET_0$ Nasa Power ($mm\\times month^{-1})$')
    plt.ylabel('$ET_0$ IDEAM ($mm\\times month^{-1})$')
    D = et0.max() - et0.min()
    y = numpy.linspace(et0.min() + D*0.01,et0.min() + D*0.25,5,endpoint=True)
    # plt.text(x, y[0], 'RMSE: ' + str(round(rmse,4)), fontsize=9,fontweight='bold')
    # plt.text(x, y[1], 'NRMSE: ' + str(round(nrmse,4)), fontsize=9,fontweight='bold')
    # plt.text(x, y[2], 'R$^2$: ' + str(round(r2score,4)), fontsize=9,fontweight='bold')
    # plt.text(x, y[3], 'WAI: ' + str(round(wai,4)), fontsize=9,fontweight='bold')
    # plt.text(x, y[4], 'RE(%): ' + str(round(RE,4)), fontsize=9,fontweight='bold')
    string = 'Evapotranspiration ($ET_0$): IDEAM vs Nasa Power, Region ' + str(i+1) + '\n'
    if model.params[1] > 0:
        string = string + '$ET_0^{IDEAM}=' + str(round(model.params[0],4)) + '+' + str(round(model.params[1],4)) + '\\times ET_0 ^{Nasa}$'
    else:
        string = string + '$ET_0^{IDEAM}=' + str(round(model.params[0],4)) + str(round(model.params[1],4)) + '\\times ET_0 ^{Nasa}$'
    plt.title(string,fontsize=12)
    plt.legend(loc='upper left', ncol=1, shadow=False, 
            fancybox=True, fontsize=9)
    plt.colorbar()
    plt.savefig(root + 'Figures2/Regions/IDEAM vs Nasa Power, Region' +str(i+1) +'.jpg',dpi=600)
    plt.close()

# overlap = {name for name in mcd.CSS4_COLORS
#            if "xkcd:" + name in mcd.XKCD_COLORS}
# listOfColors = list(overlap)
# listOfColors.sort()
# colors = [0,3,5,6,7,11,14,17,38] #,22,26,29,30,33,35,36,37,38,44,47]

# for i in range(classes):
#     et0 = et0_R[i].flatten()
#     boolean1 = et0 > 0
#     et0_nasa = et0_R_nasa[i].flatten()
#     boolean2 = et0_nasa > 0
#     et0 = et0[boolean1 & boolean2]
#     et0_nasa = et0_nasa[boolean1 & boolean2]
#     plt.plot(et0,et0_nasa,'.',color='xkcd:'+listOfColors[colors[i]],label=str(i+1))

# plt.title('Nasa Power vs IDEAM')
# plt.xlabel('$ET_0$ Nasa Power ($mm\\times month^{-1})$)')
# plt.ylabel('$ET_0$ IDEAM ($mm\\times month^{-1})$')
# plt.legend(loc='upper right', ncol=1, shadow=False, 
#             fancybox=True, fontsize=8)    
# plt.savefig(root + 'Figures2/IDEAM vs Nasa Power Region by Region.jpg',dpi=600)

#################
# TERRA CLIMATE #
#################
ListUsingRegions_Terra = [[]]*classes
for i in range(classes):
    et0 = et0_R[i].flatten()
    boolean1 = et0 > 0
    et0_terra = et0_R_terra[i].flatten()
    boolean2 = et0_terra > 0
    et0 = et0[boolean1 & boolean2]
    et0_terra = et0_terra[boolean1 & boolean2]
    #model = sm.OLS(et0_,sm.add_constant(et0_terra_)).fit()
    _,kind,norm = bestRLMnorm(et0,sm.add_constant(et0_terra))
    print('Terra Climate: ' + kind)
    model = sm.RLM(et0,sm.add_constant(et0_terra),M=norm).fit(maxiter=1000)
    P_TERRA[i+1,:] = model.params
    W_TERRA[i+1] = model.weights
    et0_hat = model.predict(sm.add_constant(et0_terra))
    (rmse,nrmse,r2score,wai,RE,MAE) = Crossvalidation(et0,et0_terra,10,model.weights)
    ListUsingRegions_Terra[i] = (rmse,nrmse,r2score,wai,RE,MAE)
    x = et0_terra.min() + (et0_terra.max()-et0_terra.min())*0.7
    y = et0.min() + (et0.max()-et0.min())*0.25
    plt.scatter(et0_terra,et0,s=model.weights*10,c=model.weights,
            label='Weighted data',cmap='jet')
    plt.plot(et0_terra,et0_hat,'g',linewidth=2,label='Fitted')
    plt.xlabel('$ET_0$ Terra Climate ($mm\\times month^{-1})$')
    plt.ylabel('$ET_0$ IDEAM ($mm\\times month^{-1})$')
    D = et0.max() - et0.min()
    y = numpy.linspace(et0.min() + D*0.01,et0.min() + D*0.25,5,endpoint=True)
    # plt.text(x, y[0], 'RMSE: ' + str(round(rmse,4)), fontsize=9,fontweight='bold')
    # plt.text(x, y[1], 'NRMSE: ' + str(round(nrmse,4)), fontsize=9,fontweight='bold')
    # plt.text(x, y[2], 'R$^2$: ' + str(round(r2score,4)), fontsize=9,fontweight='bold')
    # plt.text(x, y[3], 'WAI: ' + str(round(wai,4)), fontsize=9,fontweight='bold')
    # plt.text(x, y[4], 'RE(%): ' + str(round(RE,4)), fontsize=9,fontweight='bold')
    string = 'Evapotranspiration ($ET_0$): IDEAM vs Terra Climate, Region ' + str(i+1) + '\n'
    string = string + '$ET_0^{IDEAM}=' + str(round(model.params[0],4)) + '+' + str(round(model.params[1],4)) + '\\times ET_0 ^{Terra}$'
    plt.title(string,fontsize=12)
    plt.legend(loc='upper left', ncol=1, shadow=False, 
            fancybox=True, fontsize=9)
    plt.colorbar()
    plt.savefig(root + 'Figures2/Regions/IDEAM vs Terra Climate, Region' +str(i+1) +'.jpg',dpi=600)
    plt.close()

# for i in range(classes):
#     et0 = et0_R[i].flatten()
#     boolean1 = et0 > 0
#     et0_terra = et0_R_terra[i].flatten()
#     boolean2 = et0_terra > 0
#     et0 = et0[boolean1 & boolean2]
#     et0_terra = et0_terra[boolean1 & boolean2]
#     plt.plot(et0,et0_terra,'.',color='xkcd:'+listOfColors[colors[i]],label=str(i+1))

# plt.title('Terra Climate vs IDEAM')
# plt.xlabel('$ET_0$ Terra Climate ($mm\\times month^{-1})$)')
# plt.ylabel('$ET_0$ IDEAM ($mm\\times month^{-1})$')
# plt.legend(loc='upper right', ncol=1, shadow=False, 
#             fancybox=True, fontsize=8)    
# plt.savefig(root + 'Figures2/IDEAM vs Terra Climate Region by Region.jpg',dpi=600)

numpy.save(root + 'P_NASA_1981_2020.npy', P_NASA)
numpy.save(root + 'P_TERRA_1981_2020.npy', P_TERRA)

with open(root + 'W_NASA_1981_20220.txt', "wb") as f:
    pickle.dump(W_NASA, f)

with open(root + 'W_TERRA_1981_20220.txt', "wb") as f:
    pickle.dump(W_TERRA, f)

L = ListUsingAllData_Nasa + ListUsingAllData_Terra 

Table_All = pandas.DataFrame(L,index=['Nasa power','Terra Climate'],
                columns=['RMSE','NRMSE','R2score','WAI','RE(%)','MAE'])

L = ListUsingRegions_Nasa + ListUsingRegions_Terra 
    
labels = ['Nasa Power, Region: ','Terra Climate, Region: ']
indices = []
for i in range(len(labels)):
    for r in range(classes):
        indices += [labels[i] + str(r+1)]
   
Table_Regions = pandas.DataFrame(L,index=indices,
                columns=['RMSE','NRMSE','R2score','WAI','RE(%)','MAE'])

writer = pandas.ExcelWriter(root + 'Summary_Regression2.xlsx', engine="xlsxwriter")

# Write each dataframe to a different worksheet.
Table_All.to_excel(writer, sheet_name="Using all data")
Table_Regions.to_excel(writer, sheet_name="Using Regionalized data")
writer.close()

