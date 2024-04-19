# install.packages("robustbase")
# install.packages("readxl")
# install.packages("MLmetrics")
#install.packages("caret")
library(robustbase)
library(readxl)
library(MLmetrics)
#library(caret)

R2_adjusted = function(Y,Y_hat){
  num = median(abs(Y-Y_hat))
  den = median(abs(Y-mean(Y)))
  return (1.0 - (num/den)^2)
}

root = 'D:/OneDrive - AGROSAVIA - CORPORACION COLOMBIANA DE INVESTIGACION AGROPECUARIA/Escritorio/Corpoica/ET0 Gustavo/' 
file_et0 = paste(root, 'ET0_IDEAM.xlsx', sep = "")
et0 = read_excel(file_et0)
file_et0_nasa = paste(root, 'ET0_NasaPower.xlsx', sep = "")
et0_nasa = read_excel(file_et0_nasa)
file_et0_terra = paste(root, 'ET0_TerraClimate.xlsx', sep = "")
et0_terra = read_excel(file_et0_terra)
file_et0_eumetsat = paste(root, 'ET0_EUMETSAT.xlsx', sep = "")
et0_eumetsat = read_excel(file_et0_eumetsat)

R2score = matrix(0,3,2)
rownames(R2score) = c('Nasa Power','Terra Climate','EUMETSAT')
colnames(R2score) = c('2005-2020','1981-2020')

contr=lmrob.control()
contr$max.it = 1000
contr$maxit.scale = 1000
contr$k.max = 1000
contr$setting = 'KS2014'
contr$compute.outlier.stats = 'S'
set.seed(123) 

##########################
#        2005-2020       #
##########################
# NASA POWER
et0_2005 = et0[(et0$Year > 2004)>0,]
et0_2005 = et0_2005[,7:18,with=FALSE]
et0_nasa_2005 = et0_nasa[(et0_nasa$Year > 2004),]
et0_nasa_2005 = et0_nasa_2005[,7:18,with=FALSE]
bool = et0_2005 >0 & et0_nasa_2005 >0
et0_2005 = et0_2005[bool]
et0_nasa_2005 = et0_nasa_2005[bool]
mmfit <- lmrob(et0_2005 ~ et0_nasa_2005,method="MM",control=contr)
et0_hat = predict(mmfit)
W = sqrt(weights(mmfit, type = "robustness"))
R2score[1,1] = R2_Score(et0_2005*W,et0_hat*W)

# TERRA CLIMATE
et0_2005 = et0[(et0$Year > 2004)>0,]
et0_2005 = et0_2005[,7:18,with=FALSE]
et0_terra_2005 = et0_terra[(et0_terra$Year > 2004),]
et0_terra_2005 = et0_terra_2005[,7:18,with=FALSE]
bool = et0_2005 >0 & et0_terra_2005 >0
et0_2005 = et0_2005[bool]
et0_terra_2005 = et0_terra_2005[bool]
mmfit <- lmrob(et0_2005 ~ et0_terra_2005,method="MM",control=contr)
et0_hat = predict(mmfit)
W = sqrt(weights(mmfit, type = "robustness"))
R2score[2,1] = R2_Score(et0_2005*W,et0_hat*W)

# EUMETSAT
et0_2005 = et0[(et0$Year > 2004)>0,]
et0_2005 = et0_2005[,7:18,with=FALSE]
et0_eumetsat_2005 = et0_eumetsat[(et0_eumetsat$Year > 2004),]
et0_eumetsat_2005 = et0_eumetsat_2005[,7:18,with=FALSE]
bool = et0_2005 >0 & et0_eumetsat_2005 >0
et0_2005 = et0_2005[bool]
et0_eumetsat_2005 = et0_eumetsat_2005[bool]
mmfit <- lmrob(et0_2005 ~ et0_eumetsat_2005,method="MM",control=contr)
et0_hat = predict(mmfit)
W = sqrt(weights(mmfit, type = "robustness"))
R2score[3,1] = R2_Score(et0_2005*W,et0_hat*W)

##########################
#        1981-2020       #
##########################
# NASA POWER
et0_1981 = et0[,7:18,with=FALSE]
et0_nasa_1981 = et0_nasa[,7:18,with=FALSE]
bool = et0_1981 >0 & et0_nasa_1981 >0
et0_1981 = et0_1981[bool]
et0_nasa_1981 = et0_nasa_1981[bool]
mmfit <- lmrob(et0_1981 ~ et0_nasa_1981,method="MM",control=contr)
et0_hat = predict(mmfit)
W = sqrt(weights(mmfit, type = "robustness"))
R2score[1,2] = R2_Score(et0_1981*W,et0_hat*W)

# TERRA CLIMATE
et0_1981 = et0[,7:18,with=FALSE]
et0_terra_1981 = et0_terra[,7:18,with=FALSE]
bool = et0_1981 >0 & et0_terra_1981 >0
et0_1981 = et0_1981[bool]
et0_terra_1981 = et0_terra_1981[bool]
mmfit <- lmrob(et0_1981 ~ et0_terra_1981,method="MM",control=contr)
et0_hat = predict(mmfit)
W = sqrt(weights(mmfit, type = "robustness"))
R2score[2,2] = R2_Score(et0_1981*W,et0_hat*W)

print('Results using the full models')
print(R2score)

#########################
#   CROSS-VALIDATION    #
#########################
R2score_cv = matrix(0,10,2)
colnames(R2score_cv) = c('Nasa Power','Terra Climate')

##########################
#        1981-2020       #
##########################
#NASA POWER
et0_1981 = et0[,7:18,with=FALSE]
et0_nasa_1981 = et0_nasa[,7:18,with=FALSE]
bool = et0_1981 >0 & et0_nasa_1981 >0
et0_1981 = et0_1981[bool]
et0_nasa_1981 = et0_nasa_1981[bool]

mmfit=lmrob(et0_1981 ~ et0_nasa_1981, method = "MM",control=contr) 
W = sqrt(weights(mmfit, type = "robustness")) 

folds=cut(sample(et0_1981),breaks=10,labels=FALSE)
for(i in 1:10){
  trainIndexes=which(folds!=i,arr.ind=TRUE)
  testIndexes=which(folds==i,arr.ind=TRUE)
  W_i = W[testIndexes]
  mmfit_i=lmrob(et0_1981[trainIndexes]~et0_nasa_1981[trainIndexes],method="MM",control=contr) 
  et0_hat_i = predict(mmfit_i,data.frame(et0_1981[testIndexes]))
  R2score_cv[i,1] = R2_Score(et0_1981[testIndexes]*W_i,et0_hat_i*W_i)
}

#TERRA CLIMATE
et0_1981 = et0[,7:18,with=FALSE]
et0_terra_1981 = et0_terra[,7:18,with=FALSE]
bool = et0_1981 >0 & et0_terra_1981 >0
et0_1981 = et0_1981[bool]
et0_terra_1981 = et0_terra_1981[bool]
mmfit <- lmrob(et0_1981 ~ et0_terra_1981,method="MM",control=contr) 
W = sqrt(weights(mmfit, type = "robustness")) 

folds=cut(sample(et0_1981),breaks=10,labels=FALSE)
for(i in 1:10){
  trainIndexes=which(folds!=i,arr.ind=TRUE)
  testIndexes=which(folds==i,arr.ind=TRUE)
  W_i = W[testIndexes]
  mmfit_i=lmrob(et0_1981[trainIndexes]~et0_terra_1981[trainIndexes],method="MM",control=contr) 
  et0_hat_i = predict(mmfit_i,data.frame(et0_1981[testIndexes]))
  R2score_cv[i,2] = R2_Score(et0_1981[testIndexes]*W_i,et0_hat_i*W_i)
}

print('Results using Cross-validation')

print(R2score_cv)


