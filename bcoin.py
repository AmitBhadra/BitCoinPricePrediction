import pandas as pd
import quandl
import math
import numpy as np  
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = ''

df=quandl.get('BCHAIN/ATRCT')
df13=quandl.get('BCHAIN/TOTBC')
df4=quandl.get('BCHAIN/DIFF')
df2=quandl.get('BCHAIN/AVBLS')
df3=quandl.get('BCHAIN/CPTRA')
df5=quandl.get('BCHAIN/ETRAV')
df6=quandl.get('BCHAIN/HRATE')
df7=quandl.get('BCHAIN/MKTCP')
df8=quandl.get('BCHAIN/MIREV')
#df9=quandl.get('BCHAIN/
#df10=quandl.get('BCHAIN/
df11=quandl.get('BCHAIN/NTRAT')
df12=quandl.get('BCHAIN/NADDU')
df14=quandl.get('BCHAIN/MKPRU')


df=df[['Value']]
df['AVBLS']=(df2['Value'])
df['CPTRA']=(df3['Value'])
df['DIFF']=(df4['Value'])
df['ETRAV']=(df5['Value'])
df['HRATE']=(df6['Value'])
df['MKTCP']=(df7['Value'])
df['MIREV']=(df8['Value'])
#df['Diff']=(df4['Value'])
#df['Diff']=(df4['Value'])
df['NTRAT']=(df11['Value'])
df['NADDU']=(df12['Value'])
df['TOTBC']=(df13['Value'])
df['MKPRU']=(df14['Value'])



df=df[['Value','AVBLS','CPTRA','DIFF','ETRAV','HRATE','MKTCP','MIREV','NTRAT','NADDU','TOTBC','MKPRU']]
#print(df)
df=df[:-1]

forecast_col='MKPRU'
df.fillna(-99999,inplace=True)
forecast_out=1


df['Future']=df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)



X=np.array(df.drop(['Future'],1))
X=preprocessing.scale(X)
X_lately=X[-forecast_out:]
X=X[:-forecast_out]


#X=X[:-forecast_out+1]
df.dropna(inplace=True)
y=np.array(df['Future'])

#y=np.array(df['Future'])


print(len(X),len(y))

X_train, X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf=LinearRegression(n_jobs=-1)
#clf=svm.SVR()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
#print(df)
print(accuracy)

forecast_set=clf.predict(X_lately)
print(forecast_set,forecast_out)

print df.tail()

