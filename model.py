#!/usr/bin/env python
# coding: utf-8

# # KAGGLE PROJECT: House Price Prediction
# 

# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import PCA


# In[5]:


#Load train.csv into a pandas dataFrame.
data=pd.read_csv("/Users/radhika/Desktop/House_pred_input/train.csv")


# In[6]:


print(data.shape)


# In[7]:


print(data.columns)


# In[5]:


pd.set_option('display.max_columns', None)
# to display all the column names


# In[6]:


data.head()


# In[ ]:





# In[7]:


data.info()


# In[5]:


#checking for null values if any
pd.set_option("display.max_rows" , None)
data.isnull().sum()


# In[6]:


data["MSSubClass"].value_counts()
#to find the sum of distinct categories


# In[10]:


data["MSZoning"].value_counts()
#there are 8diff zones but data is there for only 5zones


# In[11]:


data["Street"].value_counts()


# In[12]:


data["Alley"].value_counts()


# This indicates only 91 rows have value for Alley, rest 1369 are nulls.

# Street and Alley are similar and since we have many nulls in Alley, we can remove this column and use only street

# In[ ]:





# In[13]:


# 2-D Scatter plot with color-coding for each house type/class.

sns.set_style("whitegrid");
sns.FacetGrid(data, hue="MSZoning", height=4)    .map(plt.scatter, "MSSubClass", "LotArea")    .add_legend();
plt.show();


#meaningless


# In[ ]:





# In[29]:


sns.set_style("whitegrid");
sns.FacetGrid(data, hue="SaleCondition", height=4)    .map(plt.scatter, "MSSubClass", "LotArea")    .add_legend();
plt.show();


# 1.most of the abnormal sale conditions are of <50000 plot area and b/w subclass of 50-100
# 2.most of the family sale condition are <25000 plot area roughly
# 3.

# In[27]:



sns.set_style("whitegrid");
sns.FacetGrid(data, hue="Street", height=4)    .map(plt.scatter, "MSSubClass", "LotArea")    .add_legend();
plt.show();


# In[34]:



sns.set_style("whitegrid");
sns.FacetGrid(data, hue="MSZoning", height=4)    .map(plt.scatter, "SalePrice", "LotArea")    .add_legend();
plt.show();


# In[35]:



sns.set_style("whitegrid");
sns.FacetGrid(data, hue="Street", height=4)    .map(plt.scatter, "SalePrice", "LotArea")    .add_legend();
plt.show();


# In[46]:


# pairwise scatter plot: Pair-Plot
# Dis-advantages: 
##Can be used when number of features are high.
##Cannot visualize higher dimensional patterns in 3-D and 4-D. 
#Only possible to view 2D patterns.
plt.close();
sns.set_style("whitegrid");
sns.pairplot(data, hue="MSZoning", height=3);
plt.show()
# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.


# In[40]:


sns.FacetGrid(data, hue="MSZoning", height=5)    .map(sns.distplot, "SalePrice")    .add_legend();
plt.show();


# In[39]:


sns.boxplot(x='MSZoning',y='SalePrice', data=data)
plt.show()


# In[42]:


sns.boxplot(x='MSSubClass',y='SalePrice', data=data)
plt.show()


# In[43]:


sns.boxplot(x='LotShape',y='SalePrice', data=data)
plt.show()


# In[45]:


sns.violinplot(x="YrSold", y="SalePrice", data=data, size=8)
plt.show()


# HANDLING CATEGORICAL DATA
# Categorical data is of two types 1.Nominal data --> data not in any order --> onehotencoder is used 2.Ordinal data --> data in order --> LabelEncoder is used

# In[49]:


#MSZoning vs saleprice
sns.catplot(y='SalePrice',x='MSZoning',data=data.sort_values("SalePrice",ascending=False),kind="boxen",height=6,aspect=3)
plt.show()


# # Replacing nulls

# In[8]:


null_fillers={'BsmtQual':0,'BsmtCond':0,'BsmtExposure':0,'BsmtFinType1':0,'BsmtFinType2':0,'GarageType':'NoGrg',
              'GarageYrBlt':2005,'GarageFinish':0,'GarageQual':0,'GarageCond':0,'LotFrontage':60,
              'MasVnrType':'None','MasVnrArea':0}
data['MasVnrType'].value_counts()


# In[9]:


#As MSZoning is nominal categorical data,we will perform Onehot encoding

MSZone=data[["MSZoning"]]
MSZone=pd.get_dummies(MSZone,drop_first=True)
MSZone.head()


# In[10]:


street=data[["Street"]]
street=pd.get_dummies(street,drop_first=True)
street.head()


# In[11]:


utilities=data[["Utilities"]]
utilities=pd.get_dummies(utilities,drop_first=True)
utilities.head()


# In[12]:


bldgType=data[["BldgType"]]
bldgType=pd.get_dummies(bldgType,drop_first=True)
bldgType.head()


# In[13]:


houseStyle=data[["HouseStyle"]]
houseStyle=pd.get_dummies(houseStyle,drop_first=True)
houseStyle.head()


# In[14]:


garageType=data[["GarageType"]]
garageType=pd.get_dummies(garageType,drop_first=True)
garageType.head()


# In[15]:


saleType=data[["SaleType"]]
saleType=pd.get_dummies(saleType,drop_first=True)
saleType.head()


# In[16]:


saleCondition=data[["SaleCondition"]]
saleCondition=pd.get_dummies(saleCondition,drop_first=True)
saleCondition.head()


# In[ ]:


#Categorizing columns into Numerical or Categorical (norminal/ordinal) data types 

numerical_columns= ['Id', 'LotFrontage', 'LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','Bedroom','Kitchen','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']
categorical_nominal_columns= ['MSSubClass','LotShape', 'OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','KitchenQual','FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence']
categorical_ordinal_columns= ['MSZoning', 'Street', 'Alley','Landcontour' 'Utilities', 'LotConfig','LandSlope','Neighborhood', 'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','Electrical','Functional','GarageType','SaleType','SaleCondition']
#Some of ordinal columns could be nominal which can only be identified by checking comparing it with the output


# In[17]:


data.drop(['MSZoning', 'Street','Utilities','BldgType','HouseStyle','GarageType','SaleType','SaleCondition',
                              'LandContour','LotConfig','LandSlope','Neighborhood', 'Condition1','Condition2',
                              'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                              'Foundation','Heating','Electrical','Functional','MiscFeature'
                              ],axis=1,inplace=True)


# In[18]:


data.shape


# In[ ]:





# In[19]:


#since MSSubClass in int type
data["MSSubClass"].value_counts()


# In[19]:


data=pd.concat([data,MSZone,street,utilities,bldgType,houseStyle,garageType,saleType,saleCondition],axis=1)


# In[20]:


data.shape


# # Dropping a column

# In[21]:


data.drop(["Alley"],axis=1,inplace=True)


# In[22]:


data.shape


# In[ ]:





# In[ ]:





# 

# In[16]:





# # 1. Converting categorical nominal columns to numerical columns by label encoding

# In[23]:


def LotShape(val):
  lables={'Reg':3,'IR1':2,'IR2':3,'IR3':4}
  if val in lables: return lables[val]
  else:return val

def Multiple(val):
  lables={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}
  if val in lables: return lables[val]
  else:return val

def BsmtExposure(val):
  lables={'Gd':4,'Av':3,'Mn':2,'No':1,'NA':0}
  if val in lables: return lables[val]
  else:return val

def BsmtFinType(val):
  lables={'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}
  if val in lables: return lables[val]
  else:return val

def CentralAir(val):
  lables={'Y':1,'N':0}
  if val in lables: return lables[val]
  else:return val

def GarageFinish(val):
  lables={'Fin':3,'RFn':2,'Unf':1,'NA':0}
  if val in lables: return lables[val]
  else:return val

def PavedDrive(val):
  lables={'Y':2,'P':1,'N':0}
  if val in lables: return lables[val]
  else:return val

def Fence(val):
  lables={'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'NA':0}
  if val in lables: return lables[val]
  else:return val

data['LotShape']=data['LotShape'].apply(LotShape)
data['BsmtExposure']=data['BsmtExposure'].apply(BsmtExposure)
data['BsmtFinType1']=data['BsmtFinType1'].apply(BsmtFinType)
data['BsmtFinType2']=data['BsmtFinType2'].apply(BsmtFinType)
data['CentralAir']=data['CentralAir'].apply(CentralAir)
data['GarageFinish']=data['GarageFinish'].apply(GarageFinish)
data['PavedDrive']=data['PavedDrive'].apply(PavedDrive)
data['Fence']=data['Fence'].apply(Fence)
data['ExterQual']=data['ExterQual'].apply(Multiple)
data['ExterCond']=data['ExterCond'].apply(Multiple)
data['BsmtQual']=data['BsmtQual'].apply(Multiple)
data['BsmtCond']=data['BsmtCond'].apply(Multiple)
data['HeatingQC']=data['HeatingQC'].apply(Multiple)
data['KitchenQual']=data['KitchenQual'].apply(Multiple)
data['FireplaceQu']=data['FireplaceQu'].apply(Multiple)
data['GarageQual']=data['GarageQual'].apply(Multiple)
data['GarageCond']=data['GarageCond'].apply(Multiple)
data['PoolQC']=data['PoolQC'].apply(Multiple)
     


# In[25]:


categorical_nominal_columns= ['MSSubClass','LotShape', 'OverallQual','OverallCond','ExterQual',
                              'ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                              'HeatingQC','CentralAir','KitchenQual','FireplaceQu','GarageFinish','GarageQual',
                              'GarageCond','PavedDrive','PoolQC','Fence']


for i in categorical_nominal_columns:
  print(i)
  print(data[i].unique())
     


# In[ ]:





# # Time related

# In[24]:


data["age"]=data["YrSold"]-data["YearBuilt"]


# In[25]:


data["age"]


# In[26]:


data.drop(['YrSold', 'YearBuilt'], axis=1,inplace=True)


# In[33]:


pd.set_option('display.max_columns', None)
data.head()


# # Dropping few columns

# In[32]:


data.drop(['PoolQC', 'Fence'], axis=1,inplace=True)


# In[31]:


data.drop(['BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageFinish','GarageQual','GarageCond'],axis=1,inplace=True)


# In[44]:


data.drop(['BsmtCond','FireplaceQu'],axis=1,inplace=True)


# In[54]:


data.drop(['LotFrontage','GarageYrBlt','MasVnrArea'],axis=1,inplace=True)


# In[55]:


data.shape


# In[56]:


#checking for null values if any
pd.set_option("display.max_rows" , None)
data.isnull().sum()


# # FEATURE SELECTION

# In[57]:


data.shape


# In[ ]:





# In[58]:


data.columns


# In[41]:



pd.set_option('display.max_columns', None)
data.head()


# In[59]:


X=data.loc[:,['Id', 'MSSubClass', 'LotArea', 'LotShape', 'OverallQual', 'OverallCond',
       'YearRemodAdd', 'ExterQual', 'ExterCond', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', 'CentralAir', '1stFlrSF',
       '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
       'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 
       'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM',
       'Street_Pave', 'Utilities_NoSeWa', 'BldgType_2fmCon', 'BldgType_Duplex',
       'BldgType_Twnhs', 'BldgType_TwnhsE', 'HouseStyle_1.5Unf',
       'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf',
       'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl',
       'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn',
       'GarageType_CarPort', 'GarageType_Detchd', 'SaleType_CWD',
       'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw',
       'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'SaleCondition_AdjLand',
       'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal',
       'SaleCondition_Partial', 'age']]
X.head()


# In[60]:


Y=data.loc[:,[ 'SalePrice']]
Y.head()


# In[44]:


#Find correlation b/w dependent and independent features

plt.figure(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,cmap="RdYlGn")

plt.show()

#If two or more independent features are correlating more than 80%,
#they all represent the same and duplicates can be dropped to reduce the curse of dimensionality


# In[61]:


#Important feature using ExtraTreeRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor(n_estimators=100, random_state=0)
selection.fit(X,Y)


# In[62]:


print(selection.feature_importances_)


# In[63]:


#plot a graph of important features for better visualization

plt.figure(figsize=(12,8))
feat_importances=pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind="barh")
plt.show()


# # Fitting model using Random forest

# In[65]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=None)


# In[66]:


from sklearn.ensemble import RandomForestRegressor
reg_rf=RandomForestRegressor()
reg_rf.fit(X_train,y_train)


# In[67]:


y_pred=reg_rf.predict(X_test)


# In[68]:


reg_rf.score(X_train,y_train)
#R-squared sccore


# In[69]:


reg_rf.score(X_test,y_test)


# In[71]:


plt.scatter(y_test,y_pred,alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[72]:


from sklearn import metrics


# In[73]:


print("MAE:",metrics.mean_absolute_error(y_test,y_pred))
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[74]:


metrics.r2_score(y_test,y_pred)


# In[75]:


from sklearn.preprocessing import StandardScaler


# In[77]:


data = data
scale= StandardScaler()


# In[78]:


# separate the independent and dependent variables
X_data = data.data
target = data.target
 
# standardization of dependent variables
scaled_data = scale.fit_transform(X_data) 
print(scaled_data)


# In[ ]:




