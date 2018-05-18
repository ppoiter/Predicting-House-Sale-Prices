
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error as m_s_e
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle

get_ipython().magic('matplotlib inline')


# In[2]:


data = pd.read_csv("AmesHousing.tsv", delimiter="\t")
data.head(10)


# In[3]:


data.dtypes


# In[4]:


def transform_features(df):
    return df


# In[5]:


def select_features(df):
    return df[['Gr Liv Area', 'SalePrice']]


# In[6]:


def train_and_test(df):
    train = df[:1460]
    test = df[1460:]
    numeric_train = train.select_dtypes(include = ['integer', 'float'])
    numeric_test = test.select_dtypes(include = ['integer', 'float'])
    features = numeric_train.columns.drop('SalePrice')
    
    lr = LinearRegression()
    lr.fit(train[features], train['SalePrice'])
    test_predictions = lr.predict(test[features])

    test_mse = m_s_e(test['SalePrice'],test_predictions)
    test_rmse = np.sqrt(test_mse)
    
    return test_rmse
    
transform_data = transform_features(data)
filtered_data = select_features(transform_data)
rmse = train_and_test(filtered_data)

rmse


# In[7]:


data.shape[0]


# In[8]:


num_missing = data.isnull().sum()


# In[9]:


num_missing


# In[10]:


#Setting up filter so comment out for now
#filter series to drop anything missing more than 25%


# In[11]:


#drop_missing = num_missing[(num_missing > len(data)/4)].sort_values()
#data = data.drop(drop_missing.index, axis = 1)


# In[69]:


#data.isnull().sum()


# In[70]:


#Numerical columns: For columns with missing values, fill in with mode
#note that there may be better ways of doing this
#num_missing = data.select_dtypes(include = ['float', 'int']).isnull().sum()
#fixable_columns have more than one, less than 25% missing:
#fixable_numeric = num_missing[(num_missing <
                               #len(data)/4) &
                              #(num_missing > 0)].sort_values()
#fixable_numeric


# In[14]:


## Compute the most common value for each column in `fixable_numeric_missing_cols`.


# In[71]:


#replacement_values = data[fixable_numeric.index].mode().to_dict(orient = 'records')[0]


# In[72]:


#replacement_values


# In[73]:


#replace with replacement_values. Following command shows an advantage of dictionaries


# In[18]:


#data = data.fillna(replacement_values)


# In[19]:


#Now only columns with missing values should be text


# In[74]:


#data.isnull().sum().value_counts()


# In[75]:


#data.dtypes.value_counts()


# In[77]:


#num_missing_text = data.select_dtypes(include = ['object']).isnull().sum()


# In[78]:


#num_missing_text


# In[79]:


#num_missing_num = data.select_dtypes(include = ['int','float']).isnull().sum()


# In[80]:


#num_missing_num


# In[81]:


#for now, get rid of data with missing text values


# In[82]:


## Series object: column name -> number of missing values
#text_mv_counts = data.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

## Filter Series to columns containing *any* missing values
#drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]

#data = data.drop(drop_missing_cols_2.index, axis=1)


# In[83]:


#data.isnull().sum()


# In[84]:


#data.isnull().sum().value_counts()


# In[85]:


#Working with full data set


# In[86]:


#look at creating some columns that better capture relationships: eg year info


# In[87]:


#years_sold = data['Yr Sold'] - data['Year Built']
#check for nonsensical values
#years_sold[years_sold < 0]


# In[89]:


#years_since_remod = data['Yr Sold'] - data['Year Remod/Add']
#check for nonsensical values
#years_since_remod[years_since_remod < 0]


# In[90]:


#new columns
#data['years_sold'] = years_sold
#data['years_since_remod'] = years_since_remod

#drop nonsensical:
#data = data.drop([1702,2180,2181], axis = 0)

#drop original columns
#data = data.drop(['Year Built', 'Year Remod/Add'], axis = 1)


# In[91]:


#Drop features that arent going to be useful for ML like Parcel ID (PID)


# In[92]:


#data = data.drop(["PID", "Order"], axis = 1)


# In[39]:


#Drop columns that leak info about the final sale
#data = data.drop(["Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis = 1)


# In[93]:


#Now can update transform features with previous changes


# In[10]:


def transform_features(df):
    #drop columns with more than 25% missing values
    num_missing = df.isnull().sum()
    drop_missing = num_missing[(num_missing > len(df)/4)].sort_values()
    df = df.drop(drop_missing.index, axis=1)
    #Numerical columns: For columns with missing values, fill in with mode
    num_missing = df.select_dtypes(include = ['float', 'int']).isnull().sum()
    #fixable_columns have more than one, less than 25% missing:
    fixable_numeric = num_missing[(num_missing < len(df)/4) & (num_missing > 0)].sort_values()
    replacement_values = df[fixable_numeric.index].mode().to_dict(orient = 'records')[0]
    df = df.fillna(replacement_values)
    #For now, get rid of text columns with missing values
    ## Series object: column name -> number of missing values
    text_mv_counts = df.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)
    ## Filter Series to columns containing *any* missing values
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    df = df.drop(drop_missing_cols_2.index, axis=1)
    #look at creating some columns that better capture relationships: eg year info
    years_sold = df['Yr Sold'] - df['Year Built']
    years_since_remod = df['Yr Sold'] - df['Year Remod/Add']
    df['years_sold'] = years_sold
    df['years_since_remod'] = years_since_remod
    #drop nonsensical rows, where year count is negative:
    df = df.drop([1702,2180,2181], axis = 0)
    #drop original columns
    #Drop features that arent going to be useful for ML like Parcel ID (PID)
    #Drop columns that leak info about the final sale
    df = df.drop(["Mo Sold", 
                  "Sale Condition", "Sale Type", "Yr Sold",
                 "PID", "Order", 'Year Remod/Add'], axis = 1)
    return df


# In[11]:


def select_features(df):
    return df[['Gr Liv Area', 'SalePrice']]


# In[12]:


def train_and_test(df):
    train = df[:1460]
    test = df[1460:]
    numeric_train = train.select_dtypes(include = ['integer', 'float'])
    numeric_test = test.select_dtypes(include = ['integer', 'float'])
    features = numeric_train.columns.drop('SalePrice')
    
    lr = LinearRegression()
    lr.fit(train[features], train['SalePrice'])
    test_predictions = lr.predict(test[features])

    test_mse = m_s_e(test['SalePrice'],test_predictions)
    test_rmse = np.sqrt(test_mse)
    
    return test_rmse


# In[13]:


transform_data = transform_features(data)
filtered_data = select_features(transform_data)
rmse = train_and_test(filtered_data)

rmse


# In[14]:


numeric_data = transform_data.select_dtypes(include = ['integer', 'float'])
numeric_data


# In[15]:


#visualising correlation coefficients
corr_coefs = numeric_data.corr()['SalePrice'].abs().sort_values()
corr_coefs


# In[16]:


#lets only keep if over 0.3 (arbitrary)


# In[17]:


corr_coefs[corr_coefs > 0.3]


# In[18]:


#can also drop from dataframe if less than 0.3
transform_data = transform_data.drop(corr_coefs[corr_coefs < 0.3].index, axis = 1)


# In[19]:


corr_above3 = corr_coefs[corr_coefs > 0.3]


# In[20]:


corr_above3


# In[21]:


sorted_corr_above3 = corr_above3.abs().sort_values()
sorted_corr_above3


# In[22]:


corrmat = numeric_data[sorted_corr_above3.index].corr()


# In[23]:


sns.heatmap(corrmat)


# Strong correlations with Saleprice: OverallQual, Gr Liv Area
# 
# Reasonably Strong correlation with SalePrice: 1st Flr SF, Garage Area, Total Bsmt SF, Garage Cars
# 
# Negative Correlations with SalePrice: years sold and years since remod
# 
# Columns that are correlated with each other: Tot Rooms Above Ground/Gr Liv Area, Year Built/Garage Year Built, Bsmt Fin SF 1/Total Bsmt SF, Gr Liv Area/Full Bath, Total Bsmt SF/1st Floor SF, Garage Cars/Garage Area, Years Sold/Years Since Remod
# 
# Columns that are negatively correlated with each other: Years Sold/Garage Year Built, Years Since Remod/Garage Year Built, Overall Qual/Years Sold, Overall Qual/Years since remod, Garage Cars/Years Sold, Garage Cars/Years Since remod
# 

# In[24]:


# Think about which features in the df should be converted to categorical
#If listed as nominal, strong candidates
#List nominal


# Nominal:
# 
# MS SubClass, MS Zoning, Street, Alley, Land Contour, Lot Config, Neighbourhood, Condition 1, Condition 2, Bldg Type, House Style, Roof Style, Roof Matl, Exterior 1, Exterior 2, Mas Vnr Type, Foundation, Heating, Central Air, Garage Type, Misc Feature, Sale Type, Sale Condition
# 
# No categories with 100s of possible values

# In[25]:


#Next check columns with mostly same value
#create categorical list


# In[26]:


categorical = ['PID', 'MS SubClass', 'MS Zoning', 'Street', 'Alley', 'Land Contour', 
               'Lot Config','Neighbourhood', 'Condition 1', 'Condition 2', 'Bldg Type', 
               'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 
               'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air', 'Garage Type', 
               'Misc Feature', 'Sale Type', 'Sale Condition']


# In[27]:


#Now look at columns that are numerical but whose numbers dont have any semantic meaning


# In[28]:


num_no_meaning = ['Overall Qual', 'Overall Cond']
#not sure what to do with these


# In[29]:


#Check if categoricals still in transform_data


# In[31]:


transform_cat_cols = []
for col in categorical:
    if col in transform_data:
        transform_cat_cols.append(col)
        
#How many unique values are in each column?
unique_counts = transform_data[transform_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()
#drop if more than 15 to avoid length of dataset blowing out
#In a more comprehensive example this would be done with a bit more care
drop_cat = unique_counts[unique_counts > 15].index
transform_data = transform_data.drop(drop_cat, axis = 1)


# In[32]:


#now update select features


# In[33]:


def transform_features(df):
    #drop columns with more than 25% missing values
    num_missing = df.isnull().sum()
    drop_missing = num_missing[(num_missing > len(df)/4)].sort_values()
    df = df.drop(drop_missing.index, axis=1)
    #Numerical columns: For columns with missing values, fill in with mode
    num_missing = df.select_dtypes(include = ['float', 'int']).isnull().sum()
    #fixable_columns have more than one, less than 25% missing:
    fixable_numeric = num_missing[(num_missing < len(df)/4) & (num_missing > 0)].sort_values()
    replacement_values = df[fixable_numeric.index].mode().to_dict(orient = 'records')[0]
    df = df.fillna(replacement_values)
    #For now, get rid of text columns with missing values
    ## Series object: column name -> number of missing values
    text_mv_counts = df.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)
    ## Filter Series to columns containing *any* missing values
    drop_missing_cols_2 = text_mv_counts[text_mv_counts > 0]
    df = df.drop(drop_missing_cols_2.index, axis=1)
    #look at creating some columns that better capture relationships: eg year info
    years_sold = df['Yr Sold'] - df['Year Built']
    years_since_remod = df['Yr Sold'] - df['Year Remod/Add']
    df['years_sold'] = years_sold
    df['years_since_remod'] = years_since_remod
    #drop nonsensical rows, where year count is negative:
    df = df.drop([1702,2180,2181], axis = 0)
    #drop original columns
    #Drop features that arent going to be useful for ML like Parcel ID (PID)
    #Drop columns that leak info about the final sale
    df = df.drop(["Mo Sold", 
                  "Sale Condition", "Sale Type", "Yr Sold",
                 "PID", "Order", 'Year Remod/Add'], axis = 1)
    return df

def select_features(df, coef_threshold = 0.3, uniq_threshold = 15):
    numeric_data = df.select_dtypes(include = ['integer', 'float'])
    abs_corr_coefs = df.corr()['SalePrice'].abs().sort_values()
    df = df.drop(corr_coefs[abs_corr_coefs < coef_threshold].index, axis = 1)
    categorical = ['PID', 'MS SubClass', 'MS Zoning', 'Street', 'Alley', 'Land Contour', 
               'Lot Config','Neighbourhood', 'Condition 1', 'Condition 2', 'Bldg Type', 
               'House Style', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 
               'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air', 'Garage Type', 
               'Misc Feature', 'Sale Type', 'Sale Condition']
    transform_cat_cols = []
    for col in categorical:
        if col in df:
            transform_cat_cols.append(col)
    #How many unique values are in each column?
    unique_counts = df[transform_cat_cols].apply(lambda col: len(col.value_counts())).sort_values()
    #drop if more than 15 to avoid length of dataset blowing out
    #In a more comprehensive example this would be done with a bit more care
    drop_cat = unique_counts[unique_counts > uniq_threshold].index
    df = df.drop(drop_cat, axis = 1)
    #now take care of categoricals
    text_cols = df.select_dtypes(include = ['object'])
    for col in text_cols:
        df[col] = df[col].astype('category')
    df = pd.concat([df, pd.get_dummies(df.select_dtypes(include = ['category']))], axis = 1)
    return df


# In[36]:


from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, KFold

def train_and_test(df, k=0):
    numeric_df = df.select_dtypes(include = ['integer', 'float'])
    features = numeric_df.columns.drop('SalePrice')
    lr = LinearRegression()
    
    if k == 0:
        train = df[:1460]
        test = df[1460:]
        lr.fit(train[features], train['SalePrice'])
        test_predictions = lr.predict(test[features])
        test_mse = m_s_e(test['SalePrice'],test_predictions)
        test_rmse = np.sqrt(test_mse)
        return test_rmse
    
    if k == 1:
        df = shuffle(df)
        fold_one = df[:1460]
        fold_two = df[1460:]
        lr.fit(fold_one[features], fold_one['SalePrice'])
        test_predictions_1 = lr.predict(fold_two[features])
        test_1_mse = m_s_e(fold_two['SalePrice'],test_predictions_1)
        test_1_rmse = np.sqrt(test_1_mse)
        
        lr.fit(fold_two[features], fold_two['SalePrice'])
        test_predictions_2 = lr.predict(fold_one[features])
        test_2_mse = m_s_e(fold_one['SalePrice'],test_predictions_2)
        test_2_rmse = np.sqrt(test_2_mse)
        average_rmse = (test_1_rmse+test_2_rmse) / 2
        return average_rmse
    
    else:
        kf = KFold(n_splits=k, shuffle=True)
        rmse_values = []
        for train_index, test_index, in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            lr.fit(train[features], train["SalePrice"])
            predictions = lr.predict(test[features])
            mse = m_s_e(test["SalePrice"], predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
        print(rmse_values)
        avg_rmse = np.mean(rmse_values)
        return avg_rmse

df = pd.read_csv("AmesHousing.tsv", delimiter="\t")
transform_df = transform_features(df)
filtered_df = select_features(transform_df)
rmse = train_and_test(filtered_df, k=4)

rmse
        

