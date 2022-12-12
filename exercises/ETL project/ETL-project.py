from pyspark import SparkConf
from pyspark.sql import SparkSession
import pandas as pd

BUCKET = "dmacademy-course-assets"
KEYafter = "vlerick/after_release.csv"
KEYpre = "vlerick/pre_release.csv"

config = {
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
    "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
}
conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

#Read files from S3 bucker
dfpre_spark = spark.read.csv(f"s3a://{BUCKET}/{KEYpre}", header=True)
dfafter_spark = spark.read.csv(f"s3a://{BUCKET}/{KEYafter}", header=True)
dfpre_spark.show()

#Transform DataFrames to Pandas
dfpre = dfpre_spark.toPandas()
dfafter = dfafter_spark.toPandas()

#!/usr/bin/env python
# coding: utf-8

# # Introduction
# I am an entrenpreneur, trying to make it in the movie-world. I want to distinguish myself by offering specific services as a movie consultant. I want my speciality to be that I am able to make predictions of what imdb-score a movie will have. In this way, I hope to offer my services to movie-production companies, directors, actors... I hope to be able to predict what budget, cast, duration, language... the movie needs to have in order to score high!
# 
# My customers can invoke my services e.g. as a director/actor who only wants to have high-rated movies on his resume, as a production company to try and make a popular/high rated movie...

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from xgboost import XGBClassifier
import xgboost as xgb

# ## Merging the two dataframes and Visual exploration

# #### Comment
# 
# You do an inner join because not all the movie titles from the dfpre are in the dfafter, so you take a shortcut and don't need to clean these out!

# In[9]:


data = pd.merge(dfafter, dfpre, on='movie_title', how="inner")


# # Data cleaning

# #### Comment
# 
# Remove the columns which where part of after-release data that will not be predicted. 

# In[17]:


data = data.drop(["gross", "num_critic_for_reviews", "movie_title", "num_voted_users", "num_user_for_reviews", "movie_facebook_likes"], axis=1)


# #### Comment
# Remove potential duplicates

# In[19]:


data = data.duplicated(keep="first")


# #### Comment
# 
# Remove content_rating because of its missing values; We could opt to replace the missing values with the most common values, but this potentially drastically changes how the movie will be evaluated by the model!
# 

# In[21]:


data = data.drop(["content_rating"], axis=1)


# #### Comment
# 
# Drop the few left-over lines with missing values!

# In[24]:


data = data.dropna(inplace=True)


# In[32]:


data = data.drop(["director_name","actor_1_name", "actor_2_name", "actor_3_name"], axis=1)


# #### Comment
# 
# Keep every language that has a presence of >1% in the dataset as a seperate value, put other languages in a new category: other_language

# In[35]:


vals_l = data["language"].value_counts()[:3].index
data['language'] = data.language.where(data.language.isin(vals_l), 'other_language')



# #### Comment
# 
# Transform the language column into dummies and add them to the dataframe!

# In[37]:


ohe = OneHotEncoder()
data_l = ohe.fit_transform(data[['language']])
print(data_l.toarray())
data[ohe.categories_[0]] = data_l.toarray()


# #### Comment
# 
# Keep every country that has a presence of >1% in the dataset as a seperate value, put other languages in a new category: other_country
# 

# In[39]:


vals_l = data["country"].value_counts()[:6].index
print (vals_l)
data['country'] = data.country.where(data.country.isin(vals_l), 'other_country')


# #### Comment
# 
# Transform the country column into dummies and add them to the dataframe!

# In[41]:


ohe = OneHotEncoder()
data_c = ohe.fit_transform(data[['country']])
print(data_c.toarray())
data[ohe.categories_[0]] = data_c.toarray()


# #### Comment
# 
# Drop language and country after inserting their respective dummies.

# In[42]:


data = data.drop(["language", "country"], axis=1)


# #### Comment
# 
# Drop 1 random dummy from language and 1 random dummy from country to account for the coëfficient

# In[43]:


data = data.drop(["other_language", "other_country"], axis=1)


# #### Comment
# 
# Split up the genres into dummies!

# In[46]:


data_g = data['genres'].str.get_dummies(sep = '|')
combined_frames = [data, data_g]
data = pd.concat(combined_frames, axis = 1)
data = data.drop('genres', axis = 1)


# #### Comment
# 
# Drop actor_1_facebook_likes, actor_2_facebook_likes and actor_3_facebook_likes to anticipate multicollinearity with cast_total_facebook_likes

# In[47]:


data = data.drop(["actor_1_facebook_likes", "actor_2_facebook_likes", "actor_3_facebook_likes"], axis=1)


# # Linear prediction methods:
# Below, you are able to find models to predict the exact imdb_score a movie would have.

# ## Final preprocessing

# #### Comment
# 
# Define dependent and independent variables

# In[50]:


x = data.drop(["imdb_score"], axis=1) 
y = data["imdb_score"]


# #### Comment
# Split into training and validation dataset

# In[51]:


seed = 123 
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.15, random_state = seed)


# ## Linear Regression 

# #### Comment
# 
# Below I add an intercept to X and train the model

# In[52]:


xc_train = sm.add_constant(x_train)
xc_val = sm.add_constant(x_val)

mod = sm.OLS(y_train, xc_train)
olsm = mod.fit()


# #### Comment
# In the next step I predict the values and add these to a seperate dataframe so that I can easily use acces the predicted values in the future if necessary.

# In[53]:


array_pred = np.round(olsm.predict(xc_val),1) 

y_pred = pd.DataFrame({"y_pred": array_pred},index=x_val.index) 
val_pred_linreg = pd.concat([y_val,y_pred,x_val],axis=1)


# #### Comment
# Below I evaluate the model, calculating the R-square and the mean absolute error.

# In[54]:


act_value = val_pred_linreg["imdb_score"]
pred_value = val_pred_linreg["y_pred"]
rsquare = r2_score(act_value, pred_value)
mae = mean_absolute_error(act_value, pred_value)
pd.DataFrame({'eval_criteria': ['r-square','MAE'],'value':[rsquare,mae]})

# Auxiliar functions
def equivalent_type(f):
    if f == 'datetime64[ns]': return TimestampType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return DoubleType()
    elif f == 'float32': return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    try: typo = equivalent_type(format_type)
    except: typo = StringType()
    return StructField(string, typo)

# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)

# predictions = createDataFrame(val_pred_linreg)
# val_pred_linreg.show()