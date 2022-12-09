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



# ## Merging the two dataframes and Visual exploration

# #### Comment
# 
# You do an inner join because not all the movie titles from the dfpre are in the dfafter, so you take a shortcut and don't need to clean these out!

# In[9]:


data = pd.merge(dfafter, dfpre, on='movie_title', how="inner")
data.shape


# # Data cleaning

# #### Comment
# 
# Remove the columns which where part of after-release data that will not be predicted. 

# In[17]:


data = data.drop(["gross", "num_critic_for_reviews", "movie_title", "num_voted_users", "num_user_for_reviews", "movie_facebook_likes"], axis=1)


# #### Comment
# Remove potential duplicates

# In[19]:


data.duplicated(keep="first")


# In[21]:


data = data.drop(["content_rating"], axis=1)


# In[24]:


data.dropna(inplace=True)


# In[32]:


data = data.drop(["director_name","actor_1_name", "actor_2_name", "actor_3_name"], axis=1)


# In[35]:


vals_l = data["language"].value_counts()[:3].index
print (vals_l)
data['language'] = data.language.where(data.language.isin(vals_l), 'other_language')


# In[37]:


ohe = OneHotEncoder()
data_l = ohe.fit_transform(data[['language']])
print(data_l.toarray())
data[ohe.categories_[0]] = data_l.toarray()


# In[39]:


vals_l = data["country"].value_counts()[:6].index
print (vals_l)
data['country'] = data.country.where(data.country.isin(vals_l), 'other_country')


# In[41]:


ohe = OneHotEncoder()
data_c = ohe.fit_transform(data[['country']])
print(data_c.toarray())
data[ohe.categories_[0]] = data_c.toarray()


# In[42]:


data = data.drop(["language", "country"], axis=1)


# In[43]:


data = data.drop(["other_language", "other_country"], axis=1)


# In[46]:


data_g = data['genres'].str.get_dummies(sep = '|')
combined_frames = [data, data_g]
data = pd.concat(combined_frames, axis = 1)
data = data.drop('genres', axis = 1)


# In[47]:


data = data.drop(["actor_1_facebook_likes", "actor_2_facebook_likes", "actor_3_facebook_likes"], axis=1)
data.shape


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


# # Classification

# In[57]:


data["imdb_score_classes"]=pd.cut(data['imdb_score'], bins=[0,5,7,10], right=True, labels=False)
data


# ## Final preprocessing

# In[59]:


data_c = data.drop(["imdb_score"], axis=1)


# In[60]:


x = data_c.drop(["imdb_score_classes"], axis=1) 
y = data_c["imdb_score_classes"]


# In[61]:


seed = 123 
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.15, random_state = seed)


# In[62]:


sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_val = sc_x.transform(x_val)


# In[ ]:


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


# ## Logistic regression

# In[64]:


logit =LogisticRegression()
logit.fit(x_train,np.ravel(y_train,order='C'))
y_pred=logit.predict(x_val)
cnf_matrix = metrics.confusion_matrix(y_val, y_pred)
make_confusion_matrix(cnf_matrix, figsize = (10,10))

predictions = createDataFrame(y_pred)
y_pred.show()