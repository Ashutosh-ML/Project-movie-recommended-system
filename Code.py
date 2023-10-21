#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
movies= pd.read_csv('tmdb_5000_movies.csv')
credit=pd.read_csv('tmdb_5000_credits.csv')


movies['production_companies'].values
movies.shape

credit.head(5)


# In[69]:


#Merge both Dataset title comman column and name it as 'Movie'

movies = movies.merge(credit,on='title')
movies.head()


# # We will create tags on basis of that will create recommendation system 
# 

# In[70]:


#select most import column which can help us create tags

movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)


# In[71]:


movies.columns


# In[72]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies


# In[73]:


#Check missing data
movies.isnull().sum()
# three missing values present in Overview
#Drop missing value
movies.dropna(inplace=True)


# In[74]:


#check Duplcate in dataset
movies.duplicated().sum()

#dataset in zero duplicate values


# In[75]:


#We have string here in the form of dictionary in genres columns and we have to extract name and save it to new column
movies.iloc[0].genres


# In[76]:


# extract genres from genres column through above fuction and save it to genres column

import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[77]:


movies.dropna(inplace=True)


# In[78]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[79]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[80]:


#we have to extract name of top three characters, cast coulmn is collection of dictionary so we have to extract name of characters from first three dictionaries
import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[101]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L


# In[102]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[88]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[95]:


#extract director from crew column and save it to crew column
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[103]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[105]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[106]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[107]:


movies.head()


# In[108]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[109]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[110]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])


# In[111]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[113]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[115]:


vector = cv.fit_transform(new['tags']).toarray()


# In[116]:


vector.shape


# In[118]:


from sklearn.metrics.pairwise import cosine_similarity


# In[119]:


similarity = cosine_similarity(vector)


# In[120]:


similarity


# In[121]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[122]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[123]:


recommend('Gandhi')


# In[124]:


import pickle


# In[125]:


pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




