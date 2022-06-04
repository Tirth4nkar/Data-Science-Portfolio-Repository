import numpy as np
import pandas as pd

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
df.head()
movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()

#merging two tables
df = pd.merge(df,movie_titles,on='item_id')
df.head()

#Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
df.groupby('title')['rating'].count().sort_values(ascending=False).head()
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)
ratings['rating'].hist(bins=70)
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
#Okay! Now that we have a general idea of what the data looks like.
#let's move on to creating a simple recommendation system.

#Recommending Similar Movies
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()
#Let's choose two movies: starwars, a sci-fi movie. And Liar Liar, a comedy. 
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
#Now if we sort the dataframe by correlation, we should get the most similar movies, however note that we get some results that don't really make sense. This is because there are a lot of movies only watched once by users who also watched star wars (it was the most popular movie).
#Let's fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier).
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.dropna(inplace=True)
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()

#Now the same for the comedy Liar Liar:
corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()