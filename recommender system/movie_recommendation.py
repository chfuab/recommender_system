import pandas as pd
from scipy.spatial.distance import hamming
from scipy.spatial.distance import euclidean
import numpy as np
import streamlit as st


@st.cache
def get_rating():
    url = "data/ml-100k/u.data"
    names = ["user id", "item id", "rating", "timestamp"]
    return pd.read_csv(url, sep="\t", names=names)


@st.cache
def get_movie_details():
    url = 'data/ml-100k/u.item'
    encoding = 'unicode_escape'
    names = ["movie id", "movie title", "release date", "video release date",
             "IMDb URL", "unknown", "Action", "Adventure", "Animation",
             "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
             "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
             "Thriller", "War", "Western"]
    return pd.read_csv(url, sep="|", encoding=encoding, names=names)


df = get_rating()
df_item = get_movie_details()
df_item = df_item.set_index("movie id")
df_full = df.join(df_item, on="item id", how="left")


userItemMatrix = pd.pivot_table(df, index=['user id'], columns=[
                                'item id'], values="rating")


def nearestneighbours(user, K):
    # create a user df that contains all users except active user
    allUsers = pd.DataFrame(userItemMatrix.index)
    allUsers = allUsers[allUsers["user id"] != user]
    # Add a column to this df which contains distance of active user to each user
    allUsers["distance"] = allUsers["user id"].apply(
        lambda x: hamming(userItemMatrix.loc[user], userItemMatrix.loc[x]))
    KnearestUsers = allUsers.sort_values(
        ["distance"], ascending=True)["user id"][:K]
    return KnearestUsers


def topN(user, N=3):
    KnearestUsers = nearestneighbours(user, 10)
    # get the ratings given by nearest neighbours
    NNRatings = userItemMatrix[userItemMatrix.index.isin(KnearestUsers)]
    # Find the average rating of each movie rated by nearest neighbours
    avgRating = NNRatings.apply(np.nanmean).dropna()
    # drop the movies already watched by active user
    moviesAlreadyWatched = userItemMatrix.loc[user].dropna().index
    avgRating = avgRating[~avgRating.index.isin(moviesAlreadyWatched)]
    topNMovies = avgRating.sort_values(ascending=False).index[:N]
    return topNMovies


def movie_recommender(distance_method, movie_name, N):
    movie_id = df_item[df_item["movie title"] == movie_name].index[0]
    df_item_distance = pd.DataFrame(data=df_item.index)
    df_item_distance = df_item_distance[df_item.index != movie_id]
    df_item_distance['distance'] = df_item_distance["movie id"].apply(
        lambda x: distance_method(df_item.loc[x][4:], df_item.loc[movie_id][4:]))
    df_item_distance.sort_values(by='distance', inplace=True)
    return(df_item_distance.head(N))


st.title("Movie recommendation system")
st.header("List of movies:")
st.dataframe(df_item)
st.subheader("Enter user id to get recommendation:")
user_id = st.text_input("User ID")
if user_id:
    df_user_recommendation = df_item.iloc[topN(int(user_id)), 0:4]
    st.dataframe(df_user_recommendation)
else:
    st.dataframe()

st.subheader("Enter movie name to get recommendation:")
movie_name = st.text_input("Movie name")
if movie_name:
    df_movie_recommendation = movie_recommender(hamming, movie_name, 3)
    df_movie_recommendation = df_movie_recommendation.join(
        df_item, on="movie id", how='left')
    df_movie_recommendation = df_movie_recommendation.iloc[:, 2:6]
    st.dataframe(df_movie_recommendation)
else:
    st.dataframe()
