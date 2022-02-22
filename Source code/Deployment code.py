#!/usr/bin/env python
# coding: utf-8

# In[1]:

# installing necessary packages
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation

# In[2]:


st.title("Devotional Songs Recommender System")

# In[ ]:
# reading CSV file
dataz = pd.read_csv(r'C:\Users\vaibh\triplets_file.csv')
data = dataz.sample(14000)
users = list(data['user_id'].unique())

var1 = st.sidebar.radio("New user?", ["Yes", "No"])


if var1 == "Yes":
    #Cold start code
    data1 =data.groupby('song_id')['listen_count'].sum()
    data12 = pd.DataFrame(data1)
    sorted_data12  = data12.sort_values("listen_count", ascending= False)
    Sorted_data12_reset_index = sorted_data12.reset_index()

    st.header("Recommended list of songs is :")

    for i in range(20):
        st.write(i+1, Sorted_data12_reset_index["song_id"][i])

else:
    
    customer_id = st.selectbox("Please select the user-ID: ", users)
    # Creating pivot table
    user_songs_df = data.pivot_table(index='user_id',
                                     columns='song_id',
                                     values='listen_count')

    # replacing NANs with 0
    user_songs_df.fillna(0, inplace=True)

    #Calculating Cosine Similarity between Users

    user_sim = 1 - pairwise_distances( user_songs_df.values,metric='cosine')

    # storing user similarity inside the dataframe 
    user_sim_df = pd.DataFrame(user_sim)

    #Set the index and column names to user ids 
    user_sim_df.index = list(user_songs_df.index)
    user_sim_df.columns = list(user_songs_df.index)

    np.fill_diagonal(user_sim, 0)


    # function for giving the recommendation
    def give_reco(cus_id):
        tem=list(user_sim_df.sort_values([customer_id],ascending=False).head(10).index)
        songs_list=[]
        for i in tem:
            songs_list=songs_list+list(data[data["user_id"]==i]["song_id"])

        a = set(songs_list)-set(data[data["user_id"]==customer_id]["song_id"])
        
        st.header("Recommended list of songs is:")
        for item in a:
            st.write(item)
            
    give_reco(customer_id)









