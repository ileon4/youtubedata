#!/usr/bin/env python
# coding: utf-8

# In this project, I am analyzing YouTube data. I will look at specific data within the top genres of YouTube channels and compare them to other variables. I first import pandas, numpy, and matplotlib. I will then upload the data, then look at the first five rows of the data set.

# In[79]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[80]:


df = pd.read_csv('/Users/inezleon/Downloads/top-300-youtube-channels.csv')


# In[81]:


df.head()


# Each row is a different channel with static statistics extracted at the time of the creation on this data set. Let's see how many columns and rows are in the data set.

# In[82]:


df.shape


# There are 296 rows (or channels) and 8 columns (variables). Let's see if there are any nulls we need to take care of.

# In[83]:


df.isna().any()


# Thankfully there are none. Let's see what percentage each genre of channel takes up of all the channels in the whole dataset.

# In[84]:


genre_pct = df['Genre'].value_counts(normalize=True)


# In[85]:


print(genre_pct)


# Let's see how many channels are in each genre in this data set.

# In[86]:


df['Genre'].value_counts()


# In[133]:


df.groupby('Genre').mean()


# Let's populate each genre with however many channel entries are necessary to get as close to Music's count in the data set. The calculation will be based off of the mean of any numerical categorical variables which are Rank, Subscriber_Count, Video_Views, Video_Count, and Channel_Started. 

# In[87]:


#Get the count of entries for each genre.
genre_counts = df['Genre'].value_counts()

#Get the name of the top genre.
top_genre = genre_counts.index[0]

#Get the mean statistics for each genre.
genre_means = df.groupby('Genre').mean()

# Create a new empty dataframe to store the balanced dataset
balanced_df = pd.DataFrame(columns=df.columns)

# Loop through the genres that are not the top genre
for genre in genre_counts.index[1:]:
    # Calculate the number of entries to add
    n_entries_to_add = genre_counts[top_genre] - genre_counts[genre]

    # Get the mean statistics for the genre
    genre_mean = genre_means.loc[genre]

    # Repeat the mean statistics n_entries_to_add times
    repeated_mean = pd.concat([genre_mean]*n_entries_to_add, axis=1).T
    
    # Add the genre column to the repeated mean statistics
    repeated_mean['Genre'] = genre

    # Append the repeated mean statistics to the balanced dataframe
    balanced_df = pd.concat([balanced_df, repeated_mean], ignore_index=True)

# Append the original top genre entries to the balanced dataframe
balanced_df = pd.concat([balanced_df, df[df['Genre'] == top_genre]], ignore_index=True)

# Shuffle the balanced dataframe
balanced_df


# Let's populate the new rows's Channel Name column.

# In[89]:


balanced_df['Channel_Name'] = balanced_df['Channel_Name'].fillna('channel')
balanced_df


# Let's turn any decimal values in the data set into integers.

# In[127]:


balanced_df['Channel_Started'] = balanced_df['Channel_Started'].astype('int')
balanced_df['Subscriber_Count'] = balanced_df['Subscriber_Count'].astype('int')
balanced_df['Video_Views'] = balanced_df['Video_Views'].astype('int')
balanced_df['Video_Count'] = balanced_df['Video_Count'].astype('int')

balanced_df


# Let's now see how many channels per genre there are.

# In[123]:


balanced_df['Genre'].value_counts()


# The data is no longer skewed. Although it is good to be able to populate a data set, the unequal distribution of channels per genre may accurately represent how many of each genre are actually on YouTube. For this reason, I will maintain df instead of balanced_df for the rest of the project.

# The top five genres of video are Music, Entertainment, People and Blogs, Film & Animation, and Gaming. Let's subset these so that they are in their own data frame which we can work with.

# In[128]:


top5 = balanced_df[(balanced_df['Genre'] == "Music") | (balanced_df['Genre'] == "Entertainment") | (balanced_df['Genre'] == "People & Blogs") | (balanced_df['Genre'] == "Film & Animation") | (balanced_df['Genre'] == "Gaming")]


# In[129]:


top5.head()


# This lets us know that any calculations we perform will be skewed towards those genres that have more channels. We will only be using the top 5 channels in this data set since the rest are so underrepresented.

# In[130]:


top5sub = top5[['Genre', 'Channel_Started', 'Subscriber_Count']]
top5sub


# Let's see how many total views each genre has in the original data set.

# In[131]:


df.groupby("Genre")["Video_Views"].sum().plot(kind='bar')
plt.show()


# Now in the populated one.

# In[132]:


balanced_df.groupby("Genre")["Video_Views"].sum().plot(kind='bar')
plt.show()


# Wow there's a big difference between the results from the original data and the populated data. (ADD TO)

# In[95]:


genre_year = df.pivot_table(values='Subscriber_Count', columns ='Channel_Started', index='Genre', aggfunc=np.sum)
genre_year


# Let's subset the genres that have the least null values.

# In[96]:


genre_year = genre_year.loc[['Music', 'Entertainment', 'Film & Animation', 'Gaming']]
genre_year


# Now let's get rid of the year columns that have way too many nulls to be able to visualize.

# In[97]:


genre_year = genre_year.drop(columns=[2017, 2018, 2019, 2020, 2021])


# In[98]:


genre_year


# Let's fill the null values with 0 so that calculations can be executed correctly.

# In[99]:


genre_year = genre_year.fillna(0)


# Now let's calculate the percentage of subscriber count of each year compared to the total subscriber count amonst all the years in the data frame per genre.

# In[100]:


genre_year['total'] = genre_year[2005] + genre_year[2006] + genre_year[2007] + genre_year[2008] + genre_year[2009] + genre_year[2010] + genre_year[2011] + genre_year[2012] + genre_year[2013] + genre_year[2014] + genre_year[2015] + genre_year[2016]
genre_year['2005pct'] = round((genre_year[2005]/genre_year['total'])*100, 2)
genre_year['2006pct'] = round((genre_year[2006]/genre_year['total'])*100, 2)
genre_year['2007pct'] = round((genre_year[2007]/genre_year['total'])*100, 2)
genre_year['2008pct'] = round((genre_year[2008]/genre_year['total'])*100, 2)
genre_year['2009pct'] = round((genre_year[2009]/genre_year['total'])*100, 2)
genre_year['2010pct'] = round((genre_year[2010]/genre_year['total'])*100, 2)
genre_year['2011pct'] = round((genre_year[2011]/genre_year['total'])*100, 2)
genre_year['2012pct'] = round((genre_year[2012]/genre_year['total'])*100, 2)
genre_year['2013pct'] = round((genre_year[2013]/genre_year['total'])*100, 2)
genre_year['2014pct'] = round((genre_year[2014]/genre_year['total'])*100, 2)
genre_year['2015pct'] = round((genre_year[2015]/genre_year['total'])*100, 2)
genre_year['2016pct'] = round((genre_year[2016]/genre_year['total'])*100, 2)
genre_year_pct = genre_year[['total', '2005pct', '2006pct', '2007pct', '2008pct', '2009pct', '2010pct', '2011pct', '2012pct', '2013pct', '2014pct', '2015pct', '2016pct']]
genre_year_pct


# In[160]:


year_pct_plot = pd.melt(genre_year_pct, id_vars='total', value_vars=['2005pct','2006pct','2007pct','2008pct','2009pct','2010pct','2011pct','2012pct','2013pct','2014pct','2015pct','2016pct'], var_name='Year', value_name='subcount_pct_genre', col_level=0)
year_pct_plot = year_pct_plot[['Year', 'subcount_pct_genre']]
year_pct_plot


# In[166]:


year_pct_plot.plot(kind='bar')


# We can analyse this data and note years within each genre that make up the highest subscriber count out of all the years. For example, 16.76% of the music genre's total subscriber count for this data set were earned from channels started in 2006.

# Let's use seaborn in order to make effective visualizations.

# I will make a line plot comparing subscriber count and the year channels started across the top five genres of channel.

# In[120]:


genre_chanstart_subcount = sns.relplot(x='Channel_Started', y='Subscriber_Count', kind='line', data=top5, hue='Genre', ci=None)
plt.show()


# In this plot there are spikes of note: Film & Animation in 2006/2007, Gaming in 2010 (probably because of the release of Minecraft?), People & Blogs in 2016 (the height of vlogger culture). If this were an analysis for YouTube corporate, it may be smart to subsidise sponsorships with channels in fairly stable and successful genres of YouTube channels.
# 
# Let's create the same plot but this time comparing video views and years channels were started.

# In[117]:


genre_chanstart_vidviews = sns.relplot(x='Channel_Started', y='Video_Views', kind='line', data=top5, hue='Genre', ci=None)
plt.show()


# In this data set, Film and Animation had a huge spike of video views around 2006/7. The rest of the notable spikes relatively follow those of the subscriber plot. This plot is also not as dramatic as the previous plot since people always subscribe to channels without actually consistently watching those channels' videos.
# 
# Let's look at the IQR's and the range of different statistics across the top five channels comparing video views.

# In[118]:


sns.catplot(x='Genre', y='Video_Views', data=top5, kind='box', hue='Genre', sym="")
plt.xticks(rotation=45)
plt.show()


# Most variance and range of video views between channels in Film & Animation genre and tightest range for channels in the Gaming genre.
# 
# Let's make the same plot comparing subscriber count across the top five channels.

# In[119]:


sns.catplot(x='Genre', y='Subscriber_Count', data=top5, kind='box', hue='Genre', sym="")
plt.xticks(rotation=45)
plt.show()


# Most variance and widest range between channels in the People & Blogs genre and the tightest range between channels in the Gaming genre again. Gaming channels have most consistency in subscriber count and video views.
