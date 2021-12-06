import pandas as pd

df = pd.read_csv('IMDb movies.csv')
df = df[["title", "original_title","date_published", "duration", "country", "language", "description", "avg_vote", "votes"]]

# Identify the movies where “date published” column value is not in proper format for Year, Month, Day (yyyy-mm-dd).
import datetime
def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        return False
    return True

print("Invalid Published Date Movies")
print(df[df['date_published'].apply(lambda date: not validate(date))][["title", "date_published"]])

# Apply scaling technique on “duration” column so that values are scaled between 0 to 1
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
scaled_df = df.copy()
scaled_df['scaled_duration'] = min_max_scaler.fit_transform(scaled_df[['duration']]).flatten()
print("Scaled Durations: ")
print(scaled_df[['title','scaled_duration']])

print("Top-5 Length Movies")
top_5 = scaled_df.sort_values('scaled_duration', ascending=False).head(5).drop(columns=['scaled_duration'])
print(top_5)
top_5.to_csv("top-5_length_movies.csv",index=False)

# Find all the movies which got a title change 
# (original title doesn’t match with the title of the release). 
# For those movies, use Fuzzy matching to find the match score of title of the release and original title. 
# Save those movies in a csv file and add a column for the match score.
from thefuzz import fuzz
def title_match_score(origin_title, title):
    score = fuzz.ratio(origin_title, title)
    return score
scores = []
for i in range(len(df)):
    title = df['title'][i]
    origin = df['original_title'][i]
    scores.append(title_match_score(origin, title))
    
df['fuzzy_scores'] = scores


title_changed_df = df[df['fuzzy_scores'].apply(lambda score: score < 100)]

print("Title Changed Movies")
print(title_changed_df)
title_changed_df.to_csv('title_changed_movies.csv', index=False)

# Output the names of all French movies with avg_vote higher than 6.5
print("All French movies with avg_vote higher than 6.5")
print(df[(df["country"] == 'France') | (df['avg_vote'] > 6.5)])

# Output the names of the movies which are about Romanian War (hints: look at the description column)
print("About Romanian War")
print(df[df['description'].apply(lambda text: "Romanian War".lower() in str(text).lower())])

# What percentages of the movies are missing “language” field? For all those which are missing “language” field, 
# fill the language field by looking at value in the “country” column
miss_lang_df = df[df['language'].isnull()]
print("Missing Language Movies: ", miss_lang_df)
# Percentage of missing language
print("Percentage: ", len(miss_lang_df)/len(df))
# update with the country
df.loc[list(miss_lang_df.index)]['language'] = list(miss_lang_df['country'])


# Now look at the “votes” column and create a new column “rank” 
# by applying scaling technique so that values of the “rank” column are between 0 to 1
vote_min_max_scaler = MinMaxScaler()
df['rank'] = vote_min_max_scaler.fit_transform(df[['votes']]).flatten()
print("Scaled as Ranks: ")
print(df)