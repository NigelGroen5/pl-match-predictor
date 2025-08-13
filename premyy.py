#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
matches = pd.read_csv("matches.csv", index_col=0)
matches.head()


# In[6]:


matches.shape


# In[8]:


matches.dtypes


# In[10]:


matches["date"] = pd.to_datetime(matches["date"])
matches.dtypes


# In[15]:


matches["venue_code"] = matches["venue"].astype("category").cat.codes


# In[19]:


matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")
matches


# 

# In[30]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split =10, random_state=1)


# In[32]:


train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']


# In[33]:


predictors = ["venue_code", "opp_code", "hour", "day_code"]


# In[34]:


rf.fit(train[predictors], train["target"])


# In[36]:


preds = rf.predict(test[predictors])


# In[39]:


from sklearn.metrics import accuracy_score


# In[40]:


acc = accuracy_score(test["target"], preds)


# In[41]:


acc


# In[46]:


combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))


# In[47]:


pd.crosstab(index=combined["actual"], columns=combined["prediction"])


# In[48]:


from sklearn.metrics import precision_score


# In[54]:


precision_score(test["target"], preds)


# In[55]:


grouped_matches = matches.groupby("team")


# In[56]:


group = grouped_matches.get_group("Manchester City")


# In[57]:


group


# In[58]:


def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


# In[59]:


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]


# In[60]:


new_cols


# In[61]:


rolling_averages(group, cols, new_cols)


# In[104]:


matches_rolling = (
    matches.groupby("team")
    .apply(lambda x: rolling_averages(x, cols, new_cols), include_groups=False)
    .reset_index()  # <— DON'T drop, keeps 'team' as a column
)


# In[116]:


matches_rolling


# In[117]:


matches_rolling.index = range(matches_rolling.shape[0])


# In[118]:


matches_rolling


# In[122]:


def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision


# In[123]:


combined, precision = make_predictions(matches_rolling, predictors + new_cols) # add new predictors in new_cols.


# In[124]:


precision


# In[125]:


combined


# In[126]:


combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index = True, right_index = True) # merge in more columns


# In[128]:


combined


# In[129]:


class MissingDict(dict): # makes names matchh
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"} 
mapping = MissingDict(**map_values)



# In[130]:


mapping["Arsenal"] #doesnt change bc of mapping


# In[131]:


mapping["West Ham United"]


# In[137]:


combined["new_team"] = combined["team"].map(mapping)


# In[138]:


combined


# In[139]:


merged = combined.merge(combined, left_on = ["date","new_team"], right_on=["date","opponent"]) #merge repeated games see if predictions are the same. merged into same row


# In[140]:


merged


# In[ ]:


merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()#when model predicted when team would win and team b would lose, what actuallyhppend


# In[1]:


27/40

