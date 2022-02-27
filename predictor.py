# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Read training dataset
df = pd.read_csv('all_matches.csv',low_memory=False)
# Get summary of dataset
#print(df.info())

# Group data into sets of 6 overs each
df1 = df.loc[(df['ball'] < 6.1)]

# Calculate batsmen strike rate
# Strike rate = Total runs/Total balls faced
df2 = df1.groupby(['striker']).size().reset_index(name='total_balls_faced')
df3 = df1.groupby(['striker']).sum().reset_index()
# Drop not needed columns
df3 =  df3.drop(columns=['ball','extras','wides','noballs','byes','legbyes','penalty','other_wicket_type','other_player_dismissed'])
dfbatsmen = pd.merge(df2,df3,on=['striker'])
dfbatsmen['strike_rate'] = dfbatsmen['runs_off_bat']/dfbatsmen['total_balls_faced']
# Print new batsmen dataset
#dfbatsmen.loc[dfbatsmen['striker'] == 'SK Raina']

# Get new strike rates from ESPN cric info site and compute new strike rates
newdf = pd.read_csv('newbatsmen.csv', low_memory=False)
newdf = newdf.fillna(0)
newdf.replace(r'\s*(.*?)\s*', r'\1', regex=True)
# Balls played = Total runs - Runs off which 4 and 6 were scored
newdf['balls_played'] = (newdf['runs_off_bat'] - (newdf['4s']*3+newdf['6s']*5))
newdf.drop(columns=['match_id', '4s', '6s'], inplace=True)
# Merge old and new strike rates into 1 file
dfnew = pd.merge(dfbatsmen,newdf,on=['striker'])
dfnew['total_balls_faced_new'] = dfnew['total_balls_faced'] + dfnew['balls_played']
dfnew['runs_off_bat'] = dfnew['runs_off_bat_x']+dfnew['runs_off_bat_y']
dfnew.drop(columns=['total_balls_faced','innings_x','innings_y',
                    'runs_off_bat_y','runs_off_bat_x','strike_rate'], inplace=True)
dfnew['strike_rate_new'] = dfnew['runs_off_bat']/dfnew['total_balls_faced_new']
# Rename coulmns to original values
dfnew.rename(columns={'strike_rate_new':'strike_rate',
              'total_balls_faced_new':'total_balls_bowled'},inplace=True)
#dfnew.loc[dfnew['striker'] == 'MS Dhoni']
dfbatsmen = dfnew
#dfbatsmen

# Calculate bowler economy
# Economy = Runs conceeded/Balls bowled
df2 = df1.groupby(['bowler']).size().reset_index(name='total_balls_bowled')
df3 = df1.groupby(['bowler']).sum().reset_index()
# Drop not needed columns
df3 =  df3.drop(columns=['ball','extras','wides','noballs','byes','legbyes','penalty','other_wicket_type','other_player_dismissed'])
dfbowler = pd.merge(df2,df3,on=['bowler'])
dfbowler['economy_rate'] = dfbowler['runs_off_bat']/dfbowler['total_balls_bowled']
# Display new bowler dataset
#dfbowler

# Read new dataset of bowler's economy from ESPN cricinfo
newbowlers = pd.read_csv('newbowlers.csv')
# Compute total balls as overs * 6
newbowlers['total_balls_bowled'] = newbowlers['overs']*6
newbowlers1 = pd.merge(dfbowler,newbowlers,on=['bowler'])
# Get total balls bowled as old + new balls bowled
newbowlers1['total_balls_bowled'] = newbowlers1['total_balls_bowled_x'] + newbowlers1['total_balls_bowled_y']
newbowlers1['runs_off_bat'] = newbowlers1['runs_off_bat_x'] + newbowlers1['runs_off_bat_y']
# Drop not needed columns
newbowlers1.drop(columns=['total_balls_bowled_x','total_balls_bowled_y',
                          'match_id_x','match_id_y','innings_x','innings_y',
                          'Econ','runs_off_bat_y',
                          'runs_off_bat_x'], inplace=True)

newbowlers1['economy_rate_new'] = newbowlers1['runs_off_bat']/newbowlers1['total_balls_bowled']
dfbowler = newbowlers1
#dfbowler

# Function to replace player string with numeric values
def unique_player(player,player_type):
  #print("Before: ", player)
  test = player.split(',')
  test = [s.strip() for s in test]
  # Get unique player names
  test = set(test)
  test = list(test)
  value = 0.0

  for item in test:
    if player_type == 0:
      # Get bowlers economy 
      try:
        value = value + dfbowler[dfbowler['bowler'] == item]['economy_rate'].values[0]
      except:
        value = value + 0.0
    elif player_type == 1:
      # Get batsmen strike rate
      try:
        value = value + dfbatsmen[dfbatsmen['striker'] == item]['strike_rate'].values[0]
      except:
        value = value + 0.0
      
  # Return average value for every row in dataset
  value = value/len(test)
  return value

# Group by total runs scored per match
matches_sum = df1.groupby(['match_id','innings']).sum()

# Append total runs per match to new dataframe
total_runs = matches_sum[['runs_off_bat']]
total_runs = total_runs.reset_index()

# Group venue, innings, ball, batting_team, bowling_team, bastmen and bowlers
matches_test = df1.iloc[:,0:11]
# Merge into 1 row
merged_batsmen = matches_test.groupby(['match_id','batting_team','bowling_team','venue','innings'],as_index=False).agg({'striker':', '.join}).reset_index()
merged_bowler = matches_test.groupby(['match_id','batting_team','bowling_team','venue','innings'],as_index=False).agg({'bowler':', '.join}).reset_index()

# Merge total_runs and common data to get final dataset
matches = pd.merge(merged_batsmen,merged_bowler,on=['match_id','batting_team','bowling_team','innings','venue'])
#check = matches_test.groupby(['venue', 'innings']).sum().reset_index()
df5 = matches

df5['average_strikerate'] = df5['striker']
for i, row in matches.iterrows():
  #print(row['striker'])
  df5.at[i, 'average_strikerate'] = unique_player(df5.at[i, 'striker'], 1)

df5['average_economy'] = df5['bowler']
for i, row in df5.iterrows():
  df5.at[i, 'average_economy'] = unique_player(df5.at[i, 'bowler'], 0)
df6 = pd.merge(df5,total_runs,on=['match_id','innings'])
df6.drop(columns=['index_x','striker','index_y','bowler'],inplace=True)
#df6

# Convert all string values into categorical values
venue_factorized, venue_categories = pd.factorize(df6['venue'])
df6['venue'] = venue_factorized
# Display 5 rows of dataframe with numeric values
df6.head()

# Encode teams as a number
teams = {
  'Kolkata Knight Riders':1,
  'Royal Challengers Bangalore':2,
  'Chennai Super Kings':3,
  'Kings XI Punjab':4, 
  'Punjab Kings':4,
  'Delhi Daredevils':10,
  'Rajasthan Royals':5, 
  'Mumbai Indians':6, 
  'Deccan Chargers':7,
  'Kochi Tuskers Kerala':8, 
  'Pune Warriors':11, 
  'Sunrisers Hyderabad':9,
  'Rising Pune Supergiants':12,
  'Gujarat Lions':13,
  'Rising Pune Supergiant':14, 
  'Delhi Capitals':10
  }

# Replace team name by number
df6['batting_team'].replace(teams,inplace=True)
df6['bowling_team'].replace(teams,inplace=True)
df6.drop(columns=['match_id'],inplace=True)
# Display numeric dataset
#df6

# Get input and output data
venue = df6['venue'].to_numpy()
innings = df6['innings'].to_numpy()
batting_team = df6['batting_team'].to_numpy()
bowling_team = df6['bowling_team'].to_numpy()
bowler = df6['average_economy'].to_numpy()
batsmen = df6['average_strikerate'].to_numpy()

# Output
y = df6['runs_off_bat'].to_numpy()
# Display no.of rows for input
#print(np.shape(venue_factorized))

# Create training and testing datasets
x = np.zeros((np.shape(venue_factorized)[0],6))
x[:,0] = venue
x[:,1] = innings
x[:,2] = batting_team
x[:,3] = bowling_team
x[:,4] = bowler
x[:,5] = batsmen
#print("Input size: ",x.shape)
#print("Output size: ",y.shape)

# Split datasets into testing and training
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# Using Random Forest regressor model
lin = RandomForestRegressor(n_estimators=100,max_features=None)
lin.fit(X_train,y_train)
y_pred = lin.predict(X_test)

# Metrics to check for
score = lin.score(X_test,y_test)*100
#print("R2 squared value:",score)
#print("Expected:",y_test)
#print("Predicted:",y_pred)

# Save model using pickle
filename = 'regression_model.pkl'
pickle.dump(lin,open(filename,"wb"))
#print('Saved model to disk!')

def predictRuns(testInput):

  # Read dataframe to test
  dataframe = pd.read_csv(testInput)
  dataframe1 = dataframe.rename(columns={'batsmen':'striker'})

  # Encode teams as a number
  teams = {
    'Kolkata Knight Riders':1,
    'Royal Challengers Bangalore':2,
    'Chennai Super Kings':3,
    'Kings XI Punjab':4, 
    'Punjab Kings':4,
    'Delhi Daredevils':10,
    'Rajasthan Royals':5, 
    'Mumbai Indians':6, 
    'Deccan Chargers':7,
    'Kochi Tuskers Kerala':8, 
    'Pune Warriors':11, 
    'Sunrisers Hyderabad':9,
    'Rising Pune Supergiants':12,
    'Gujarat Lions':13,
    'Rising Pune Supergiant':14, 
    'Delhi Capitals':10
    }

  # Replace team name by number
  dataframe1['batting_team'].replace(teams,inplace=True)
  dataframe1['bowling_team'].replace(teams,inplace=True)

  # Convert all string values into categorical values
  venue_factorized, venue_categories = pd.factorize(dataframe1['venue'])
  dataframe1['venue'] = venue_factorized

  # Create a column average economy in final datset
  dataframe1['average_economy'] = dataframe1['bowlers']
  dataframe1['average_strikerate'] = dataframe1['striker']

  for i, row in dataframe1.iterrows():
    # Call function to replace bowler string in dataset
    dataframe1.at[i, 'average_economy'] = unique_player(dataframe1.at[i, 'bowlers'], 0)
    dataframe1.at[i, 'average_strikerate'] = unique_player(dataframe1.at[i, 'striker'], 1)

      # Get input data to predict for
  venue = dataframe1['venue'].to_numpy()
  innings = dataframe1['innings'].to_numpy()
  batting_team = dataframe1['batting_team'].to_numpy()
  bowling_team = dataframe1['bowling_team'].to_numpy()
  bowler = dataframe1['average_economy'].to_numpy()
  batsmen = dataframe1['average_strikerate'].to_numpy()

  # Create training and testing datasets
  input_x = np.zeros((np.shape(venue_factorized)[0],6))
  input_x[:,0] = venue
  input_x[:,1] = innings
  input_x[:,2] = batting_team
  input_x[:,3] = bowling_team
  input_x[:,4] = bowler
  input_x[:,5] = batsmen

  # Load the model from disk
  filename = 'regression_model.pkl'

  # Read byte from pickle model
  loaded_model = pickle.load(open(filename,"rb"))
  runs = loaded_model.predict(input_x)

  # Get final score
  prediction = (int)(runs[0])
  
  return prediction