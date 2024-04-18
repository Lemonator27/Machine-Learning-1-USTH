import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
Matches = pd.read_csv('C:/Users/Lam/OneDrive/Máy tính/Data Stuffs/Machine-Learning-1-USTH/CSV files/WorldCupMatches.csv')
Champion = pd.read_csv("C:/Users/Lam/OneDrive/Máy tính/Data Stuffs/Machine-Learning-1-USTH/CSV files/WorldCups.csv")
Ranking = pd.read_csv("C:/Users/Lam/OneDrive/Máy tính/Data Stuffs/Machine-Learning-1-USTH/CSV files/fifa_ranking-2023-07-20.csv", parse_dates=["rank_date"])
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2022-02-11')
Ranking = Ranking[(Ranking["rank_date"]>start_date)&(Ranking["rank_date"]<end_date)]
Ranking = Ranking.drop(["rank","country_abrv","previous_points","confederation","rank_change","rank_date"], axis = 1)
print(Ranking.head(10))
Champion = Champion.drop(["Year","Country","Runners-Up","Third","Fourth","GoalsScored","QualifiedTeams","MatchesPlayed","Attendance"],axis = 1)
#Since variables like Year,Datetime,Stage
def info(df):
    variables = []
    data_types = []
    count = []
    unique = []
    missing_values = []
    #Getting the info of each columns and finding total missing values
    for item in df.columns:
        variables.append(item)
        data_types.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing_values.append(df[item].isna().sum())
        
    output = pd.DataFrame({
        'variable': variables, 
        'data type': data_types,
        'count': count,
        'unique': unique,
        'missing values': missing_values
    })    
        
    return output
#This shows that the Matches dataframe has null values
print(info(Matches))
Matches = Matches.dropna()

print(Matches.columns)

# sns.countplot(Champion["Country"])
# plt.show()
#Replacing names from the Soviet era

matches = Matches.drop(["Year"], axis = 1)
#relabeling the team name
matches = pd.merge(matches, Ranking, left_on="Home Team Name", right_on="country_full", how="left")
matches = pd.merge(matches, Ranking, left_on="Away Team Name", right_on="country_full", how="left", suffixes=("_home", "_away"))
champion_count = Champion["Winner"].value_counts().to_dict()

def indexing_theteams(df):
    teams = {}
    index = 0
    for lab,row in matches.iterrows():
        if row["Home Team Name"] not in teams.keys():
            teams[row["Home Team Name"]] = index
            index += 1
        if row["Away Team Name"] not in teams.keys():
            teams[row["Away Team Name"]] = index
            index += 1
    return teams
teams_index = indexing_theteams(matches)
print(teams_index)

matches["Home Team Name"] = matches["Home Team Name"].apply(lambda x: teams_index[x])
matches["Away Team Name"] = matches["Away Team Name"].apply(lambda x: teams_index[x])
matches["Who Wins"] = 0
matches["Goal Difference"] = 0
matches["Championship Won"] = 0
matches["Goal Difference"] = matches["Home Team Goals"] - matches["Away Team Goals"]

def gettingwhowins(df):
    if df["Goal Difference"] > 0:
        df["Who Wins"] = 1 #Home team wins
    if df["Goal Difference"] < 0:
        df["Who Wins"] = 0 #Away team wins
    return df

matches = matches.apply(gettingwhowins,axis = 1)
matches = matches.drop(["country_full_home","country_full_away"],axis = 1)
# matches = matches.drop(["Home Team Goals","Away Team Goals"],axis = 1)

def getting_missing(df):
    missing_values = {}
    away_name = []
    home_name = []
    away_point = []
    home_point = []
    
    for index, row in df.iterrows():
        if row.isna().any():
            home_name.append(row.loc["Home Team Name"])
            away_name.append(row.loc["Away Team Name"])
            home_point.append(row.loc["total_points_home"])
            away_point.append(row.loc["total_points_away"])
    
    missing_values["home_name"] = home_name
    missing_values["away_name"] = away_name
    missing_values["home_point"] = home_point
    missing_values["away_point"] = away_point
    return pd.DataFrame(missing_values)

MISS = getting_missing(matches)
print(MISS)
def getting_team_info(df):
    Teams = {}
    Name = []
    Fifa_point = []
    Championship = []
    for lab,row in df.iterrows():
        if row["Home Team Name"] not in Name:
            Name.append(row["Home Team Name"])
        if row["Away Team Name"] not in Name:
            Name.append(row["Away Team Name"])
print(matches.tail(10))

        