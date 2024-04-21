import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#Getting the necessary DataFrame
Matches = pd.read_csv('C:/Users/Lam/OneDrive/Máy tính/Data Stuffs/Machine-Learning-1-USTH/CSV files/WorldCupMatches.csv')
Champion = pd.read_csv("C:/Users/Lam/OneDrive/Máy tính/Data Stuffs/Machine-Learning-1-USTH/CSV files/WorldCups.csv")
Ranking = pd.read_csv("C:/Users/Lam/OneDrive/Máy tính/Data Stuffs/Machine-Learning-1-USTH/CSV files/fifa_ranking-2023-07-20.csv", parse_dates=["rank_date"])

#Since we want to predict future matches it is best to take the data of modern days
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2022-02-11')

#Dropping the unnecessary columns 
Ranking = Ranking[(Ranking["rank_date"]>start_date)&(Ranking["rank_date"]<end_date)]
Ranking = Ranking.drop(["rank","country_abrv","previous_points","confederation","rank_change","rank_date"], axis = 1)
Champion = Champion.drop(["Year","Country","Runners-Up","Third","Fourth","GoalsScored","QualifiedTeams","MatchesPlayed","Attendance"],axis = 1)
matches = Matches.drop(["Year"], axis = 1)


#Getting information on the columns
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

print(info(Matches))

print(Matches.columns)

# sns.countplot(Champion["Country"])
# plt.show()
#Replacing names from the Soviet era
def change_name(df):
    
    if df["Winner"] in ["Germany FR"]:
        df["Winner"] = "Germany"
        
    return df
Champion = Champion.apply(change_name,axis =1)
def change_name_team(df):
    
    if df["Home Team Name"] in ["IR Iran"]:
        df["Home Team Name"] = "Iran"
        
    if df["Away Team Name"] in ["IR Iran"]:
        df["Away Team Name"] = "Iran"
        
    if df["Home Team Name"] in ["Serbia and Montenegro"]:
        df["Home Team Name"] = "Serbia"
        
    if df["Away Team Name"] in ["Serbia and Montenegro"]:
        df["Away Team Name"] = "Serbia"
        
    return df
matches = matches.apply(change_name_team,axis = 1)
print("Results:", "Serbia" in list(Ranking["country_full"]))
def change_name_ranking(df):
    
    if df["country_full"] in ["IR Iran"]:
        df["country_full"] = "Iran"
    #This is because pandas cant find netherlands for some reason
    if df["country_full"] in ["Netherlands"]:
        df["country_full"] = "Netherlands"
    
    if df["country_full"] in ["Côte d'Ivoire"]:
        df["country_full"] = "Ivory Coast"
        
    return df
Ranking = Ranking.apply(change_name_ranking,axis = 1)

#Merge in order to get the fifa point for away and home team
matches = pd.merge(matches, Ranking, left_on="Home Team Name", right_on="country_full", how="left")
matches = pd.merge(matches, Ranking, left_on="Away Team Name", right_on="country_full", how="left", suffixes=("_home", "_away"))
champion_count = Champion["Winner"].value_counts().to_dict()

#relabeling the team name
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

#Adding columns for the amount of championship wons
matches["Championship Home"] = 0
matches["Championship Away"] = 0

def get_champion(row):
    if row["Home Team Name"] in champion_count:
        row["Championship Home"] = champion_count[row["Home Team Name"]]
        
    if row["Away Team Name"] in champion_count:
        row["Championship Away"] = champion_count[row["Away Team Name"]]
    
    return row
matches = matches.apply(get_champion, axis=1)

#Indexing the teams
matches["Home Team Name"] = matches["Home Team Name"].apply(lambda x: teams_index[x])
matches["Away Team Name"] = matches["Away Team Name"].apply(lambda x: teams_index[x])

#News columns for who wins
matches["Who Wins"] = 0
matches["Goal Difference"] = 0
matches["Goal Difference"] = matches["Home Team Goals"] - matches["Away Team Goals"]

def gettingwhowins(df):
    if df["Goal Difference"] > 0:
        df["Who Wins"] = 1 #Home team wins
        
    if df["Goal Difference"] < 0:
        df["Who Wins"] = 0 #Away team wins
    return df

matches = matches.apply(gettingwhowins,axis = 1)
matches = matches.drop(["country_full_home","country_full_away"],axis = 1)

#Seeing if any row have a NaN value
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
print(matches.head(10))
#Dropping columns that doesnt have fifapoints
matches = matches.dropna()
print(matches.shape)

#Dropping unpredictable columns
matches = matches.drop(["Home Team Goals","Away Team Goals","Goal Difference"],axis = 1)
print(matches.columns)
print(matches[(matches["Home Team Name"] == 36)|(matches["Away Team Name"] == 36)])
print(list(Ranking["country_full"]))
X = matches[["Home Team Name","Away Team Name","total_points_home","total_points_away","Championship Home","Championship Away"]].values
y = matches[["Who Wins"]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logreg = LogisticRegression()
# Fit the model to the training data
logreg.fit(X_train, y_train)

# Predict on the test data
y_pred = logreg.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)

rf = RandomForestClassifier()

# Fit the model to the training data
rf.fit(X_train, y_train)

# Predict on the test data
y_pred = rf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("Confusion Matrix:")
print(cm)
#To get user input
def getting_input(name1, name2):
    if name1 == name2:
        print("Invalide names")
        raise ValueError("Same Names")
    x = []
    try:
        x.append(teams_index[name1])
        x.append(teams_index[name2])
        x.append(Ranking.loc[Ranking["country_full"] == name1, "total_points"].values[0])
        x.append(Ranking.loc[Ranking["country_full"] == name2, "total_points"].values[0])
        
        if name1 in champion_count.keys():
            x.append(champion_count[name1])
        else:
            x.append(0)
            
        if name2 in champion_count.keys():
            x.append(champion_count[name2])
        else:
            x.append(0)
        
        x = np.array(x).reshape(1, -1)
    except Exception as e:
        print("Not valid names")
    return logreg.predict(x)[0]
print(getting_input("Germany","Brazil"))

def mua_giai(arr):
    print("Current matches:", arr)
    print("")
    if len(arr) == 1:
        print("Finals: ", arr)
        if getting_input(arr[0][0],arr[0][1]) == 1:
            return arr[0][0]
        else:
            return arr[0][1]
    
    next_round = []
    
    for i in range(0, len(arr), 2):
        match1 = arr[i]
        match2 = arr[i + 1]
        
        result1 = getting_input(match1[0], match1[1])
        result2 = getting_input(match2[0], match2[1])
        if result1 == 1:
            winner1 = match1[0]
            print("Match: ",match1)
            print("Winner is: ",match1[0])
            print("")
        else:
            winner1 = match1[1]
            print("Match: ",match1)
            print("Winner is: ",match1[1])
            print("")
            
        if result2 == 1:
            winner2 = match2[0]
            print("Match: ",match2)
            print("Winner is: ",match2[0])
            print("")
        else:
            winner2 = match2[1]
            print("Match: ",match2)
            print("Winner is: ",match2[1])
            print("")
        
        next_round.append([winner1, winner2])
    
    return mua_giai(next_round)

matches = [
    ["Netherlands", "Brazil"],
    ["Iran", "Spain"],
    ["Italy", "Argentina"],
    ["Paraguay", "Portugal"]
]

result = mua_giai(matches)
print(result)