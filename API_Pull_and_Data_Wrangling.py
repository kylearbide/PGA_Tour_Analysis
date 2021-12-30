# Databricks notebook source
# MAGIC %sh
# MAGIC pip install --upgrade pip
# MAGIC pip install wget

# COMMAND ----------

import datetime
import wget
import requests
import json
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC Importing the Players and Tournament Datasets

# COMMAND ----------


instance_id = "https://api.sportsdata.io/golf/v2/json/Players?key="
key = "c76c6101adbf4b0abb54a7a6eb5ddbb4"
url = f"{instance_id}{key}"

response = requests.get(
url = url)

# COMMAND ----------

Players = pd.DataFrame((json.loads(response.text)))

# COMMAND ----------

#Players.to_csv('/dbfs/FileStore/karbide/Players.txt')

# COMMAND ----------

instance_id = "https://api.sportsdata.io/golf/v2/json/Tournaments?key="
key = "c76c6101adbf4b0abb54a7a6eb5ddbb4"
url = f"{instance_id}{key}"

response = requests.get(
url = url)

Tournaments = pd.DataFrame((json.loads(response.text)))

# COMMAND ----------

Tournaments["New_Date"] = pd.to_datetime(Tournaments["StartDate"])

# COMMAND ----------

Last_Season = Tournaments.loc[Tournaments["New_Date"]>"2020-09-08"]
Last_Season = Last_Season.loc[Last_Season["New_Date"]<"2021-09-10"]
Last_Season = Last_Season.loc[Last_Season["Name"] != 'QBE Shootout']

# COMMAND ----------

#Last_Season.to_csv('/dbfs/FileStore/karbide/Last_Season.txt')

# COMMAND ----------

# MAGIC %md
# MAGIC Lets pull just the top 150 players

# COMMAND ----------

top150 = spark.read.csv("/FileStore/karbide/top150players.csv")
top150 = top150.toPandas()
top150.columns = ["Id","Name","Rating"]

# COMMAND ----------

Players['Full Name'] = Players[['FirstName','LastName']].agg(' '.join, axis=1)

# COMMAND ----------

top150Players = Players.merge(top150, how = "inner", left_on= "Full Name", right_on = "Name")
top150Players.shape

# COMMAND ----------

#we lost some players, lets try to see why
top150_list = top150["Name"].tolist()
mergedPlayers_list = top150Players["Name"].tolist()
dropped = [x for x in top150_list if x not in mergedPlayers_list]
droppednames = pd.DataFrame(dropped)
droppednames.columns = ["Name"]
droppednames[["Name1","Name2","Name3"]] = droppednames["Name"].str.split(" ",2,expand=True)


# COMMAND ----------

#We lose less players if we use the draft kings names
top150Players2 = Players.merge(top150, how = "inner", left_on= "DraftKingsName", right_on = "Name")
top150Players2.shape

# COMMAND ----------

mergedPlayers_list = top150Players2["Name"].tolist()
dropped = [x for x in top150_list if x not in mergedPlayers_list]
droppednames = pd.DataFrame(dropped)
droppednames.columns = ["Name"]
droppednames[["Name1","Name2","Name3"]] = droppednames["Name"].str.split(" ",2,expand=True)
droppednames

# COMMAND ----------

#this is not small enough that we can search 1 by 1
print(Players.loc[Players["FirstName"] == 'Erik'])
Players.at[4443,'DraftKingsName'] = "Erik van Rooyen"

# COMMAND ----------

print(Players["DraftKingsName"].loc[Players["LastName"] == 'Lee'])
print(top150.loc[top150["Name"]== 'K.H. Lee'])
top150.at[58,"Name"] = "Kyoung-Hoon Lee"

# COMMAND ----------

print(Players["DraftKingsName"].loc[Players["FirstName"] == 'Robert'])
Players.at[2688,"DraftKingsName"] = "Robert MacIntyre"

# COMMAND ----------

print(Players["DraftKingsName"].loc[Players["LastName"] == 'Munoz'])
print(top150.loc[top150["Name"]== 'Sebasti�n Mu�oz'])
top150.at[66,"Name"] = "Sebastian Munoz"

# COMMAND ----------

print(Players["DraftKingsName"].loc[Players["LastName"] == 'Davis'])
print(top150.loc[top150["Name"]== 'Cam Davis'])
top150.at[69,"Name"] = "Cameron Davis"

# COMMAND ----------

Players.loc[Players["LastName"] == 'Van Tonder']
Players.at[4446,"DraftKingsName"] = "Daniel van Tonder"

# COMMAND ----------

top150Players2 = Players.merge(top150, how = "inner", left_on= "DraftKingsName", right_on = "Name")
top150Players2.shape
#now we have all the players

# COMMAND ----------

#the sport data api requires us to input the player and tournament ID for the hole by hole scores, lets see if we can loop and dowload automatically
top150PlayerIDs = top150Players2["PlayerID"].tolist()
Tourney_IDs = Last_Season["TournamentID"].tolist()

# COMMAND ----------

#top150Players2.to_csv('/dbfs/FileStore/karbide/top150playersexpanded.txt')

# COMMAND ----------

#tournament ID
a = 453
#player ID
b = 40000047

instance_id = "https://api.sportsdata.io/golf/v2/json/PlayerTournamentStatsByPlayer/"
key = "?key=c76c6101adbf4b0abb54a7a6eb5ddbb4"
url = f"{instance_id}{a}/{b}{key}"

response = requests.get(
url = url)

test = json.loads(response.text)

# COMMAND ----------

test2 = test['Rounds']
RoundData = pd.DataFrame(test2)

# COMMAND ----------

R1Scores = test2[0]["Holes"]
R2Scores = test2[1]["Holes"]
R3Scores = test2[2]["Holes"]
R4Scores = test2[3]["Holes"]
R1Scoresdf = pd.DataFrame(R1Scores)

# COMMAND ----------

def holeScore(data):
  ScoresSet = data.drop(["PlayerRoundID","Par","Score","HoleInOne","ToPar"], axis = 1)
  ScoresSet.set_index("Number", inplace = True)
  HoleScores = ScoresSet[ScoresSet == 1].stack()
  
  
  HoleScoresdf = pd.DataFrame(HoleScores).reset_index()
  HoleScoresdf.columns = ["Number","Hole_Score","1"]
  HoleScoresdf.drop(["1"],axis = 1, inplace = True)
  
  Scoresdf = data.merge(HoleScoresdf, on = "Number")
  
  return(Scoresdf)

# COMMAND ----------

def numScore(data):
  data["Hole_ScoreNum"] = [-3 if x == "DoubleEagle" else -2 if x == "Eagle" else -1 if x == "Birdie" else 0 if x == "IsPar" else 1 if x == "Bogey" else 2 if x == "DoubleBogey" else 3 for x in data["Hole_Score"]]

# COMMAND ----------

R1Scoresdf = holeScore(R1Scoresdf)
numScore(R1Scoresdf)

# COMMAND ----------

roundScore = sum(R1Scoresdf['Hole_ScoreNum'])
roundShots = sum(R1Scoresdf['Par'])-roundScore
birdies = sum(R1Scoresdf["Birdie"])

# COMMAND ----------

def roundSummary(data,roundNum, playerid, tournamentid):
  roundScore = sum(data['Hole_ScoreNum'])
  roundShots = sum(data['Par'])+roundScore
  doubleeagles = sum(data["DoubleEagle"])
  eagles = sum(data["Eagle"])
  birdies = sum(data["Birdie"])
  pars = sum(data["IsPar"])
  bogeys = sum(data["Bogey"])
  doublebogeys = sum(data['DoubleBogey'])
  worsethandoublebogeys = sum(data['WorseThanDoubleBogey'])
  roundID = data["PlayerRoundID"][0]
  
  roundStats = pd.DataFrame(np.array([[roundScore,roundShots,doubleeagles,eagles,birdies,pars,bogeys,doublebogeys,worsethandoublebogeys,roundID,roundNum,playerid,tournamentid]]),  columns=['RoundScore','RoundShots','DoubleEagles','Eagles','Birdies','Pars','Bogeys','DoubleBogeys','WorseThanDoubleBogeys','PlayerRoundID','RoundNum','PlayerID','TournamentID'])
  return(roundStats)

# COMMAND ----------

roundSummary(R1Scoresdf,1,b,a)

# COMMAND ----------

def dictToDf(scoresDict,playerid,tournamentid):
  roundDict = scoresDict['Rounds']
  rounddf = pd.DataFrame(roundDict)
  rounddf = rounddf.loc[rounddf["Par"] > 0]
  rounds = rounddf["Number"].tolist()
  dfTournamentRounds =  pd.DataFrame(columns = ['RoundScore','RoundShots','DoubleEagles','Eagles','Birdies','Pars','Bogeys','DoubleBogeys','WorseThanDoubleBogeys','PlayerRoundID','RoundNum','PlayerID','TournamentID'])
  dfTournamentHoles = pd.DataFrame(columns = ['PlayerRoundID', 'Number', 'Par', 'Score', 'ToPar', 'HoleInOne','DoubleEagle', 'Eagle', 'Birdie', 'IsPar', 'Bogey', 'DoubleBogey','WorseThanDoubleBogey', 'Round','Hole_Score', 'Hole_ScoreNum', "Player_ID", "Tournament_ID"])

  for x in rounds:
    roundHoles = roundDict[x-1]["Holes"]
    roundHoles = pd.DataFrame(roundHoles)
    roundHoles = holeScore(roundHoles)
    numScore(roundHoles)
    roundHoles["Round"] = x
    roundHoles["Player_ID"] = playerid
    roundHoles["Tournament_ID"] = tournamentid
    roundstat = roundSummary(roundHoles,x,playerid,tournamentid)
    dfTournamentRounds = pd.concat([dfTournamentRounds,roundstat])
    dfTournamentHoles = pd.concat([dfTournamentHoles,roundHoles])
  
  return(dfTournamentRounds,dfTournamentHoles)
    
    

# COMMAND ----------

# testRounds,testHoles = dictToDf(test,b,a)
#print(testHoles.head(20))
#print(testRounds)

# COMMAND ----------

# MAGIC %md
# MAGIC Testing a condition if the player didnt play in one the tournament/ the dict is empty

# COMMAND ----------

# Example of a empty tournament
url = f"{instance_id}450/40000047{key}"
response = requests.get(url = url)
bool(response.text)

# COMMAND ----------

# Example of a valid tournament
url = f"{instance_id}451/40000047{key}"
response = requests.get(url = url)
bool(response.text)

# COMMAND ----------

def allTournaments(playerID,tournamentList,key):
  
  allTournamentRounds = pd.DataFrame(columns = ['RoundScore','RoundShots','DoubleEagles','Eagles','Birdies','Pars','Bogeys','DoubleBogeys','WorseThanDoubleBogeys','PlayerRoundID','RoundNum','PlayerID','TournamentID'])
  allTournamentHoles = pd.DataFrame(columns = ['PlayerRoundID', 'Number', 'Par', 'Score', 'ToPar', 'HoleInOne','DoubleEagle', 'Eagle', 'Birdie', 'IsPar', 'Bogey', 'DoubleBogey','WorseThanDoubleBogey', 'Round','Hole_Score', 'Hole_ScoreNum'])
  
  key = f"?key={key}"
  instance_id = "https://api.sportsdata.io/golf/v2/json/PlayerTournamentStatsByPlayer/"
  
  
  
  for x in tournamentList:
    url = f"{instance_id}{x}/{playerID}{key}"
    response = requests.get(url = url)
    if bool(response.text):
      importdata = json.loads(response.text)
      testDict = importdata['Rounds']
      if bool(testDict):
        xroundDf,xholesDf = dictToDf(importdata,playerID,x)

        allTournamentRounds = pd.concat([allTournamentRounds,xroundDf])
        allTournamentHoles = pd.concat([allTournamentHoles,xholesDf])
  
  return(allTournamentRounds,allTournamentHoles)

# COMMAND ----------

#testRounds,testHoles = allTournaments(b,Tourney_IDs,"c76c6101adbf4b0abb54a7a6eb5ddbb4")

# COMMAND ----------

def allPlayers(playerList,tournamentList,key):
  allPlayerRounds = pd.DataFrame(columns = ['RoundScore','RoundShots','DoubleEagles','Eagles','Birdies','Pars','Bogeys','DoubleBogeys','WorseThanDoubleBogeys','PlayerRoundID','RoundNum','PlayerID','TournamentID'])
  allPlayerHoles = pd.DataFrame(columns = ['PlayerRoundID', 'Number', 'Par', 'Score', 'ToPar', 'HoleInOne','DoubleEagle', 'Eagle', 'Birdie', 'IsPar', 'Bogey', 'DoubleBogey','WorseThanDoubleBogey', 'Round','Hole_Score', 'Hole_ScoreNum'])
  
  for i in playerList:
    yrounds,yholes = allTournaments(i,tournamentList,key)
    allPlayerRounds = pd.concat([allPlayerRounds,yrounds])
    allPlayerHoles = pd.concat([allPlayerHoles,yholes])
    
  return(allPlayerRounds,allPlayerHoles)


# COMMAND ----------

#RoundsDf, HolesDf = allPlayers(top150PlayerIDs,Tourney_IDs,"c76c6101adbf4b0abb54a7a6eb5ddbb4")

# COMMAND ----------

#RoundsDf.to_csv('/dbfs/FileStore/karbide/Rounds.txt')
#HolesDf.to_csv('/dbfs/FileStore/karbide/Holes.txt')

# COMMAND ----------

#so I dont have to run the API pull again
RoundsDf = pd.read_csv('/dbfs/FileStore/karbide/Rounds.txt')
HolesDf = pd.read_csv("/dbfs/FileStore/karbide/Holes.txt")

# COMMAND ----------

# if a tournament doesnt have at least 100 rounds
def dropSmallTournaments(data,threshold):
  tournamentCounts = data.groupby("TournamentID").size().reset_index(name="counts")
  bigTournaments = tournamentCounts.loc[tournamentCounts["counts"] > threshold]
  
  
  
  result = data.merge(bigTournaments, how = "inner", on = "TournamentID")
  result.drop(["counts"],axis=1)
  
  dropTournaments = tournamentCounts.loc[tournamentCounts["counts"] < threshold]
  dropped = dropTournaments["TournamentID"].tolist()
  
  print("Dropped Tournaments")
  for x in dropped:
    print(x)
    
  return(result)

# COMMAND ----------

RoundsDf = dropSmallTournaments(RoundsDf,100)

# COMMAND ----------

RoundsDf.groupby("TournamentID").size().reset_index(name="counts")

# COMMAND ----------

#RoundsDf.to_csv('/dbfs/FileStore/karbide/Rounds.txt')

# COMMAND ----------

testRoundDf = pd.read_csv('/dbfs/FileStore/karbide/Rounds.txt')

# COMMAND ----------

pStats = pd.read_csv('/dbfs/FileStore/karbide/pga_tour_stats_2020.csv')

# COMMAND ----------

pStats.columns

# COMMAND ----------

pStats.describe()

# COMMAND ----------

# I'm going to select ~30 of these stats, then later check for independence 
pStatsKeep = pStats[['PLAYER NAME',"GIR_PCT_FAIRWAY_BUNKER",	"GIR_PCT_FAIRWAY", "GIR_PCT_OVERALL", 'GIR_PCT_OVER_100', 'GIR_PCT_OVER_200', 'GIR_PCT_UNDER_100', 'GREEN_PCT_SCRAMBLE_SAND', 'GREEN_PCT_SCRAMBLE_ROUGH', 'FINISHES_TOP10', 'TEE_AVG_BALL_SPEED', 'TEE_AVG_DRIVING_DISTANCE', 'TEE_DRIVING_ACCURACY_PCT','TEE_AVG_LAUNCH_ANGLE', 'TEE_AVG_LEFT_ROUGH_TENDENCY_PCT', 'TEE_AVG_RIGHT_ROUGH_TENDENCY_PCT', 'TEE_AVG_SPIN_RATE', 'PUTTING_AVG_ONE_PUTTS',
       'PUTTING_AVG_TWO_PUTTS', 'PUTTING_AVG_DIST_BIRDIE', "PUTTING_AVG_PUTTS"]]

# COMMAND ----------

# Average Birdie Putt Distance is currently in feet and inches, can we change this to just inches
def split_Dist(item):
  if item != "nan":
    spDist = item.split("' ")
    ft_ = float(spDist[0])
    in_ = float(spDist[1].replace("\"",""))
    return (12*ft_) + in_


# COMMAND ----------

pStatsKeep = pStatsKeep.astype({"PUTTING_AVG_DIST_BIRDIE":"str"})


pStatsKeep["PUTTING_AVG_DIST_BIRDIE_INCH"] = pStatsKeep["PUTTING_AVG_DIST_BIRDIE"].apply(lambda x:split_Dist(x))

# COMMAND ----------

pStatsKeep.describe()

# COMMAND ----------

pStatsKeep["FINISHES_TOP10"].fillna(0,inplace = True)
pStatsKeep.dropna(inplace = True)
pStatsKeep.drop_duplicates(inplace = True)

# COMMAND ----------

pStatsKeep.groupby("PLAYER NAME").agg({"GIR_PCT_FAIRWAY_BUNKER": "count"}).sort_values("GIR_PCT_FAIRWAY_BUNKER", ascending = False).head(3)
# we still have 8 zach johnsons so lets dump him

# COMMAND ----------

pStatsKeep = pStatsKeep.loc[pStatsKeep["PLAYER NAME"] != "Zach Johnson"]

# COMMAND ----------

#Now we have to add Player IDs

PlayerNames = pd.read_csv("/dbfs/FileStore/karbide/Players.txt")
PlayerNames = PlayerNames[["DraftKingsName","PlayerID"]]

# COMMAND ----------

pStatsKeepIDs = pStatsKeep.merge(PlayerNames, how = "left", left_on = "PLAYER NAME", right_on = "DraftKingsName")

# COMMAND ----------

pStatsKeepIDsDropped = pStatsKeepIDs.loc[pStatsKeepIDs["PlayerID"].isna()]

# COMMAND ----------

pStatsKeepIDsDropped["PLAYER NAME"]

# COMMAND ----------

#again, we have to manually adjust these names

# COMMAND ----------

PlayerNames2 = pd.read_csv("/dbfs/FileStore/karbide/Players.txt")
PlayerNames2 = PlayerNames2[["DraftKingsName","PlayerID","FirstName","LastName"]]

# COMMAND ----------

print(PlayerNames2.loc[PlayerNames2["FirstName"] == "Ted"])
pStatsKeepIDs.at[5,"PlayerID"] = 40001173


# COMMAND ----------

print(PlayerNames2.loc[PlayerNames2["FirstName"] == "Fabian"])
pStatsKeepIDs.at[25,"PlayerID"] = 40000514

# COMMAND ----------

print(PlayerNames2.loc[PlayerNames2["LastName"] == "Gordon"])
pStatsKeepIDs.at[36,"PlayerID"] = 40003663

# COMMAND ----------

print(PlayerNames2.loc[PlayerNames2["LastName"] == "Ventura"])
pStatsKeepIDs.at[74,"PlayerID"] = 40003179

# COMMAND ----------

print(PlayerNames2.loc[PlayerNames2["LastName"] == "Fitzpatrick"])
pStatsKeepIDs.at[136,"PlayerID"] = 40000430

# COMMAND ----------

print(PlayerNames2.loc[PlayerNames2["LastName"] == "Pan"])
pStatsKeepIDs.at[140,"PlayerID"] = 40001109

# COMMAND ----------

print(PlayerNames2.loc[PlayerNames2["FirstName"] == "Sebastian"])
pStatsKeepIDs.at[161,"PlayerID"] = 40001682

# COMMAND ----------

sum(pStatsKeepIDs["PlayerID"].isna())
#now every player has a name and ID
#I might consider runnning the API pull again for this list of players

# COMMAND ----------

pStatsKeepIDs.drop(["DraftKingsName"], axis = 1, inplace = True)

# COMMAND ----------

pStatsKeepIDs = pStatsKeepIDs.astype({"PlayerID":"int"})

# COMMAND ----------

#pStatsKeepIDs.to_csv('/dbfs/FileStore/karbide/PlayerStats.txt')

# COMMAND ----------

StatPlayers = pStatsKeepIDs["PlayerID"].tolist()

# COMMAND ----------

#RoundsDf, HolesDf = allPlayers(StatPlayers,Tourney_IDs,"c76c6101adbf4b0abb54a7a6eb5ddbb4")

# COMMAND ----------

RoundsDf = dropSmallTournaments(RoundsDf,100)

# COMMAND ----------

#RoundsDf.to_csv('/dbfs/FileStore/karbide/Rounds.txt')
#HolesDf.to_csv('/dbfs/FileStore/karbide/Holes.txt')

# COMMAND ----------

StrokesGained = pd.read_csv("/dbfs/FileStore/karbide/StrokesGained.csv", encoding = 'latin-1')

# COMMAND ----------

StrokesGained.head()
print(StrokesGained.shape())

# COMMAND ----------

pIDs = pStatsKeepIDs[["PLAYER NAME", "PlayerID"]]
StrokesGainedIDs = StrokesGained.merge(pIDs, how = "inner", on = "PLAYER NAME")

# COMMAND ----------

#StrokesGainedIDs.to_csv("/dbfs/FileStore/karbide/StrokesGainedIDs.txt")