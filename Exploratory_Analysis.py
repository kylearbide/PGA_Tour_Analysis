# Databricks notebook source
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

# COMMAND ----------

roundsDf = pd.read_csv("/dbfs/FileStore/karbide/Rounds.txt")
holesDf = pd.read_csv("/dbfs/FileStore/karbide/Holes.txt")
holesDf.drop(["Score",	"ToPar", "Unnamed: 0"],axis=1,inplace = True)
roundsDf.drop(["Unnamed: 0"],axis=1,inplace = True)

# COMMAND ----------

roundsDf.head(10)

# COMMAND ----------

print(roundsDf.shape)
print(roundsDf["PlayerID"].nunique())
print(roundsDf["TournamentID"].nunique())

# COMMAND ----------

holesDf.head(10)

# COMMAND ----------

print(holesDf.shape)
print(holesDf["Player_ID"].nunique())
print(holesDf["Tournament_ID"].nunique())

# COMMAND ----------

# MAGIC %md
# MAGIC For our exploratory analysis, lets go one dataset at a time
# MAGIC 
# MAGIC **ROUNDS**

# COMMAND ----------

roundsDf["RoundScore"].describe()

# COMMAND ----------

roundsDf.describe()
# I want to see the round where there were no Pars

# COMMAND ----------

roundsDf.loc[roundsDf["Pars"]==0]
#they are mostly from tournament 448, lets bring in the tournament names and find this one

# COMMAND ----------

Tournaments = pd.read_csv("/dbfs/FileStore/karbide/Last_Season.txt")

# COMMAND ----------

Tournaments.loc[Tournaments["TournamentID"] == 429]

# COMMAND ----------

# the Barracuda Championship uses an alternate scoring format than that of the rest of the tour so it is easiest just to remove it from our sets
roundsDf = roundsDf.loc[roundsDf["TournamentID"] != 448]
holesDf = holesDf.loc[holesDf["Tournament_ID"] != 448]

# COMMAND ----------

# lets also remove rounds where they were not completed 

roundsDf["Total_Holes"] = roundsDf["DoubleEagles"] + roundsDf["Eagles"] + roundsDf["Birdies"]+ roundsDf["Pars"]+ roundsDf["Bogeys"]+ roundsDf["DoubleBogeys"]+ roundsDf["WorseThanDoubleBogeys"]
roundsDf = roundsDf.loc[roundsDf["Total_Holes"] == 18]

# COMMAND ----------

#roundsDf.to_csv("/dbfs/FileStore/karbide/RoundsReg.txt")

# COMMAND ----------

roundsDf.describe()

# COMMAND ----------

display(roundsDf["RoundScore"].hist(bins = 27))
#looks symetrical and normally distributed

# COMMAND ----------

Tournaments = pd.read_csv("/dbfs/FileStore/karbide/Last_Season.txt")
TournamentNames = Tournaments[["TournamentID","Name"]]


# COMMAND ----------

Players = pd.read_csv("/dbfs/FileStore/karbide/PlayerStats.txt")
PlayerNames = Players[["PlayerID","PLAYER NAME"]]

# COMMAND ----------

roundsDf = roundsDf.merge(PlayerNames, how = "left", left_on = "PlayerID", right_on = "PlayerID")

# COMMAND ----------

roundsDf = roundsDf.merge(TournamentNames, how = "left", on = "TournamentID")

# COMMAND ----------

# average round score by tournament
tournamentAverages = roundsDf.groupby("Name").agg({"RoundScore" : ['mean']})
tournamentAverages.reset_index(inplace = True)
tournamentAverages.columns = ["TournamentName","Mean"]
tournamentAverages.sort_values("Mean", inplace = True)

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC fig, ax = plt.subplots(figsize=(5, 13))
# MAGIC ax.barh(tournamentAverages["TournamentName"],tournamentAverages["Mean"])
# MAGIC 
# MAGIC plt.autoscale(enable=True, axis='y', tight=False)
# MAGIC plt.xlabel("Average Round Score (to Par)")
# MAGIC plt.ylabel("Tournament")
# MAGIC plt.title("Average Round Score by Tournament")
# MAGIC plt.grid(axis='x')
# MAGIC 
# MAGIC display(fig)

# COMMAND ----------

PlayerAverages = roundsDf.groupby(["PLAYER NAME","PlayerID"]).agg({"RoundScore" : ['mean']})
PlayerAverages.reset_index(inplace = True)
PlayerAverages.columns = ["PlayerName","PlayerID","Mean"]
PlayerAverages.sort_values("Mean", inplace = True)

PlayerAveragesTop = PlayerAverages[0:10]
PlayerAveragesBottom = PlayerAverages[-10:]

PlayersChart = pd.concat([PlayerAveragesTop,PlayerAveragesBottom])
PlayersChart.sort_values("Mean", inplace = True, ascending = False)

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC fig, ax = plt.subplots(figsize=(5, 8))
# MAGIC ax.barh(PlayersChart["PlayerName"],PlayersChart["Mean"], color = ["Red","Red","Red","Red","Red","Red","Red","Red","Red","Red","Green","Green","Green","Green","Green","Green","Green","Green","Green","Green"])
# MAGIC 
# MAGIC #Design
# MAGIC plt.autoscale(enable=True, axis='y', tight=False)
# MAGIC plt.xlabel("Average Round Score (to Par)")
# MAGIC plt.ylabel("Player")
# MAGIC plt.title("Top and Bottom 10 Players for Average Round Score")
# MAGIC plt.grid(axis='x')
# MAGIC 
# MAGIC display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Now lets looks at the Hole by Hole dataframe.
# MAGIC Barracuda Tournament has already been removed
# MAGIC 
# MAGIC There are also a few player statistics I want to calculate and store for later (Par3Average, Par4Average, Par5Average, HolesPerBirdie, HolesPerEagle)

# COMMAND ----------

holesDf.head(20)

# COMMAND ----------

holesDf.describe()

# COMMAND ----------

sum(holesDf["HoleInOne"])
# 29 hole in ones

# COMMAND ----------

holesDf["Hole_ScoreNum"].hist(bins = 7)

# COMMAND ----------

holeNumberScores = holesDf.groupby("Number").agg({"Hole_ScoreNum" : ["mean"]})
holeNumberScores.reset_index(inplace = True)
holeNumberScores.columns = ["HoleNumber","Mean"]
holeNumberScores["Mean_adj"] = holeNumberScores["Mean"]*10
holeNumberScores.sort_values("HoleNumber",inplace = True)
holeNumberScores = holeNumberScores.astype({"HoleNumber": str})

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC fig, ax = plt.subplots(figsize=(10, 7))
# MAGIC ax.scatter(holeNumberScores["HoleNumber"], holeNumberScores['Mean'])
# MAGIC 
# MAGIC #Design
# MAGIC plt.autoscale(enable=True, axis='y', tight=False)
# MAGIC plt.ylabel("Average Score to Par")
# MAGIC plt.xlabel("Hole Number")
# MAGIC plt.title("Average Score by Hole Number")
# MAGIC plt.grid(axis='y')
# MAGIC 
# MAGIC display(fig)

# COMMAND ----------

holeParScores = holesDf.groupby("Par").agg({"Hole_ScoreNum" : ["mean"]})
holeParScores.reset_index(inplace = True)
holeParScores.columns = ["HolePar","Mean"]
holeParScores["Mean_adj"] = holeParScores["Mean"]*10
holeParScores.sort_values("HolePar",inplace = True)

xdist = holeParScores["HolePar"].tolist()

holeParScores = holeParScores.astype({"HolePar": str})

labels = holeParScores["Mean"].tolist()
lables = list(map(str, labels))



# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC fig, ax = plt.subplots(figsize=(5, 7))
# MAGIC ax.bar(holeParScores["HolePar"], holeParScores['Mean'], color = ["Red","Gray","Green"])
# MAGIC 
# MAGIC #Design
# MAGIC plt.autoscale(enable=True, axis='y', tight=False)
# MAGIC plt.ylabel("Average Score (to Par)")
# MAGIC plt.xlabel("Par")
# MAGIC plt.title("Average Score by Hole Par")
# MAGIC plt.grid(axis='y')
# MAGIC ## Labels
# MAGIC #for i in range(len(labels)):
# MAGIC   #plt.text(x = xdist[i] - 3.5,y=holeParScores["Mean"][i] +0.01, s = lables[i], size = 6)
# MAGIC 
# MAGIC display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC **DISTANCE MATRICIES**
# MAGIC 
# MAGIC I want to create distance matricies for both players and courses
# MAGIC 
# MAGIC The conclusions I hope to draw through this are: *Which courses play similarly*, *Which Players perform similarly*, *Which Players play similarly*
# MAGIC 
# MAGIC We will attack each of these questions one at a time

# COMMAND ----------

# Courses
# we are going to do 4 total distances. euclidean and cosine with both filling with zero and mean
CourseMatrix = roundsDf.groupby(["PlayerID","TournamentID"]).agg({"RoundShots": "sum"})
CourseMatrix.reset_index(inplace = True)
CourseMatrix = CourseMatrix.pivot(columns="TournamentID",index = "PlayerID",values="RoundShots").fillna(0)
CoursesList = CourseMatrix.columns

# COMMAND ----------

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.cluster.hierarchy import dendrogram, linkage

# COMMAND ----------

dist = pd.DataFrame(euclidean_distances(CourseMatrix.transpose()))
dist.columns = CoursesList
dist.index = CoursesList

# COMMAND ----------

z = linkage(CourseMatrix.transpose(), metric = 'euclidean')
dendrogram(z, leaf_rotation = 90, color_threshold = 1250, labels= dist.index)
plt.show()

# COMMAND ----------

z = linkage(CourseMatrix.transpose(), metric = 'cosine')
dendrogram(z, leaf_rotation = 90, color_threshold = 0.2, labels= dist.index, orientation = 'right')
plt.show()

# COMMAND ----------

# now lets try will fill 
CourseMatrixMean = roundsDf.groupby(["PlayerID","TournamentID"]).agg({"RoundShots": "sum"})
CourseMatrixMean.reset_index(inplace = True)
CourseMatrixMean = CourseMatrixMean.pivot(columns="TournamentID",index = "PlayerID",values="RoundShots")
CourseMatrixMean.fillna(CourseMatrixMean.mean(), inplace = True)

# COMMAND ----------

z = linkage(CourseMatrixMean.transpose(), metric = 'euclidean')
dendrogram(z, leaf_rotation = 90, color_threshold = 300, labels= dist.index)
plt.show()

# COMMAND ----------

z = linkage(CourseMatrixMean.transpose(), metric = 'cosine')

plt.figure()
dendrogram(z, leaf_rotation = 90, color_threshold = 0.01, labels= dist.index)
plt.xlabel("Tournament ID")
plt.show()

# COMMAND ----------

SimilarTournaments = Tournaments.loc[(Tournaments["TournamentID"] == 452) | (Tournaments["TournamentID"] == 410 )| (Tournaments["TournamentID"] == 420 )| (Tournaments["TournamentID"] == 447)]
SimilarTournaments["Name"]

# the 4 tournaments that are grouped the closest together are   BMW Championship, Olympic Men's Golf Competition, Sentry Tournament of Champions, The ZOZO CHAMPIONSHIP

# COMMAND ----------

playerCounts = roundsDf.groupby("TournamentID").agg({"PlayerID":"nunique"})
#playerCounts
playerCounts.head()

# COMMAND ----------

# Its grouping together those tournaments that dont have cuts. Hense the values are all closer and they are being compared as matching tournaments. Lets try and remove players that missed the cut

CourseMatrixCuts = roundsDf.groupby(["PlayerID","TournamentID"]).agg({"RoundShots": "sum"})
CourseMatrixCuts.reset_index(inplace = True)
CourseMatrixCuts = CourseMatrixCuts.loc[CourseMatrixCuts["RoundShots"] >= 230]

CourseMatrixCuts = CourseMatrixCuts.pivot(columns="TournamentID",index = "PlayerID",values="RoundShots")
CourseMatrixCuts.fillna(CourseMatrixCuts.mean(), inplace = True)

# COMMAND ----------

z = linkage(CourseMatrixCuts.transpose(), metric = 'cosine')

plt.figure(figsize = [10,10])
dendrogram(z, leaf_rotation = 90, color_threshold = 0.00004, labels= dist.index)
plt.xlabel("Tournament ID")
plt.show()

# COMMAND ----------

z = linkage(CourseMatrixCuts.transpose(), metric = 'euclidean')

plt.figure(figsize = [10,10])
#dendrogram(z, leaf_rotation = 90, color_threshold = 50, labels= dist.index)
dendrogram(z, leaf_rotation = 90, color_threshold = 45, labels= dist.index)
plt.xlabel("Tournament ID")
plt.ylim(37,68)
plt.show()

#now we get some real groups

# COMMAND ----------

# Player performance

PlayerPMatrix = roundsDf.groupby(["PLAYER NAME","TournamentID"]).agg({"RoundShots": "sum"})
PlayerPMatrix.reset_index(inplace = True)
PlayerPMatrix = PlayerPMatrix.loc[PlayerPMatrix["RoundShots"] >= 230]

PlayerPMatrix = PlayerPMatrix.pivot(columns="PLAYER NAME",index = "TournamentID",values="RoundShots")
PlayerPMatrix.fillna(PlayerPMatrix.mean(), inplace = True)

# COMMAND ----------

#I split this into 3 to highlight the close relationships
pLabels = PlayerPMatrix.columns
z = linkage(PlayerPMatrix.transpose(), metric = 'euclidean')

plt.figure(figsize = [10,10])
#dendrogram(z, leaf_rotation = 90, color_threshold = 50, labels= dist.index)
dendrogram(z, leaf_rotation = 90, color_threshold = 15, labels= pLabels)
plt.xlabel("Player ID")
#plt.ylim(8, 17.)
#plt.xlim(1650,1850)
plt.show()

# COMMAND ----------

pLabels = PlayerPMatrix.columns
z = linkage(PlayerPMatrix.transpose(), metric = 'euclidean')

plt.figure(figsize = [10,10])
#dendrogram(z, leaf_rotation = 90, color_threshold = 50, labels= dist.index)
dendrogram(z, leaf_rotation = 90, color_threshold = 15, labels= pLabels)
plt.xlabel("Player Name")
plt.ylabel("Euclidian Distance")
plt.title("Player Euclidian Distances")
plt.ylim(8, 17.)
plt.xlim(1650,1850)
plt.xticks(fontsize = 10, rotation = 45, ha = "right")
plt.show()

# COMMAND ----------

plt.figure(figsize = [10,10])
#dendrogram(z, leaf_rotation = 90, color_threshold = 50, labels= dist.index)
dendrogram(z, leaf_rotation = 90, color_threshold = 15, labels= pLabels)
plt.xlabel("Player ID")
plt.ylim(2.5, 6)
plt.xlim(1200,1225)
plt.show()

# COMMAND ----------

pLabels = PlayerPMatrix.columns
z = linkage(PlayerPMatrix.transpose(), metric = 'cosine')

plt.figure(figsize = [10,10])
#dendrogram(z, leaf_rotation = 90, color_threshold = 50, labels= dist.index)
dendrogram(z, leaf_rotation = 90, color_threshold = 0.00001, labels= pLabels)
plt.xlabel("Player ID")
plt.ylim(0, 0.0000175)
plt.xlim(1675,1830)
plt.show()

# COMMAND ----------

# Player Statistics
playerStats = pd.read_csv("/dbfs/FileStore/karbide/PlayerStats.txt")
playerStats.drop(["Unnamed: 0"], axis = 1, inplace = True)

# COMMAND ----------

playerStats.columns

# COMMAND ----------

playerStats.describe()

# COMMAND ----------

# Im going to create 2 visuals for each stat category
# Greens in Regulation

playerStats["GIR_PCT_OVERALL"].hist(bins = 20)

# COMMAND ----------

playerStatsBunker = playerStats[["PLAYER NAME", "GIR_PCT_FAIRWAY_BUNKER"]]
playerStatsBunker.sort_values(["GIR_PCT_FAIRWAY_BUNKER"], inplace = True)
playerStatsBunkerTop = playerStatsBunker[0:10]
playerStatsBunkerBottom = playerStatsBunker[-10:]

playerStatsBunker = pd.concat([playerStatsBunkerTop,playerStatsBunkerBottom])

# COMMAND ----------

#best and worst performers out of the bunker

%matplotlib inline
fig, ax = plt.subplots(figsize=(5, 8))
ax.barh(playerStatsBunker["PLAYER NAME"],playerStatsBunker["GIR_PCT_FAIRWAY_BUNKER"], color = ["Red","Red","Red","Red","Red","Red","Red","Red","Red","Red","Green","Green","Green","Green","Green","Green","Green","Green","Green","Green"])

#Design
plt.autoscale(enable=True, axis='y', tight=False)
plt.xlabel("Greens in Regulation % (fairway bunker)")
plt.ylabel("Player")
plt.title("Top and Bottom 10 Players from Fairway Bunker")
plt.grid(axis='x')

display(fig)

# COMMAND ----------

import seaborn as sns

GIRSTATS = playerStats[['GIR_PCT_FAIRWAY_BUNKER', 'GIR_PCT_FAIRWAY',
       'GIR_PCT_OVERALL', 'GIR_PCT_OVER_100', 'GIR_PCT_OVER_200',
       'GIR_PCT_UNDER_100', 'GREEN_PCT_SCRAMBLE_SAND',
       'GREEN_PCT_SCRAMBLE_ROUGH']]

GIRcorr = GIRSTATS.corr()
GIRcorr

# COMMAND ----------

sns.heatmap(abs(GIRcorr),
           xticklabels = GIRcorr.columns,
           yticklabels = GIRcorr.columns,
           annot = True,
           cmap = "jet")

# COMMAND ----------

# Tee Shots
playerStats["TEE_AVG_DRIVING_DISTANCE"].hist(bins = 15)

# COMMAND ----------

playerStatsTee = playerStats[["PLAYER NAME", "TEE_AVG_BALL_SPEED"]]
playerStatsTee.sort_values(["TEE_AVG_BALL_SPEED"], inplace = True)
playerStatsTeeTop = playerStatsTee[0:10]
playerStatsTeeBottom = playerStatsTee[-10:]

playerStatsTee = pd.concat([playerStatsTeeTop,playerStatsTeeBottom])


# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC fig, ax = plt.subplots(figsize=(5, 8))
# MAGIC ax.barh(playerStatsTee["PLAYER NAME"],playerStatsTee["TEE_AVG_BALL_SPEED"], color = ["Red","Red","Red","Red","Red","Red","Red","Red","Red","Red","Green","Green","Green","Green","Green","Green","Green","Green","Green","Green"])
# MAGIC 
# MAGIC #Design
# MAGIC plt.autoscale(enable=True, axis='y', tight=False)
# MAGIC plt.xlabel("Ball Velocity (mph)")
# MAGIC plt.ylabel("Player")
# MAGIC plt.title("Top and Bottom 10 Players Tee Ball Velocity")
# MAGIC plt.grid(axis='x')
# MAGIC 
# MAGIC display(fig)

# COMMAND ----------

# Are the tee off stats coorelated

playerTeeStats = playerStats[['TEE_AVG_BALL_SPEED',
       'TEE_AVG_DRIVING_DISTANCE', 'TEE_DRIVING_ACCURACY_PCT',
       'TEE_AVG_LAUNCH_ANGLE', 'TEE_AVG_LEFT_ROUGH_TENDENCY_PCT',
       'TEE_AVG_RIGHT_ROUGH_TENDENCY_PCT', 'TEE_AVG_SPIN_RATE']]

# COMMAND ----------

Teecorr = playerTeeStats.corr()
abs(Teecorr)

# COMMAND ----------

sns.heatmap(abs(Teecorr),
           xticklabels = Teecorr.columns,
           yticklabels = Teecorr.columns,
           annot = True,
           cmap = "jet")

# COMMAND ----------

# Putting

plt.figure(figsize = (8,6))
plt.hist(playerStats["PUTTING_AVG_ONE_PUTTS"], label = "One Putts/round")
plt.hist(playerStats["PUTTING_AVG_TWO_PUTTS"], label = "Two Putts/round")

plt.xlabel("Putts (per Round)")
plt.ylabel("Frequency")
plt.title("Comparing # of One and Two Putts")
plt.legend(loc = "upper right")

# COMMAND ----------

playerStatsPutt = playerStats[["PLAYER NAME", "PUTTING_AVG_PUTTS"]]
playerStatsPutt.sort_values(["PUTTING_AVG_PUTTS"], inplace = True)
playerStatsPuttTop = playerStatsPutt[0:10]
playerStatsPuttBottom = playerStatsPutt[-10:]

playerStatsPutt = pd.concat([playerStatsPuttTop,playerStatsPuttBottom])


# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC fig, ax = plt.subplots(figsize=(5, 8))
# MAGIC ax.barh(playerStatsPutt["PLAYER NAME"],playerStatsPutt["PUTTING_AVG_PUTTS"], color = ["Green","Green","Green","Green","Green","Green","Green","Green","Green","Green","Red","Red","Red","Red","Red","Red","Red","Red","Red","Red"])
# MAGIC 
# MAGIC #Design
# MAGIC plt.autoscale(enable=True, axis='y', tight=False)
# MAGIC plt.xlabel("Putts (per Round)")
# MAGIC plt.ylabel("Player")
# MAGIC plt.title("Top and Bottom 10 Players Putts per Round")
# MAGIC plt.grid(axis='x')
# MAGIC 
# MAGIC display(fig)

# COMMAND ----------

playerPuttStats = playerStats[[ 'PUTTING_AVG_ONE_PUTTS', 'PUTTING_AVG_TWO_PUTTS', 'PUTTING_AVG_PUTTS',
       'PUTTING_AVG_DIST_BIRDIE_INCH']]

# COMMAND ----------

Puttcorr = playerPuttStats.corr()
abs(Puttcorr)

# COMMAND ----------

sns.heatmap(abs(Puttcorr),
           xticklabels = Puttcorr.columns,
           yticklabels = Puttcorr.columns,
           annot = True,
           cmap = "jet")

# COMMAND ----------

# MAGIC %md
# MAGIC # Make my own stats!
# MAGIC 
# MAGIC **I want to make 5 stats using the hole dataframe:**
# MAGIC 
# MAGIC *Average Par 3,4,and 5 Scores, Holes/Birdie, and Holes/Bogey*

# COMMAND ----------

#Average scores
AverageScores = holesDf.groupby(["Player_ID","Par"]).agg({"Hole_ScoreNum" : "mean"})
AverageScores.reset_index(inplace = True)
AverageScorePar3 = AverageScores.loc[AverageScores["Par"] == 3]
AverageScorePar4 = AverageScores.loc[AverageScores["Par"] == 4]
AverageScorePar5 = AverageScores.loc[AverageScores["Par"] == 5]

AverageScorePar3.columns = ["PlayerID","Par","Par3Average"]
AverageScorePar4.columns = ["PlayerID","Par","Par4Average"]
AverageScorePar5.columns = ["PlayerID","Par","Par5Average"]

AverageScorePar3.drop(["Par"], axis = 1, inplace = True)
AverageScorePar4.drop(["Par"], axis = 1, inplace = True)
AverageScorePar5.drop(["Par"], axis = 1, inplace = True)

# COMMAND ----------

playerStats = playerStats.merge(AverageScorePar3, how = "left", on = "PlayerID")
playerStats = playerStats.merge(AverageScorePar4, how = "left", on = "PlayerID")
playerStats = playerStats.merge(AverageScorePar5, how = "left", on = "PlayerID")

# COMMAND ----------

def holesPerResult(data,result):
  results = data.groupby("Player_ID").agg({result: ["sum", 'count']})
  results.reset_index(inplace = True)
  results.columns = ["PlayerID","Sum","Total"]
  results[f"HolesPer{result}"] = results["Total"] / results["Sum"]
  
  results.drop(["Sum","Total"], axis = 1, inplace = True)
  return(results)

# COMMAND ----------

h_birdie = holesPerResult(holesDf,"Birdie")
h_bogey = holesPerResult(holesDf,"Bogey")

# COMMAND ----------

playerStats = playerStats.merge(h_birdie, how = "left", on = "PlayerID")
playerStats = playerStats.merge(h_bogey, how = "left", on = "PlayerID")

# COMMAND ----------


plt.figure(figsize = (8,6))
plt.hist(playerStats["HolesPerBirdie"], alpha = 0.5, label = "Birdie", bins = 20)
plt.hist(playerStats["HolesPerBogey"], alpha = 0.5, label = "Bogey", bins = 20)

plt.xlabel("Holes per Result")
plt.ylabel("Frequency")
plt.title("Holes Per Birdie vs Holes Per Bogey")
plt.legend(loc = "upper right")

# COMMAND ----------

resultStats = playerStats[["Par3Average","Par4Average","Par5Average","HolesPerBirdie","HolesPerBogey"]]

# COMMAND ----------

Resultscorr = resultStats.corr()
abs(Resultscorr)

# COMMAND ----------

sns.heatmap(abs(Resultscorr),
           xticklabels = Resultscorr.columns,
           yticklabels = Resultscorr.columns,
           annot = True,
           cmap = "jet")

# COMMAND ----------

#playerStats.to_csv("/dbfs/FileStore/karbide/PlayerStatsComplete.txt")

# COMMAND ----------

playerStats.columns

# COMMAND ----------

# MAGIC %md
# MAGIC **Strokes Gained**
# MAGIC 
# MAGIC Strokes gained is the most popular statistic for predicting golf results

# COMMAND ----------

StrokesGained = pd.read_csv("/dbfs/FileStore/karbide/StrokesGainedIDs.txt")
StrokesGained.drop(["Unnamed: 0"], axis =1, inplace = True)

# COMMAND ----------

StrokesGained.corr()

# COMMAND ----------


plt.figure(figsize = (8,6))
plt.hist(StrokesGained["AVERAGE"], bins = 18)

plt.xlabel("SG")
plt.ylabel("Frequency")
plt.title("Average Stroked Gained (per Shot)")
#plt.legend(loc = "upper right")

# COMMAND ----------

SP2 = playerStats[['GREEN_PCT_SCRAMBLE_SAND','GREEN_PCT_SCRAMBLE_ROUGH', 'PlayerID']]

SP1 = SP2.merge(StrokesGained, how = "inner", on = "PlayerID")

SP1 = SP1.sort_values("TOTAL SG:T")
SP1["Inv_RankTee"] = [x for x in range(len(SP1["PlayerID"]))]
SP1 = SP1.sort_values("TOTAL SG:T2G")
SP1["Inv_RankFairway"] = [x for x in range(len(SP1["PlayerID"]))]
SP1 = SP1.sort_values("TOTAL SG:P")
SP1["Inv_RankPutt"] = [x for x in range(len(SP1["PlayerID"]))]
SP1 = SP1.sort_values('GREEN_PCT_SCRAMBLE_SAND')
SP1["Inv_RankSand"] = [x for x in range(len(SP1["PlayerID"]))]
SP1 = SP1.sort_values('GREEN_PCT_SCRAMBLE_ROUGH')
SP1["Inv_RankRough"] = [x for x in range(len(SP1["PlayerID"]))]



# COMMAND ----------

def rankedStat(c,name,data):
  data[name] = (data[c]/len(data[c]))*100
  return(data)

# COMMAND ----------

SP1 = rankedStat("Inv_RankTee","Driving (SG)",SP1)
SP1 = rankedStat("Inv_RankFairway","Fairway (SG)",SP1)
SP1 = rankedStat("Inv_RankPutt","Putting (SG)",SP1)
SP1 = rankedStat("Inv_RankRough","Rough (Green Scramble %)",SP1)
SP1 = rankedStat("Inv_RankSand","Bunker (Green Scramble %)",SP1)

# COMMAND ----------

SP3 = SP1[["Driving (SG)","Fairway (SG)","Putting (SG)","Rough (Green Scramble %)","Bunker (Green Scramble %)","PLAYER NAME"]]

# COMMAND ----------

from math import pi

# COMMAND ----------

SP3

# COMMAND ----------

values=SP3.loc[17].drop('PLAYER NAME').values.flatten().tolist()
values += values[:1]

values2 =SP3.loc[130].drop('PLAYER NAME').values.flatten().tolist()
values2 += values[:1]

angles = [n / float(5) * 2 * pi for n in range(5)]
angles += angles[:1]

categories = ["Driving","Fairway","Putting","Rough","Bunker"]

# COMMAND ----------

ax = plt.subplot(111, polar=True)

plt.xticks(angles[:-1], categories , color='grey', size=8)

ax.set_rlabel_position(0)
plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
plt.ylim(0,100)
 

ax.plot(angles, values, linewidth=1, linestyle='solid', label = "Brooks Koepka", color = "Green")
ax.fill(angles, values, 'g', alpha=0.1)

ax.plot(angles, values2, linewidth=1, linestyle='solid', label="Rickie Fowler", color = "Orange")
ax.fill(angles, values2, 'r', alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

font1 = {'family':'serif','color':'grey','size':7}

plt.title("Player Comparison")
plt.xlabel("*Putting, Driving, and Fairway from Strokes Gained. Rough and Bunker from Green Scramble % ", fontdict = font1)

plt.show()

# COMMAND ----------

