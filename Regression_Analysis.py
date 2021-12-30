# Databricks notebook source
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC #REGRESSION MODEL NOTES
# MAGIC ## We Can Conduct a few different version of this regression model by changing the dependent and independent variables
# MAGIC **Dependent Variable**
# MAGIC We can elect to make the dependent variable round score or tournament score
# MAGIC 
# MAGIC *Round Score* - would give us more data points, but could also cause higher variation
# MAGIC 
# MAGIC *Tournament Score* - would seem to be the better fit, but we may not have enough data points
# MAGIC 
# MAGIC The dependent variable can also refer to tournament score across all tournaments, or for a specific tournament
# MAGIC 
# MAGIC **Independent Variables**
# MAGIC 
# MAGIC 4 major groups of Independent Variables
# MAGIC 
# MAGIC *Greens In Regulation* : Describes how frequently the player makes in to the green at least 2 strokes away from par based on a number of situation. Evaluates a players skill in the fairways/middle game
# MAGIC 
# MAGIC     Consists of ['GIR_PCT_FAIRWAY_BUNKER', 'GIR_PCT_FAIRWAY', 'GIR_PCT_OVERALL', 'GIR_PCT_OVER_100', 'GIR_PCT_OVER_200','GIR_PCT_UNDER_100', 'GREEN_PCT_SCRAMBLE_SAND', 'GREEN_PCT_SCRAMBLE_ROUGH']
# MAGIC      
# MAGIC *Tee Box*: Describes different elements of a players driving/tee shots. Evaluates a players skill off the tee/long game
# MAGIC 
# MAGIC     Consists of ['TEE_AVG_BALL_SPEED', 'TEE_AVG_DRIVING_DISTANCE', 'TEE_DRIVING_ACCURACY_PCT', 'TEE_AVG_LAUNCH_ANGLE', 'TEE_AVG_LEFT_ROUGH_TENDENCY_PCT', 'TEE_AVG_RIGHT_ROUGH_TENDENCY_PCT', 'TEE_AVG_SPIN_RATE']
# MAGIC     
# MAGIC *Putting*: Describes a players performance on the green. Evaluates a players putting skill/short game
# MAGIC 
# MAGIC     Consists of ['PUTTING_AVG_ONE_PUTTS', 'PUTTING_AVG_TWO_PUTTS','PUTTING_AVG_PUTTS','PUTTING_AVG_DIST_BIRDIE_INCH']
# MAGIC     
# MAGIC *Performance Based*: Descibes a players performance in terms of previous results and scores. Evaluates a players consistency and past performances
# MAGIC 
# MAGIC     Consists of ['Par3Average','Par4Average', 'Par5Average', 'HolesPerBirdie', 'HolesPerBogey','FINISHES_TOP10']
# MAGIC     
# MAGIC **Independence Between Variables**
# MAGIC 
# MAGIC To avoid creating bias in the regression model, we should avoid using the following highly coorelated independent variables together in the same model
# MAGIC 
# MAGIC *GIR*: (GIR_PCT_OVERALL: GIR_PCT_OVER_100, GIR_PCT_FAIRWAY)
# MAGIC 
# MAGIC *Tee*: (TEE_AVG_BALL_SPEED: TEE_AVG_DRIVING_DISTANCE)
# MAGIC 
# MAGIC *Putting*: (PUTTING_AVG_ONE_PUTTS: PUTTING_AVG_TWO_PUTTS : PUTTING_AVG_PUTTS)
# MAGIC 
# MAGIC *Performance Based*: (Par4Average: HolesPerBogey)

# COMMAND ----------

# Lets Start with the Dependent Variable as Round Score across all tournaments

roundsDf = pd.read_csv("/dbfs/FileStore/karbide/RoundsReg.txt")

playerStats = pd.read_csv("/dbfs/FileStore/karbide/PlayerStatsComplete.txt")

roundsDf.drop(["Unnamed: 0"], axis = 1, inplace = True)
playerStats.drop(["Unnamed: 0"], axis = 1, inplace = True)

# COMMAND ----------

roundScores = roundsDf[["PlayerID","RoundScore"]]

# COMMAND ----------

roundsReg = roundScores.merge(playerStats, how = "left", on = "PlayerID")

# COMMAND ----------

roundsReg.corr()
# none or the variables are highly coorelated with RoundScore but the performance based ones score the highest

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score

# COMMAND ----------

#selecting Independet Variables (X)
X = roundsReg[["Par4Average","HolesPerBirdie","PUTTING_AVG_DIST_BIRDIE_INCH","PUTTING_AVG_PUTTS","TEE_AVG_DRIVING_DISTANCE","TEE_DRIVING_ACCURACY_PCT", "FINISHES_TOP10", "GIR_PCT_OVERALL", "GIR_PCT_FAIRWAY_BUNKER"]]
Y = roundsReg[["RoundScore"]]

# COMMAND ----------

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33)

# COMMAND ----------

reg = linear_model.LinearRegression()
reg.fit(pd.DataFrame(X_train),pd.DataFrame(Y_train))

# COMMAND ----------

pred = reg.predict(X_test)
err = pd.Series(Y_test["RoundScore"]) - [p[0]for p in pred]

# COMMAND ----------

display(err.hist(bins=100))
# seems we get some really crazy predictions

# COMMAND ----------

predDf = pd.DataFrame(pred)
predDf.describe()

# COMMAND ----------

Y_test.describe()

# COMMAND ----------

reg.score(pd.DataFrame(X_train), pd.DataFrame(Y_train))

# COMMAND ----------

#This shows the high variance I was worried about, Lets check accuracy
r2_score(Y_test["RoundScore"],pred)

# COMMAND ----------

import statistics as stats
def rmse(errors):
  return(pow(stats.mean([pow(e,2) for e in errors]),0.5))

# COMMAND ----------

rmse(err)

# COMMAND ----------

# seems we are way off, lets change the dependent variable to tournament score

# COMMAND ----------

tournamentScore = roundsDf.groupby(["PlayerID","TournamentID"]).agg({"RoundScore":"sum"})
tournamentScore.reset_index(inplace = True)
#since we doing this across all tournaments, we can drop tournament ID
tournamentScore.drop(["TournamentID"],inplace = True, axis = 1)

# COMMAND ----------

t_Reg = tournamentScore.merge(playerStats, how = "left", on = "PlayerID")

# COMMAND ----------

t_Reg.corr()
# our coorelation are still getting stronger, but still there is little that is very strongly coorelated

# COMMAND ----------

X = t_Reg[["Par4Average","HolesPerBirdie","PUTTING_AVG_DIST_BIRDIE_INCH","PUTTING_AVG_PUTTS","TEE_AVG_DRIVING_DISTANCE","TEE_DRIVING_ACCURACY_PCT", "FINISHES_TOP10", "GIR_PCT_OVERALL", "GIR_PCT_FAIRWAY_BUNKER"]]
Y = t_Reg[["RoundScore"]]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

reg = linear_model.LinearRegression()
reg.fit(pd.DataFrame(X_train),pd.DataFrame(Y_train))

# COMMAND ----------

pred = reg.predict(X_test)
err = pd.Series(Y_test["RoundScore"]) - [p[0]for p in pred]

# COMMAND ----------

display(err.hist(bins=100))

# COMMAND ----------

predDf = pd.DataFrame(pred)
print(predDf.describe())
print(Y_test.describe())


# COMMAND ----------

print ("R2 Train")
print(reg.score(pd.DataFrame(X_train), pd.DataFrame(Y_train)))
print("R2 Test")
print(r2_score(Y_test["RoundScore"],pred))
print("RMSE")
print(rmse(err))

# COMMAND ----------

def linearReg(ind,dep,split):
  X_train, X_test, Y_train, Y_test = train_test_split(ind,dep, test_size = split)
  
  
  reg = linear_model.LinearRegression()
  reg.fit(pd.DataFrame(X_train),pd.DataFrame(Y_train))
  
  pred = reg.predict(X_test)
  err = pd.Series(Y_test["RoundScore"]) - [p[0]for p in pred]
  
  
  print ("R2 Train")
  print(reg.score(pd.DataFrame(X_train), pd.DataFrame(Y_train)))
  print("R2 Test")
  print(r2_score(Y_test["RoundScore"],pred))
  print("RMSE")
  print(rmse(err))
  return(reg.coef_,reg.intercept_)
  
  

# COMMAND ----------

# to make this easier lets make a function
X = t_Reg[["Par4Average","HolesPerBirdie","PUTTING_AVG_PUTTS","TEE_AVG_DRIVING_DISTANCE", "FINISHES_TOP10", "GIR_PCT_OVERALL", "Par5Average", "Par3Average"]]
Y = t_Reg[["RoundScore"]]

c2, i2 = linearReg(X,Y,0.2)

# COMMAND ----------

#we cant use different tournament in the dependant variable as is but its possible we can use the mean standardized version

tournamentScoreNorm = roundsDf.groupby(["PlayerID","TournamentID"]).agg({"RoundScore":"sum"})
tournamentScoreNorm.reset_index(inplace = True)
#pull the mean scores for each tournament
meanScores = tournamentScoreNorm.groupby("TournamentID").agg({"RoundScore":"mean"})
meanScores.reset_index(inplace=True)
meanScores.columns = ["TournamentID","Mean"]

tournamentScoreNorm = tournamentScoreNorm.merge(meanScores, how="left", on="TournamentID")

tournamentScoreNorm["NormScore"] = tournamentScoreNorm["RoundScore"] - tournamentScoreNorm["Mean"]
tournamentScoreNorm.drop(["TournamentID","Mean","RoundScore"], axis =1, inplace = True)
tournamentScoreNorm.columns = ["PlayerID","RoundScore"]

# COMMAND ----------

t_regNorm = tournamentScoreNorm.merge(playerStats, how ="left", on = "PlayerID")

# COMMAND ----------

t_regNorm.corr()

# COMMAND ----------

X = t_regNorm[["Par4Average","HolesPerBirdie","PUTTING_AVG_DIST_BIRDIE_INCH", "PUTTING_AVG_ONE_PUTTS", "TEE_AVG_DRIVING_DISTANCE", "TEE_AVG_LEFT_ROUGH_TENDENCY_PCT", "FINISHES_TOP10", "GREEN_PCT_SCRAMBLE_SAND", "Par5Average", "Par3Average"]]
Y = t_regNorm[["RoundScore"]]

c1, i1 = linearReg(X,Y,0.2)
#normalizing improves our R2 value

# COMMAND ----------

# again our scores are pretty bad, but maybe they'll get better if we look at just one tournament
tournamentScore2 = roundsDf.groupby(["PlayerID","TournamentID"]).agg({"RoundScore":"sum"})
tournamentScore2.reset_index(inplace = True)


# COMMAND ----------

#lets pick the tournament where we have the most players
roundsDf.groupby("TournamentID").agg({"PlayerID":"nunique"}).sort_values("PlayerID", ascending = False).head(2)
# Tournament 429, the players championship

# COMMAND ----------

tournamentScore2 =tournamentScore2.loc[tournamentScore2["TournamentID"] == 429]
tournamentScore2.drop(["TournamentID"], axis = 1, inplace = True)

# COMMAND ----------

t2_reg = tournamentScore2.merge(playerStats, how = "left", on = "PlayerID")

# COMMAND ----------

t2_reg.corr()
#now we have some much stronger coorelations, lets try to use them

# COMMAND ----------

X = t2_reg[["Par4Average","HolesPerBirdie","PUTTING_AVG_DIST_BIRDIE_INCH", "PUTTING_AVG_ONE_PUTTS", "TEE_AVG_DRIVING_DISTANCE", "TEE_AVG_LEFT_ROUGH_TENDENCY_PCT", "FINISHES_TOP10", "GREEN_PCT_SCRAMBLE_SAND", "Par5Average", "Par3Average"]]
Y = t2_reg[["RoundScore"]]

c3,i3 = linearReg(X,Y,0.2)

#now our R2 is much higher, but can we still do better?

# COMMAND ----------

#mean standardize for one tournament
mean429 = meanScores.loc[meanScores["TournamentID"]==429]
mean429 = mean429["Mean"][22]
mean429

# COMMAND ----------

Y = t2_reg[["RoundScore"]]
Y["RoundScoreNorm"] = Y["RoundScore"] - mean429
Y = Y[["RoundScoreNorm"]]
Y.columns = ["RoundScore"]

# COMMAND ----------

c4,i4 = linearReg(X,Y,0.2)
#our result is about the same

# COMMAND ----------

# lets just play w the ivs for a bit
X = t2_reg[["Par4Average","HolesPerBirdie", "Par5Average", "Par3Average", "HolesPerBogey","GREEN_PCT_SCRAMBLE_SAND"]]
Y = t2_reg[["RoundScore"]]

c5,i5 = linearReg(X,Y,0.2)

# COMMAND ----------

# lets just play w the ivs for a bit
X = t2_reg[["Par4Average","HolesPerBirdie", "Par5Average", "Par3Average", "HolesPerBogey","GIR_PCT_OVERALL"]]
Y = t2_reg[["RoundScore"]]

c5,i5 = linearReg(X,Y,0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC Since our most accurate model is specfic tournament results, lets add the 3 Strokes Gained categories

# COMMAND ----------

StrokesGained = pd.read_csv("/dbfs/FileStore/karbide/StrokesGainedIDs.txt")
StrokesGained.drop(["Unnamed: 0"], axis = 1, inplace = True)

# COMMAND ----------

t2_regSG = t2_reg.merge(StrokesGained, how = "inner", on = "PlayerID")

# COMMAND ----------

t2_regSG.columns

# COMMAND ----------

# now with SG
X = t2_regSG[["Par4Average","HolesPerBirdie", "Par5Average", "Par3Average", "HolesPerBogey","GREEN_PCT_SCRAMBLE_SAND", 'TOTAL SG:T', "TOTAL SG:T2G", 'TOTAL SG:P']]
Y = t2_regSG[["RoundScore"]]

c6,i6 = linearReg(X,Y,0.2)

# COMMAND ----------

X = t2_regSG[["Par4Average","HolesPerBirdie", "Par5Average", "Par3Average", "HolesPerBogey","GREEN_PCT_SCRAMBLE_SAND", 'TOTAL SG:T', "TOTAL SG:T2G", 'TOTAL SG:P']]
Y = t2_regSG[["RoundScore"]]

c7,i7 = linearReg(X,Y,0.2)

# COMMAND ----------

X = t2_regSG[["GREEN_PCT_SCRAMBLE_SAND", 'TOTAL SG:T', "Par4Average","HolesPerBirdie"]]
Y = t2_regSG[["RoundScore"]]

c8,i8 = linearReg(X,Y,0.2)
print(c8,i8)

# COMMAND ----------

# MAGIC %md
# MAGIC # MODEL RESULTS AND COEFFICIENTS
# MAGIC **Tournament 429**
# MAGIC 
# MAGIC *Model 1*:
# MAGIC 
# MAGIC IVs -> ["Par4Average","HolesPerBirdie","PUTTING_AVG_DIST_BIRDIE_INCH", "PUTTING_AVG_ONE_PUTTS", "TEE_AVG_DRIVING_DISTANCE", "TEE_AVG_LEFT_ROUGH_TENDENCY_PCT", "FINISHES_TOP10", "GREEN_PCT_SCRAMBLE_SAND", "Par5Average", "Par3Average"]
# MAGIC 
# MAGIC DV -> Tournament Score
# MAGIC 
# MAGIC R2 and RMSE = 0.3229964494135511, 4.795109852419135
# MAGIC 
# MAGIC coef, int = [[ 2.32846057e+01 -2.26333978e-01  6.19652183e-02  1.83444680e-01 -2.53906041e-02 -2.46384591e-01 -1.97731802e-02 -1.40051312e-01 1.61535095e+01  1.39734513e+01]], 16.67094541
# MAGIC 
# MAGIC *Model 2*
# MAGIC 
# MAGIC IVs -> ["Par4Average","HolesPerBirdie","PUTTING_AVG_DIST_BIRDIE_INCH", "PUTTING_AVG_ONE_PUTTS", "TEE_AVG_DRIVING_DISTANCE", "TEE_AVG_LEFT_ROUGH_TENDENCY_PCT", "FINISHES_TOP10", "GREEN_PCT_SCRAMBLE_SAND", "Par5Average", "Par3Average"]
# MAGIC 
# MAGIC DV -> Tournament Score/Mean Standardized
# MAGIC 
# MAGIC R2 and RMSE = 0.27481894608420776, 5.278633537736203
# MAGIC 
# MAGIC coef, int =[[ 6.31478232e+00 -8.16209567e-02  7.18796229e-02 -4.30983221e-01 -8.41258947e-03 -2.12066433e-01  7.78575502e-02 -1.20083842e-01 2.55502279e+01  7.95291971e+00]], 15.26182535
# MAGIC 
# MAGIC *Model 3*
# MAGIC 
# MAGIC IVs -> ["Par4Average","HolesPerBirdie", "HolesPerBogey", "GREEN_PCT_SCRAMBLE_SAND", "Par5Average", "Par3Average"]
# MAGIC 
# MAGIC DV -> Tournament Score
# MAGIC 
# MAGIC R2 and RMSE = 0.3787100076428008, 4.605715520235553
# MAGIC 
# MAGIC coef, int = [[ 6.81122888  1.61102777 20.98822676 19.81415731  0.39739065 -0.09967628]], 2.13676949
# MAGIC 
# MAGIC *Model 4*
# MAGIC 
# MAGIC IVs -> [["Par4Average","HolesPerBirdie", "Par5Average", "Par3Average", "HolesPerBogey","GREEN_PCT_SCRAMBLE_SAND", 'TOTAL SG:T', "TOTAL SG:T2G", 'TOTAL SG:P']]
# MAGIC 
# MAGIC DV -> Tournament Score
# MAGIC 
# MAGIC R2 and RMSE = 0.5044001176837463,3.52737074470192
# MAGIC 
# MAGIC coef, int = [[-1.46756830e+01  7.27174253e-01  4.24465311e+00 -2.42897881e+00 6.32162133e-02 -1.04608027e-01 -6.62482431e+02  6.62435768e+02 6.62423818e+02]] [4.27898458]

# COMMAND ----------

print(c6)
print(i6)

# COMMAND ----------

#Lets try some feature importance to increase our model accuracy
t2_regSG.corr(method = 'pearson')

# COMMAND ----------

t2_regSG2 = t2_regSG.drop(["PlayerID"], axis = 1)
t2_regSG2.corr(method = 'pearson')

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler

# COMMAND ----------

X = t2_regSG[["Par4Average","HolesPerBirdie", "Par5Average", "Par3Average", "HolesPerBogey","GREEN_PCT_SCRAMBLE_SAND", 'TOTAL SG:T', "TOTAL SG:T2G", 'TOTAL SG:P','AVERAGE']]
Y = t2_regSG[["RoundScore"]]

scaler = MinMaxScaler()
scaler.fit(X)
sX = scaler.transform(X)
cols = X.columns
X_new = pd.DataFrame(sX,columns = cols)

c7,i7 = linearReg(X,Y,0.2)
print(c7,i7)
#coefficients sugguest we remove some of the variables

# COMMAND ----------

X = t2_regSG[["Par4Average","HolesPerBirdie", "Par5Average", "Par3Average", "HolesPerBogey",'AVERAGE']]
Y = t2_regSG[["RoundScore"]]

scaler = MinMaxScaler()
scaler.fit(X)
sX = scaler.transform(X)
cols = X.columns
X_new = pd.DataFrame(sX,columns = cols)

c8,i8 = linearReg(X,Y,0.2)
print(c8,i8)

# COMMAND ----------

X = t2_regSG[["Par4Average", "Par5Average", "Par3Average",'AVERAGE']]
Y = t2_regSG[["RoundScore"]]

scaler = MinMaxScaler()
scaler.fit(X)
sX = scaler.transform(X)
cols = X.columns
X_new = pd.DataFrame(sX,columns = cols)

c8,i8 = linearReg(X,Y,0.2)
print(c8,i8)

# COMMAND ----------

X = t2_regSG[['TOTAL SG:T', "TOTAL SG:T2G", 'TOTAL SG:P']]
Y = t2_regSG[["RoundScore"]]

scaler = MinMaxScaler()
scaler.fit(X)
sX = scaler.transform(X)
cols = X.columns
X_new = pd.DataFrame(sX,columns = cols)

c9,i9 = linearReg(X,Y,0.2)
print(c9,i9)