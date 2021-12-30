# PGA Tour Analysis

This project takes players statistics and PGA result data to perform an exploratory analysis and create a prediction model.

The code sheets in this repository represent the following steps:

**API PULL AND DATA WRANGLING**
- Player Statistics are Imported from a kaggle dataset and formatted
- Player Tournament Results are pulled from SportData.io API, result are given by hole
- By hole results are transformed to create round summaries for each round played
- Edge cases like incomplete rounds, missed tournaments, or missed cuts are identified
- Player names are organized between dataframes through a primary key

**EXPLORATORY ANALYSIS**
- Created Exploratory Charts based on the By Hole Data
- Created Exploratory Charts based on the By Round Data
- Calculated Average Par 3, 4, and 5 score for each player
- Created Player Comparison Visuals based on both results (Dendrogram) and statistics (Spider Plot)

**PREDICTION MODEL**
- An Attempt to predict results based on player statistics
- Created models with the following dependent variables: Round Score (all tournaments), Tournament Score (all tournaments), Tournament Score (specific tournament)
- Specific tournaments were predicted most accuratley using regression, how ever lack of data hurts the model

Future Steps for the Project Include:

- Web Scraping to collect a more data from one tournament
- Acquiring more player statistics to increase the field
- Fitting new models to the new data
