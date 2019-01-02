# SoccerResultPredictor

Predicting sports can be very difficult due to the amount of factors involved in each individual game. The purpose of this project is to try out several machine learning methods to predict the results of upcoming EPL soccer games.

## Logistic Regression
The first, and most basic method will be a logistic regression model using past results. 
using `DataScraping/scrapeHistoricalResults.js` we scraped all 10292 EPL games (from 1992-2018)
and stored them in `DataSets/allPLGames.json`

using `ML/classifyData.py` we stored the data in Tensors, so that we can utilize Tensorflow to perform 
the actual logictic regression

in `ML/trainLogisticRegressionModel.py` we train a logistic regression model using the gradient descent algorithm

We utilize one vs all logistic regression in order to classify a certain game as either a "Home Win", "Draw", or "Home Loss". This means that we will create a model that represents each outcome, and we choose whichever one comes out with the highest probability of being true

## TODO
 - Implement an approach with a neural network model
 - Implement an approach with SVM
 - Make it user friendly, perhaps by connecting it to a webapp (rather than the current CLI)