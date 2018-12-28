# SoccerResultPredictor

Predicting sports can be very difficult due to the amount of factors involved in each individual game.

This application is made to try out several methods to try and predict the results of upcoming EPL soccer games.

## Logistic Regression
The first, and most basic method will be a logistic regression model using past results. 
using `DataScraping/scrapeHistoricalResults.js` we scraped all 10292 EPL games (from 1992-2018)
and stored them in `DataSets/allPLGames.json`

using `ML/classifyData.py` we stored the data in Tensors, so that we can utilize Tensorflow to perform 
the actual logictic regression

in `ML/trainLogisticRegressionModel.py` we train a model using the gradient descent algorithm, in order to get our model
`theta`, which is a very simple 4x1 vector