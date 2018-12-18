# SoccerResultPredictor

Predicting sports can be very difficult due to the amount of factors involved in each individual game.

This application is made to try out several methods to try and predict the results of upcoming EPL soccer games.

## Logistic Regression
The first, and most basic method will be a logistic regression model using past results. 
using `Data Scraping/scrapeHistoricalResults.js` we scraped all 10292 EPL games (from 1992-2018)
and stored them in `Data Sets/allPremierLeagueGames.json`
