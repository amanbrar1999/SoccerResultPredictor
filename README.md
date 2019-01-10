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

After training, we find that our models are the following: 
    ```
    HomeWinsModel = tf.Variable(
    [[ 0.10498991],
        [-0.94905062],
        [ 0.74825934],
        [ 0.84990819]], dtype = tf.float64)
    DrawsModel = tf.Variable(
    [[ 0.33612894],
        [-0.14059028],
        [-0.06648189],
        [ 0.61908273]], dtype = tf.float64)
    HomeLossModel = tf.Variable(
    [[-1.77296921],
        [-0.17580398],
        [ 0.09614677],
        [ 0.26889068]], dtype = tf.float64)
    ```

Due to the nature of logistic regression, they are very simple 4x1 tensors. Now we proceed to find how accurate our models are. 

In `ML/analyzeLogisticRegression.py` we calculate the accuracy using TensorFlow's metrics library, and find the following: 
```
accuracy on training set: 0.5955833
```
```
accuracy on test set: 0.59947646
```

These results mean that ultimately, the accuracy on both our training and test sets ended up being around 60%. It is very interesting how the 
accuracy on the test set is actually somehow higher than the training set. 

There are plenty of takeaways from this approach to predicting, the first and most obvious being that a simple logistic regression with my data is NOT a good way to make a model. Knowing how unpredictable sports are, having a simple 4x1 vector as a model seems far too simple for solving a problem of this magnitude, hence I will try other models to see how well they do compared to this one. 

Furthermore, I am quite aware that the data I have collected is probably lacklusture for this magnitude of problem, as there are many things apart from the scores and dates of previous games which can help accurately describe what happened in each game. I will try to continually scrape more types of data, such as things like the possession stats for each team, the number of passes, amount of distance covered, and many many more. 

## TODO
 - Implement an approach with a neural network model
 - Implement an approach with SVM
 - Make it user friendly, perhaps by connecting it to a webapp (rather than the current CLI)