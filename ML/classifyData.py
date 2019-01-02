# First thing we need to do is make our data useable

# We will convert the JSON structured as so: 
# {
#     "home": "Man City",
#     "away": "Everton",
#     "goals_home": "3",
#     "goals_away": "1",
#     "relevancy": 1
# },

# Into useable vectors
# we will have vectors representing games, they will look like so:
# [home team]     eg. [14] -- each team will be mapped to a number
# [away team]         [11]
# [relevancy]         [1]

# The games matrix will be a 10292 x 3 matrix, with each row representing a game

# the result vector will look like so:
# [home win ]     eg. [1]
# [draw     ]         [0]
# [home loss]         [0]

# The results matrix will originally be stored as 10292 x 3, with the vector above representing a row, 
# However it will be modified heavily as we implement One vs All logistic regression

import numpy as np
import tensorflow as tf 
import json
import random

def classify_team(team):
  teams_dict = {
    'Man City' : 1,
    'Everton' : 2,
    'Wolves' : 3,
    'Bournemouth' : 4,
    'Watford' : 5,
    'Cardiff' : 6,
    'Spurs' : 7,
    'Burnley' : 8,
    'Huddersfield' : 9,
    'Newcastle' : 10,
    'Crystal Palace' : 11,
    'Leicester' : 12,
    'Liverpool' : 13,
    'West Ham' : 14,
    'Man Utd' : 15,
    'Fulham' : 16,
    'Southampton' : 17,
    'Brighton' : 18,
    'Arsenal' : 19,
    'Chelsea' : 20,
    'Swansea' : 21,
    'Stoke' : 22,
    'West Brom' : 23,
    'Middlesbrough' : 24,
    'Hull' : 25,
    'Sunderland' : 26,
    'Norwich' : 27,
    'Aston Villa' : 28,
    'QPR' : 29,
    'Wigan' : 30,
    'Reading' : 31,
    'Bolton' : 32,
    'Blackburn' : 33,
    'Birmingham' : 34,
    'Blackpool' : 35,
    'Portsmouth' : 36,
    'Derby' : 37,
    'Sheffield Utd' : 38,
    'Charlton' : 39,
    'Leeds' : 40,
    'Ipswich' : 41,
    'Coventry' : 42,
    'Bradford' : 43,
    'Wimbledon' : 44,
    'Sheffield Wed' : 45,
    'Nott\'m Forest' : 46,
    'Barnsley' : 47,
    'Swindon' : 48,
    'Oldham' : 49 
  }
  return teams_dict[team]

np_game_matrix = np.empty([10292,3])
np_results_matrix = np.zeros([10292,3])
games_matrix = tf.get_variable("games_matrix", shape=[10292,3], dtype = tf.float64, initializer = tf.zeros_initializer)
results_matrix = tf.get_variable("results_matrix", shape=[10292,3], dtype = tf.float64, initializer = tf.zeros_initializer)
training_games_matrix = tf.get_variable("training_games_matrix", shape=[8000,3], dtype = tf.float64, initializer = tf.zeros_initializer)
training_results_matrix = tf.get_variable("training_results_matrix", shape=[8000,3], dtype = tf.float64, initializer = tf.zeros_initializer)
test_games_matrix = tf.get_variable("test_games_matrix", shape=[2292,3], dtype = tf.float64, initializer = tf.zeros_initializer)
test_results_matrix = tf.get_variable("test_results_matrix", shape=[2292,3], dtype = tf.float64, initializer = tf.zeros_initializer)

with open('../DataSets/AllPLGames.json') as f:
  data = json.load(f)

random.shuffle(data) # ensure randomness in data order

for i in range(len(data)):
  np_game_matrix[i][0] = classify_team(data[i]['home'])
  np_game_matrix[i][1] = classify_team(data[i]['away'])
  np_game_matrix[i][2] = data[i]['relevancy']
  goal_differential = int(data[i]['goals_home']) - int(data[i]['goals_away'])
  if goal_differential > 0:
    np_results_matrix[i][0] = 1
  elif goal_differential == 0:
    np_results_matrix[i][1] = 1
  else:
    np_results_matrix[i][2] = 1

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  sess.run(games_matrix.assign(np_game_matrix))
  sess.run(results_matrix.assign(np_results_matrix))
  sess.run(training_games_matrix.assign(np_game_matrix[0:8000]))
  sess.run(training_results_matrix.assign(np_results_matrix[0:8000]))
  sess.run(test_games_matrix.assign(np_game_matrix[8000:]))
  sess.run(test_results_matrix.assign(np_results_matrix[8000:]))
  # Save the variables to disk.
  save_path = saver.save(sess, "../models/dataMatrices.ckpt")
  print("Model saved in path: %s" % save_path)