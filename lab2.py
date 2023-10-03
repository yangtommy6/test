## Import appropriate libraries and functions

## Part 1:  Gradient Descent

#1
def linreg(X, y):
    pass # your code here

#2
def normaleq(X, y):
    pass # your code here

#3
def gd(X, y, alpha, n_epochs):
    pass # your code here


#4
def sgd(X, y, alpha, n_epochs, batch_size=1): # batch_size optional
    pass # your code here


#5
# read in data

# process data

# run functions

# make and print beta comparison table


################################
## Part 2: Pipelines
#6
#Read in data and split into training and test sets



#7
# build pipeline
pipe = Pipeline()

# train pipeline
pipe.fit(Xtrain,ytrain)

# predict with pipeline
yhat_train = pipe.predict(Xtrain)
yhat = pipe.predict(Xtest)

# print the training and testing MSE

##############
# Answer in comments:
# Is the model (pipeline) able to predict well or not?
#
#
#
#

# 8
# tune hyperparameters
# Set up the possible option in a dictionary where the 
# key is `step_name__argument` and the value are the options
# that you are considering
# for example, assuming you used 'knn' as the name in the
# pipeline:  `knn__n_neighbors`: list(range(5,100,5))

pipe = # build new pipeline, remove hyperparameter being tested
params = {}
cv = GridSearchCV(pipe, param_grid=params, cv=, scoring=)
cv.fit(Xtrain, ytrain)

# Get the "best" model

# print the training and testing MSE

##############
# Answer in comments:
# Is this model (pipeline) any better at predicting water temperature 
# than the model in problem (7)?
#
#
#
#


#9
# Save your best model

from joblib import dump
filename = 'final_model.joblib'
dump(your_trained_model, filename)


#10
# Try an advanced pipeline
# See chapter 2 in HOML for help

# print the training and testing MSE

##############
# Answer in comments:
# Do the extra variables improve predictability?
#
#
#
#