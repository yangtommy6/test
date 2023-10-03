# Lab 2 - Pipelines and Gradient Descent

### Objective
The purpose of this lab is to get more practice with Scikit-learn, particularly with Scikit-learn pipelines, to review linear regression, and to gain a deeper understanding of gradient descent. 



### Directions
* *Accept the Assignment*: When you accept the assignment in GitHub Classroom, a repository named `lab-02` will be automatically generated for you under the "s486Fall2023" organization.
* *Locate Your Repository*: Your personal repository for this homework can be found at `https://github.com/s486Fall2023/lab-02-your_user_name`.
* *Clone the Repository*: 
    - Open your terminal and navigate to the directory where you want to download the repository.
    - Run `git clone [your repository URL]` to clone the repository onto your local machine.
* *Working Directory*: The cloning process will create a new directory named `lab-02-your_user_name`. Ensure that you perform all your work inside this directory.
* *Commit Your Progress*: As you work on the assignment, remember to commit your changes periodically. You can easily do this using Visual Studio Code (VS Code).
* *Remote Connection*: If you've cloned the repository correctly and are working within the created directory, the remote link to your GitHub repository should already be configured.
* *File and Function Names*: 
    - As you work out the problems and experiment with code, use either `work.py` or `work.ipynb`
    - For your final answers and code, use the file `lab2.py`
    - Try to follow the guidelines in the comments 
    - **Do not** change the names of the functions or the file.
    - Do not include any superfluous code
    - Make sure that your code runs without error before submission
* *Submitting on Gradescope*: 
    - Once you've completed the assignment, go to Gradescope and select your personal homework repository (`https://github.com/s486Fall2023/lab-03-your_user_name`) as the source for your submission.
    - Make sure the repository contains the completed `lab2.py` file, the `final_model.joblib`, and any work you did in `work.py` or `work.ipynb`.
    

## Gradient Descent

For this part, use the `abalone.csv` data found in [this](https://github.com/esnt/Data/tree/main/CleanData) repository.  The goal is the estimate the age of an abalone using length, diameter, height, and weight.  

Matrix algebra tips:
* Add an intercept column to the X matrix by using `PolynomialFeatures`:
  ```
  from sklearn.preprocessing import PolynomialFeatures 
  poly = PolynomialFeatures(degree=1)
  X = poly.fit_transform(X)
  ```
* For the matrix multiplication to be compatible, also reshape the target:
  (This code assumes that y starts a pandas Series)
  ```
  y = y.values.reshape(n,1) 
  ```
* Matrix multiplication is done with the `@` operator
* `np.linalg.inv()` can invert a matrix
* The `.T` attribute is a matrix's transpose (i.e., `X.T`)

### 1. Linear Regression
* Write a function that uses Scikit-learn to fit a linear regression (function should be in lab2-functions.py)
  - The function should take as input an X matrix (or DataFrame) and a y vector (or Series)
  - The function should return the estimates of the betas


### 2. Normal Equations (function should be in lab2-functions.py)
* Write a python function that uses the normal equations to estimate the betas
  - The function should take as input an X matrix and y vector
  - The function should return the estimates of the betas


### 3. Gradient Descent (function should be in lab2-functions.py)
* Write a python function that uses gradient descent to compute the betas for a linear regression.
  - The function should take as input an X matrix, y vector, learning rate (alpha), and number of epochs
  - The function should return the estimates of the betas 


### 4. Stochastic Gradient Descent (function should be in lab2-functions.py)
* Write a python function that uses **stochastic** gradient descent to compute the betas for a linear regression.
  - The function should take as input an X matrix, y vector, learning rate (alpha), and number of epochs
  - The function should return the estimates of the betas 
  - (optional) Include a `batch_size` argument so that your function can also perform mini-batch gradient descent

### 5. Compare the Betas
* Use each of your functions on the abalone data to estimate the linear regression parameter for prediction age from the rest of the variables
  - Be sure the standardize the data first.  The non-optimized gradient descent algorithms that we are writing have a hard time if the data is not standardized
  - For gradient descent, alpha=0.15 with 300 epochs worked well for me.  Play around with different values. If your function outputs NaN values, your learning rate is probably too high
  - For stochastic gradient descent, alpha=0.05 with 500 epochs (of n iterations) worked well for me.  
* Make a table (similar to the one below) of the estimates of the betas for each method.  Round to 4 digits.  The estimates show be pretty similar, though you'll find that stochastic gradient descent can be fairly different
* (The table should be a pandas DataFrame, not a markdown table, for reproducibility) 
  
|               | LR     | NEq    | GD     | SGD    |
|---------------|--------|--------|--------|--------|
| beta0         |        |        |        |        |
| beta1         |        |        |        |        |    
| beta2         |        |        |        |        |            
| beta3         |        |        |        |        |
| beta4         |        |        |        |        |            


## Pipelines 
This part uses the ocean temperature data.  This data is a subset of the [oceanographic data](https://calcofi.org/data/oceanographic-data/bottle-database/) from [The California Cooperative Oceanic Fisheries Investigations](https://calcofi.org/)
You can find the data in [this](https://github.com/esnt/Data/tree/main/OceanicFisheries) repository.
The goal is to predict the water temperature (T_degC) 

## 6. Prepare the data
* Make a training and a test set using `train_test_split`.  Use `random_state=307`and the default test size (25% of the data)

## 7. Pipelines Practice 
* For this part, only use the variables 'Salnty','O2ml_L', 'O2Sat',  and 'Depthm' to predict 'T_degC'
* Use the `Pipeline` function from `sklearn.pipeline` to build a pipeline that does the following:
      * Imputes missing values with `SimpleImputer`
      * Adds 2rd order polynomial functions of the Xs 
      * Standardizes the data
      * Fits a KNN regression model (using 10 neighbors)
* Fit your pipeline to the training data 
* Report the training and the test MSE
* Is the model (pipeline) able to predict well or not?  Explain your answer. 

## 8. Pipelines and Hyperparameter Tuning 
* For this part, continue to only use the variables 'Salnty','O2ml_L', 'O2Sat',  and 'Depthm' to predict 'T_degC'
* Use `GridSearchCV` to use 10-fold cross validation and scoring metric negative mse to find the optimal hyperparameters  from among the following options:
  - `strategy="mean"` vs `strategy="median"` with `SimpleImputer`
  - Polynomial terms of order 1, 2, and 3
  - Number of neighbors 5 to 100 by 5
  - KNN neighbor weights `uniform` vs `distance`
* Print the best hyperparameter combinations and the best score
* Refit the pipeline using the best hyperparameters 
* Report the training and the test MSE 
* Is this model (pipeline) any better at predicting water temperature than the model in problem (7)?

## 9. Save your best model
* Save your best model as a `joblib` file called "final_model.joblib"
  ```
  from joblib import dump
  filename = 'final_model.joblib'
  dump(your_trained_model, filename)
  ```
  where `your_trained_model` is the pipeline you trained at the end of problem 8
* *Be sure to include the model file in the GitHub repo*

## 10. Advanced Pipelines

* For this part, continue use 'Salnty','O2ml_L', 'O2Sat',  and 'Depthm', but also include "Wea", "Cloud_Typ", and "Cloud_Amt". 
* Use the `Pipeline` function from `sklearn.pipeline` to build a pipeline that does the following:
  - For Numeric variables:
    * Imputes missing values using the mean with `SimpleImputer`
    * Adds 2rd order polynomial functions of the Xs 
    * Standardizes the data
  - For Categorical variable:
    * Impute missing values using the mode with `SimpleImputer`
    * Appropriately encodes (with OneHot or Ordinal encoding)
    * Fits a **Linear Regression** model to the processed data
* Fit the pipeline to the training data
* Report the training and test mse
* Do the extra variables improve predictability? 
