# Weight-Height-ML-Experimentation-
Some concepts I've learnt in Scikit-Learn being put to practice on a simple weight-height dataset to build a regression model, from pipelining, to different types of models(from XGBoost to SVMs) to hyperparameter tuning(GridSearch and RandomisedSearch)

We first read the dataset using "read_csv" command and graphbthe dataset as a whole to try to understand the nature of the dataset

# Removing Outliers
We use IQR method to remove outliers in the dataset(We keep track of lower limit and upper limit for weight, which will be required later).

# Pipelines
We first execute a train test split 
Then, we create a preprocessing pipeline (with SimpleImputer and OneHotEncoder for categorical_cols) after identifying categorical (Gender) and numerical (Height) columns
We print the preprocessed X_train value to keep track of how pipeline affects the training data

# Models : 1)XGBoost

We first create an XGBRegressor model pipeline with hyperparameters set according to developer discretion and build the final pipeline.
We find predicted values and accuracy using error metrics (R2 and RMSE, commonly used in regression problems). This value is noted down

# 2) RandomForest
We then create a RandomnForestRegressor model with hyperparameters set according to developer discretion to compare accuracy to the earlier model, by fitting to the already designed pipeline to find accuracy values
As expected, XGBRegressor performs better than RandomForestRegressor here

# GridSearchCV
To improve accuracy, we will perform a grid search using GridSearchCV, for some possible hyperparameter values for the XGBRegressor model.
This process provides us with the best parameter combination (best_params_), accuracy and these hyperparameter values can be directly fitted to an XGBoost model(best_model). We see that this results in better accuracy than the first XGBRegressor model with guess-based user selected hyperparameters
Note : We see that GridSearchCV is limited to grid values, as seen in a later iteration where values which weren't provided in param_grid provided better values

# Support Vector Machines (SVM)

We will be using SVR particularly here, where we try to find a hyperplane which best fits our data in the feature space, within a 'margin'.
We create an SVR model with guess-based user defined hyperparameters and run in the already established pipeline(with preprocessing).
We observe that for this dataset, SVR proves to be more effective than XGBRegressor

# RandomSearchCV

We now perform a random search, which is a more effective method of hyperparameter tuning, involving random selection of hyperparameters across the params_grid to provide best hyperparameter results.

We run a randomised search multiple times for different values of SVR hyperparametrs in the param_grid, with results produced being kept track of
(hyperparameter values and corresponding error metrics for that model). We arrive at a final result after some itereations
Note : RandomisedSearchCV can be run further in this instance. I have stopped here as this exercise was only for a demonstration of certain concepts, further iterations of the process for different param_grid values can be further executed to further optimise the SVR Model

Note : further processes in the preprocessing section, like scaling, KNN or Iterative Imputing, advanced hyperparameter tuning for models(max_depth,sampling_method,etc. for XGBRegressor, and degree,coef0,etc. for SVR) can be performed if required, for further performance optimisation or for more complex datasets that might require such processes and configurations.
