import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import KFold, cross_validate
from bayes_opt import BayesianOptimization
import time

# Global drawing settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


# Load data
data = pd.read_excel("database_china.xlsx")
print(data.shape)

# Split the database into a training set and a test set
y = data[data.columns.values[0]]
x = data.drop(['Qb,ad', 'Oad', 'Fcad'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# StandardScaler
transfer = StandardScaler()
x_train_scale = transfer.fit_transform(x_train)
x_test_scale = transfer.transform(x_test)

X = x_train_scale
Y = y_train

# Define the objective function
def bayesopt_object(learning_rate, n_estimators, max_depth, gamma, min_child_weight,
                       subsample, colsample_bytree, reg_alpha, reg_lambda):
    xgb = XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        gamma=gamma,
        min_child_weight=int(min_child_weight),
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=0,
        # verbose=False,
        n_jobs=60)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    validation_loss = cross_validate(xgb, X, Y,
                                     scoring="neg_mean_squared_error",
                                     cv=cv,
                                     verbose=False,
                                     n_jobs=60,
                                     error_score="raise"
                                     )
    return np.mean(validation_loss["test_score"])

# Define the parameter space (note that it is a tuple)
param_grid_simple = {
    'learning_rate': (0.01, 0.3),
    'n_estimators': (1, 1000),
    'max_depth': (3, 10),
    'gamma': (0, 0.5),
    'min_child_weight': (1, 8),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0, 0.5),
    'reg_lambda': (0, 0.5)
}

# Define the optimization objective function and specific process
def param_bayes_opt(param_grid_simple, init_points, n_iter):

    # Instantiation Optimizer
    opt = BayesianOptimization(f=bayesopt_object,
                               pbounds=param_grid_simple,
                               random_state=0)

    # bayes_opt only supports maximization
    opt.maximize(init_points=init_points,  # Number of initial observations
                 n_iter=n_iter)  # Number of iterations

    # Obtain the best parameters and the best scores
    params_best = opt.max["params"]
    score_best = opt.max["target"]
    print("\n", "\n", "best params :", params_best)
    print("\n", "\n", "best cvscore :", score_best)

    return params_best, score_best

# Define the validation function
def bayes_opt_validation(params_best):
    xgb = XGBRegressor(
        learning_rate=params_best["learning_rate"],
        n_estimators=int(params_best["n_estimators"]),
        max_depth=int(params_best["max_depth"]),
        gamma=int(params_best["gamma"]),
        min_child_weight=params_best["min_child_weight"],
        subsample=params_best["subsample"],
        colsample_bytree=params_best["colsample_bytree"],
        reg_alpha=params_best["reg_alpha"],
        reg_lambda=params_best["reg_lambda"],
        random_state=0,
        # verbose=False,
        n_jobs=60
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    validation_loss = cross_validate(xgb, X, Y,
                                     scoring="neg_mean_squared_error",
                                     cv=cv,
                                     verbose=False,
                                     n_jobs=60,
                                     error_score="raise"
                                     )
    return np.mean(validation_loss["test_score"])

# Execute the optimization process
start = time.time()
# Run Bayesian
params_best, score_best = param_bayes_opt(param_grid_simple, 20, 280)
# Print results
print("Training time: %s mins" % ((time.time()-start)/60))
validation_score = bayes_opt_validation(params_best)
print("\n", "\n", "validation_score: ", validation_score)
