import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

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


# Define Model
def model_linearregression():
    lr = LinearRegression()
    return lr


def model_Ridge(alpha):
    ridge = Ridge(alpha)
    return ridge


def model_SVR_rbf(c, gamma):
    svr_rbf = SVR(kernel='rbf', C=c, gamma=gamma)
    return svr_rbf


def model_tree(max_depth, min_sample_leaf):
    tree = DecisionTreeRegressor(max_depth=max_depth,
                                 min_samples_leaf=min_sample_leaf,
                                 random_state=0
                                 )
    return tree


def model_forest(max_depth, n_estimators, max_features, min_samples_split):
    forest = RandomForestRegressor(max_depth=max_depth,
                                   n_estimators=n_estimators,
                                   max_features=max_features,
                                   min_samples_split=min_samples_split,
                                   random_state=0
                                   )
    return forest


def model_gbdt(learning_rate, n_estimators, max_depth, min_impurity_decrease, subsample):
    gbdt = GradientBoostingRegressor(learning_rate=learning_rate,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     min_impurity_decrease=min_impurity_decrease,
                                     subsample=subsample,
                                     random_state=0
                                     )
    return gbdt

def model_xgb(learning_rate, n_estimators, max_depth,
              min_child_weight, subsample, colsample_bytree, gamma
              ):
    xgb = XGBRegressor(learning_rate=learning_rate,
                       n_estimators=n_estimators,
                       max_depth=max_depth,
                       min_child_weight=min_child_weight,
                       subsample=subsample,
                       colsample_bytree=colsample_bytree,
                       gamma=gamma,
                       random_state=0
                       )
    return xgb

# Hyperparametric optimization method
def model_GridSearchCV(model, param_grid, cv):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(x_train_scale, y_train)
    print("Test set score: {:.3f}".format(grid_search.score(x_test_scale, y_test)))
    print("Best parameters: {}".format(grid_search.best_params_))
    print("Best cross-validation score: {:.3f}".format(grid_search.best_score_))
    print("Best estimator:\n{}".format(grid_search.best_estimator_))
    cv_results = pd.DataFrame(grid_search.cv_results_)
    pd.set_option("display.max_columns", None)
    print(cv_results)

# Model evaluation
def model_score(model):
    print('Result of {}:'.format(model))
    y_pre_train = model.fit(x_train_scale, y_train).predict(x_train_scale)
    # pd.DataFrame(y_pre_train).to_csv('{}_y_pre_train.csv'.format(model))
    r2_train = r2_score(y_train, y_pre_train)
    adj_r2_train = 1 - ((1 - r2_train) * (len(y_train) - 1) / (len(y_train) - x_train_scale.shape[1] - 1))
    mae_train = mean_absolute_error(y_train, y_pre_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pre_train)
    mse_train = mean_squared_error(y_train, y_pre_train)
    rmse_train = np.sqrt(mse_train)
    print('Train_R^2={:.3f}'.format(r2_train))
    print('Train_Adj_R^2={:.3f}'.format(adj_r2_train))
    print('Train_MAE={:.3f}'.format(mae_train))
    print('Train_MAPE={:.5f}'.format(mape_train))
    print('Train_MSE={:.3f}'.format(mse_train))
    print('Train_RMSE={:.3f}'.format(rmse_train))
    print('*********************************************')
    y_pre = model.fit(x_train_scale, y_train).predict(x_test_scale)
    # pd.DataFrame(y_pre).to_csv('{}_y_pre_test.csv'.format(model))
    r2_test = r2_score(y_test, y_pre)
    adj_r2_test = 1 - ((1 - r2_test) * (len(y_test) - 1) / (len(y_test) - x_test_scale.shape[1] - 1))
    mae_test = mean_absolute_error(y_test, y_pre)
    mape_test = mean_absolute_percentage_error(y_test, y_pre)
    mse_test = mean_squared_error(y_test, y_pre)
    rmse_test = np.sqrt(mse_test)
    print('Test_R^2={:.3f}'.format(r2_test))
    print('Test_Adj_R^2={:.3f}'.format(adj_r2_test))
    print('Test_MAE={:.3f}'.format(mae_test))
    print('Test_MAPE={:.5f}'.format(mape_test))
    print('Test_MSE={:.3f}'.format(mse_test))
    print('Test_RMSE={:.3f}'.format(rmse_test))
    print('---------------------------------------------')


def plot_model_score(model):
    y_train_pre = model.fit(x_train_scale, y_train).predict(x_train_scale)
    y_test_pre = model.fit(x_train_scale, y_train).predict(x_test_scale)
    plt.figure(figsize=(5, 4))
    plt.scatter(y_train, y_train_pre, s=20, marker='o', label='Train data')
    plt.scatter(y_test, y_test_pre, s=20, marker='v', label='Test data')
    plt.axline([10, 10], slope=1, ls='-', c='black', alpha=0.5, label='Zero-Error line')
    plt.legend(loc='upper left')
    plt.xlabel('Measured calorific value')
    plt.ylabel('Predicted calorific value')
    plt.xlim(10, 40)
    plt.ylim(10, 40)
    plt.show()


if __name__ == '__main__':
    # Load data
    data = pd.read_excel('database_china.xlsx')
    print(data.shape)

    # Split the database into a training set and a test set
    y = data[data.columns.values[0]]
    x = data.drop(['Qb,ad', 'Oad', 'Fcad'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # StandardScaler
    transfer = StandardScaler()
    x_train_scale = transfer.fit_transform(x_train)
    x_test_scale = transfer.transform(x_test)

    # Model
    lr = model_linearregression()
    model_score(lr)
    plot_model_score(lr)

    # ridge = model_Ridge(alpha=1.5)
    # model_score(ridge)
    # plot_model_score(ridge)
    # param_grid = {'alpha': [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]}
    # model_GridSearchCV(ridge, param_grid, 10)

    # svr_rbf = model_SVR_rbf(c=100, gamma=0.01)
    # model_score(svr_rbf)
    # plot_model_score(svr_rbf)
    # param_grid = {'C': [200, 100, 10, 1, 0.1, 0.01],
    #               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    # model_GridSearchCV(svr_rbf, param_grid=param_grid, cv=10)

    # tree = model_tree(max_depth=7, min_sample_leaf=12)
    # model_score(tree)
    # plot_model_score(tree)
    # param_grid = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    #               # 'min_samples_split': range(2, 21, 1),
    #               'min_samples_leaf': range(1, 21, 2)}
    # model_GridSearchCV(tree, param_grid=param_grid, cv=10)

    # forest = model_forest(max_depth=10, n_estimators=364, max_features=6, min_samples_split=2)
    # model_score(forest)
    # plot_model_score(forest)
    #
    # gbdt = model_gbdt(learning_rate=0.105,
    #                   n_estimators=165,
    #                   max_depth=6,
    #                   # max_features=6,
    #                   min_impurity_decrease=1.75,
    #                   subsample=0.195)
    # model_score(gbdt)
    # plot_model_score(gbdt)

    # xgb = model_xgb(learning_rate=0.04,
    #                 n_estimators=819,
    #                 max_depth=9,
    #                 min_child_weight=1,
    #                 subsample=0.61,
    #                 colsample_bytree=0.94,
    #                 gamma=0.4
    #                 )
    # model_score(xgb)
    # plot_model_score(xgb)

