import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import shap


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

# Read in data from a folder
database_china = pd.read_excel("database_china.xlsx").drop(['Vad', 'Fcad'], axis=1)
database_USA = pd.read_excel("database_USA.xlsx").drop(['Vad', 'Fcad'], axis=1)
data_lit = pd.read_excel("coal_sample_lit.xlsx").drop(['Vad', 'Fcad'], axis=1)
data_exp = pd.read_excel("coal_sample_exp.xlsx").drop(['Vad', 'Fcad'], axis=1)

# Split the dataset into a training set and a test set
train_data, test_data = train_test_split(database_china, test_size=0.2, random_state=0)

# Load training dataset
train_data_set = TabularDataset(train_data)
label = 'Qb,ad'
y_train = train_data_set[label]
train_data_nolab = train_data_set.drop(columns=[label])
feature_names = train_data_nolab.columns

# Model Training
save_path = 'agModels-predictClass_best_china'
predictor = TabularPredictor(label=label, eval_metric='mean_squared_error', path=save_path).fit(train_data_set,
                                                                                                # num_bag_folds=5,
                                                                                                # num_bag_sets=1,
                                                                                                # num_stack_levels=1
                                                                                                presets='best_quality')

# Load test dataset
test_data_set = TabularDataset(test_data)
y_test = test_data_set[label]
test_data_nolab = test_data_set.drop(columns=[label])


# Model Evaluation
# predictor = TabularPredictor.load(save_path)  # Load the previously trained model
perf = predictor.evaluate(test_data_set)
y_train_pred = predictor.predict(train_data_nolab)
y_test_pred = predictor.predict(test_data_nolab)


# Model prediction results
model_results_test = predictor.leaderboard(test_data_set,
                                           extra_metrics=['r2', 'mae', 'mape', 'mse', 'rmse'],
                                           silent=True)
model_results_train = predictor.leaderboard(train_data_set,
                                            extra_metrics=['r2', 'mae', 'mape', 'mse', 'rmse'],
                                            silent=True)

# Feature Importance
feature_importance = predictor.feature_importance(test_data_set)

# Save results
pre_results_train = pd.DataFrame()
pre_results_train['y_train'] = y_train
pre_results_train['y_train_pred'] = y_train_pred
pre_results_train.to_csv(os.path.join(save_path, 'pre_results_train.csv'))

pre_results_test = pd.DataFrame()
pre_results_test['y_test'] = y_test
pre_results_test['y_test_pred'] = y_test_pred
pre_results_test.to_csv(os.path.join(save_path, 'pre_results_test.csv'))

model_results_test.to_csv(os.path.join(save_path, 'model_results_test.csv'))
model_results_train.to_csv(os.path.join(save_path, 'model_results_train.csv'))

feature_importance.to_csv(os.path.join(save_path, 'Feature_importance.csv'))


# SHAP
shap.initjs()

class AutogluonWrapper:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names

    def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict_proba(X, as_multiclass=False)

ag_wrapper = AutogluonWrapper(predictor, feature_names)
explainer = shap.KernelExplainer(ag_wrapper.predict_binary_prob, train_data.drop(['Qb,ad'], axis=1))
shap_values = explainer.shap_values(test_data.drop(['Qb,ad'], axis=1))
shap_mean = abs(pd.DataFrame(shap_values)).mean()

# Plot SHAP value
shap.summary_plot(shap_values, test_data.drop(['Qb,ad'], axis=1))
shap.summary_plot(shap_values, test_data.drop(['Qb,ad'], axis=1), plot_type="bar")
plt.show()

# Save SHAP value
pd.DataFrame(shap_values).to_csv('AutoGluon_shap_values.csv', index=False, header=['Cad', 'Had', 'Nad', 'St,ad', 'Mad', 'Aad', 'Vad'])
pd.DataFrame(shap_mean).to_csv('AutoGluon_mean_shap_values.csv', index=False, header=['Cad', 'Had', 'Nad', 'St,ad', 'Mad', 'Aad', 'Vad'])
print('y_base:', explainer.expected_value)








