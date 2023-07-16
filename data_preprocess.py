import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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


# 0.load data
data = pd.read_xlsx('database_raw.xlsx')
print("shape of data：\n", data.shape)

# 1.Dataset pre-processing
# 1.1 Missing value processing
data = data.replace(0.00, np.nan)
data = data.dropna()

print("The shape of the data set after removing missing values：\n", data.shape)
print("Distribution of database metrics：\n", data.describe().round(2))
data.to_excel("database_china.xlsx")
data.describe().to_csv("Distribution of database metrics.csv")

# 2 Data Visualization
# 2.1 View data distribution in a box line chart
def plot_boxplot_dataset(dataframe):
    plt.figure()
    plt.boxplot(x=dataframe,
                labels=['Qb,ad', 'Cad', 'Had', 'Oad', 'Nad', 'St,ad', 'Mad', 'Aad', 'Vad', 'Fcad'],
                whis=None,
                patch_artist=True,
                showmeans=True,
                widths=0.4,
                boxprops={'color': 'black', 'facecolor': 'thistle'},
                flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},
                meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},
                medianprops={'linestyle': '--', 'color': 'orange'})
    plt.ylim(-10, 100)
    plt.xlabel('Feature')
    plt.ylabel('Feature value')
    plt.savefig('results/data_boxplot.png')
    plt.show()

plot_boxplot_dataset(data)

# 2.2 View data distribution in histogram
def plot_hist_dataset(dataframe):
    dataframe.hist(bins=50)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.6)
    plt.ylim(0, 40)
    plt.savefig('results/data_hist.png')
    plt.show()

# plot_hist_dataset(data)

# 2.3 Normality test
def plot_norm(dataframe):
    # print(stats.normaltest(dataframe[:, 1]))

    # Calculate the mean and variance
    mean = dataframe.iloc[:, 0].mean()
    std = dataframe.iloc[:, 0].std()

    def normpdf(x, mu, sigma):
        """
        Normal distribution function
        :param x:
        :param mu: Mean value
        :param sigma: Standard deviation
        :return: pdf: Probability density function
        """
        pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        return pdf

    # Plot the normal distribution
    x = np.arange(mean - 3 * std, mean + 3 * std, 0.01)
    y = normpdf(x, mean, std)
    fig = plt.figure()
    plt.plot(x, y, 'g--', linewidth=2)
    plt.hist(dataframe.iloc[:, 0], bins=25, color='blue', alpha=0.5, rwidth=0.9, density=True)
    plt.title('Qb,ad Normal Distribution: μ = {:.2f}, σ={:.2f}'.format(mean, std))
    plt.xlabel('Qb,ad score')
    plt.ylabel('Probability density function')
    plt.xlim(0, 35)
    plt.ylim(0, 0.12)
    plt.show()


# plot_norm(data)

# 2.4 Correlation analysis - scatter plot + linear fit
def plot_scatter_lr(dataframe):
    x_list = dataframe.columns.values[1:].reshape(-1, 1).tolist()
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.98, hspace=0.3)  # 调整子图布局
    ax = axes.ravel()
    for i in range(len(x_list)):
        for x_label in x_list[i]:
            x, y = dataframe[x_label], dataframe[dataframe.columns.values[0]]
            x = x.values.reshape(-1, 1)
            y = y.values.reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(x, y)
            y_pre = lr.predict(x)
            R2 = r2_score(y, y_pre)
            ax[i].scatter(x, y, s=10)
            ax[i].plot(x, y_pre, c='red')
            ax[i].set_xlabel(x_label)
            ax[i].set_ylabel(dataframe.columns.values[0])
            # ax[i].set_xlim(0, max(x))
            ax[i].set_ylim(0, 40)
            ax[i].text(0.70*max(x), 33, '$R^2$={:.4f}'.format(R2), fontsize=12, color='r')
    plt.savefig('results/data_Scatter_lr.png')
    plt.show()


plot_scatter_lr(data)

# 2.5 Correlation Analysis - Correlation Heat Map
def plot_heatmap(dataframe):
    data_corr = dataframe.corr()  # Pearson Correlation Analysis
    plt.figure()
    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.98)
    cmap = sns.diverging_palette(260, 10, as_cmap=True)
    sns.heatmap(round(data_corr, 2), annot=True, vmin=-0.6, vmax=0.6, center=0, square=True, cmap=cmap)  # 绘制矩阵热力图
    plt.savefig('results/data_heatmap.png')
    plt.show()

# plot_heatmap(data)

# 3.Split the database into a training set and a test set
y = data[data.columns.values[0]]
x = data[data.columns.values[1:]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.16, random_state=0)

# Plot the distribution of the training and test sets
plt.figure(figsize=(12, 8))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4, wspace=0.4)
for i, col in enumerate(x_test.columns):
    ax = plt.subplot(3, 3, i+1)
    ax = sns.kdeplot(x_train[col], color="Red", fill=True)
    ax = sns.kdeplot(x_test[col], color="Blue", fill=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])
plt.savefig('results/data_kdeplot.png')
plt.show()

