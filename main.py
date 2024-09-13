import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib as plt

df = pd.read_csv('train.csv')

df.dropna()

# creating heatmap
corr_plot = sns.heatmap(df.corr())
plt.title("Titanic Data Correlation Matrix")

