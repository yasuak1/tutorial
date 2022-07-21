import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#df = pd.read_csv("iris.csv")
df = sns.load_dataset("iris")  #手元にiris.csvがない場合

sns.distplot(df.sepal_length,kde = True)
plt.show()