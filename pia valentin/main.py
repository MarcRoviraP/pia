import pandas as pd

data = pd.Series(range(9),index=["A","B","C","D","E","F","G","H","I"])
print(data.iloc[::-1])