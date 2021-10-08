import pandas as pd
from solution import predict
base_path = './data'

df = pd.read_csv('./data/dataset.csv')
print(predict(df).shape)