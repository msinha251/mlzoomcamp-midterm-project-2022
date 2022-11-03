from ml.model import train_xgboost, train_random_forest
import pandas as pd
from ml.model import save_model

# load data
df = pd.read_csv('./data/train.csv')

# train model
#model, dv = train_xgboost(df, target='Transported')
model, dv = train_random_forest(df, target='Transported')

# save model
#save_model(model, dv, '../models/xgb_model.pkl')
save_model(model, dv, './models/rf_model.pkl')


