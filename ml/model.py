import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from ml.data import basic_preprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def train_random_forest(df, target):
    '''
    Train model and return it
    '''
    from sklearn.model_selection import train_test_split

    # Clean data
    df, cat_cols_to_use, num_cols, target = basic_preprocess(df, train=True, target=target)

    # Final features
    # cat_cols_to_use = ['homeplanet', 'cabin', 'destination']
    final_cols = cat_cols_to_use + list(num_cols)

    # Split data into train and validation
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=1)

    # Split data into X and y
    X_train_df = df_train.drop(target, axis=1)
    y_train = df_train[target]

    X_val_df = df_val.drop(target, axis=1)
    y_val = df_val[target]

    # convert to dicts
    train_dicts = X_train_df[final_cols].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dicts)

    # transform dicts
    X_train = dv.transform(train_dicts)
    val_dicts = X_val_df[final_cols].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1, oob_score=True)
    model.fit(X_train, y_train)
    
    # Predict on validation data
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    y_pred = y_pred_prob > 0.5

    # Print metrics
    print_metrics(y_val, y_pred, y_pred_prob)
    
    return model, dv

def train_xgboost(df, target):
    '''
    Train model and return it
    '''
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    # Clean data
    df, cat_cols_to_use, num_cols, target = basic_preprocess(df, train=True, target=target)
    # cat_cols_to_use = ['homeplanet', 'destination', 'cabin_deck', 'cabin_side']
    final_cols = cat_cols_to_use + list(num_cols)

    # Split data into train and validation
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=1)

    # Split data into X and y
    X_train_df = df_train.drop(target, axis=1)
    y_train = df_train[target]

    X_val_df = df_val.drop(target, axis=1)
    y_val = df_val[target]

    # convert to dicts
    train_dicts = X_train_df[final_cols].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dicts)

    # transform dicts
    X_train = dv.transform(train_dicts)
    val_dicts = X_val_df[final_cols].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    # convert data to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dv.feature_names_)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=dv.feature_names_)

    # Train model
    params = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 1,
    }

    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'val')], early_stopping_rounds=10, verbose_eval=10)

    # Predict on validation data
    y_pred_prob = model.predict(dval)
    y_pred = y_pred_prob > 0.5

    print_metrics(y_val, y_pred, y_pred_prob)

    return model, dv


def print_metrics(y_true, y_pred, y_pred_prob):
    '''
    Print metrics
    '''
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
    print(f'Accuracy: {accuracy_score(y_true, y_pred):.3f}')
    print(f'Precision: {precision_score(y_true, y_pred):.3f}')
    print(f'Recall: {recall_score(y_true, y_pred):.3f}')
    print(f'F1: {f1_score(y_true, y_pred):.3f}')
    print(f'AUC: {roc_auc_score(y_true, y_pred_prob):.3f}')

def predict_batch(df, model, dv, target):
    '''
    Predict on df using model and dv
    '''
    # Clean data
    df, cat_cols, num_cols, target = basic_preprocess(df, train=False, target=target)

    # Final features
    cat_cols_to_use = ['homeplanet', 'cabin', 'destination']
    final_cols = cat_cols_to_use + list(num_cols)

    # convert to dicts
    dicts = df[final_cols].to_dict(orient='records')

    # transform dicts
    X = dv.transform(dicts)

    # Predict
    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred = y_pred_prob > 0.5

    # Add predictions to df
    df['prediction'] = y_pred
    df['prediction_probability'] = y_pred_prob

    return df

def predict_single(json_data, model, dv, xgb=False):
    '''
    Predict on json using model and dv
    '''
    import xgboost as xgb_c
    if xgb:
        logging.info('Using XGBoost')
    else:
        logging.info('Using RandomForest')
    # load model and dv
    # model, dv = load_model(model_path) # loading on startup event

    # convert json to df
    df = pd.DataFrame(json_data, index=[0])

    # Clean data
    df, cat_cols_to_use, num_cols, target = basic_preprocess(df, train=False, target='Transported')

    # Final features
    final_cols = cat_cols_to_use + list(num_cols)

    # convert to dicts
    dicts = df[final_cols].to_dict(orient='records')

    if xgb:
        # transform dicts
        X = dv.transform(dicts)
        dtest = xgb_c.DMatrix(X, feature_names=dv.feature_names_)

        # predict on test data
        y_pred_prob = model.predict(dtest)
        y_pred = y_pred_prob > 0.5
    else:
        # transform dicts
        X = dv.transform(dicts)

        # Predict
        y_pred_prob = model.predict_proba(X)[:, 1]
        y_pred = y_pred_prob > 0.5
    
    message = {"PassengerId": str(df['passengerid'][0]), "prediction": str(y_pred[0]), "prediction_probability": float(y_pred_prob[0])}
    
    return message


def save_model(model, dv, path):
    '''
    Save model and dv to path
    '''
    logging.info(f'Saving model to {path}')
    with open(path, 'wb') as f:
        pickle.dump((model, dv), f)
        
def load_model(path):
    '''
    Load model and dv from path
    '''
    logging.info(f'Loading model from {path}')
    with open(path, 'rb') as f:
        model, dv = pickle.load(f)
    return model, dv





