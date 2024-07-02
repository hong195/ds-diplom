from datetime import datetime

import dill
import pandas as pd

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.pipeline import Pipeline


def create_features(df):

    df = df.copy()

    def handle_keyword_len(keyword):
        return 0 if keyword == 'unknown' else len(keyword)

    df.loc[:, 'utm_keyword_len'] = df['utm_keyword'].apply(handle_keyword_len)

    df.loc[:, 'is_organic_traffic'] = df.utm_medium.isin(['organic', 'referral', '(none)'])

    df.loc[:, 'payable_trafic'] = df.is_organic_traffic.apply(lambda x: x != True)

    df.loc[:, 'is_social_add'] = df.utm_source.isin(['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs','IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw','gVRrcxiDQubJiljoTbGm'])

    def handle_screen_res(res):
        if res == '(not set)':
            return 0, 0

        return res.split('x')

    df.loc[:, 'screen_res_width'] = df.device_screen_resolution.apply(
        lambda x: handle_screen_res(x)[0]).astype(int)
    df.loc[:, 'screen_res_height'] = df.device_screen_resolution.apply(
        lambda x: handle_screen_res(x)[1]).astype(int)

    return df


def main():
    df = pd.read_csv('./data/train/df_sessions.csv')

    X = df.drop(['is_target_action'], axis=1)
    y = df['is_target_action']

    categorical_features = make_column_selector(dtype_include=object)
    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('feature_creator', FunctionTransformer(create_features)),
        ('column_transformer', column_transformer),
    ])

    models = [
        SGDClassifier(loss='log_loss'),
        LogisticRegression(solver='liblinear'),
    ]

    best_score = .0
    best_pipe = None
    for model in models:

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')

        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, rog-auc: {best_score:.4f}')

    best_pipe.fit(X, y)

    with open('./model/target-model.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Target action prediction model',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'rog-auc': best_score
            }
        }, file)


if __name__ == '__main__':
    main()
