from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import os
import sys

import pandas as pd
import numpy as np
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import logging

import joblib
import warnings

warnings.filterwarnings("ignore")

# sets logging level, means los messages with severity levels of warn,error,critical will be logged
logging.basicConfig(level=logging.WARN)

# creates a logger instance with the name of current module,allwos to generate log messages for this module
logger = logging.getLogger(__name__)


# function to get the data
def get_df(results):
    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) #convert to dattetime-index
    return df


# preprocessing pipeline separately
numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler()),
    ('poly', PolynomialFeatures(3))
])

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])



if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    # create connection to the db
    client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
    client.switch_database('orkney')
    
    # get data
    # last 90 days of power generation data
    generation = client.query(
    "SELECT * FROM Generation where time > now()-90d"
    )

    wind = client.query(
    "SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'"
    )

    # get data as df
    gen_df = get_df(generation)
    wind_df = get_df(wind)

    # merging the dfs
    data = pd.merge_asof(wind_df , gen_df,on="time",tolerance=pd.Timedelta('180T'), allow_exact_matches=True)
    

    # #dropping irrelevant columns
    data = data.drop(["ANM","Non-ANM"],axis=1)

    # splitting the data
    X = data.iloc[:,0:-1]
    y = data.Total

    X_train, x_test, y_train, y_test = train_test_split(X,y,test_size = .3,shuffle=False)

    # find all categorical columns
    object_cols = [col for col in X_train.columns if X_train[col].dtype=='object']


    #columns that can be safely label encoded
    good_label_cols = [col for col in object_cols if set(X_train[col]== set(x_test[col]))] 


    #problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols) - set(good_label_cols))


    #show how many are numerical
    numerical_features = X_train.select_dtypes(include='number').columns.tolist()
    print(numerical_features)
    # only categorical columns
    categorical_features = X_train.select_dtypes(exclude='number').columns.tolist()
    print(categorical_features)

    #preprocessg pipeline for numerical and categorical features
    preprocess = ColumnTransformer(transformers=[
        ('number', numeric_pipeline, numerical_features),
        ('category', categorical_pipeline, categorical_features)
    ])



    epsilon = [float(sys.argv[1])] if len(sys.argv) > 1 else [0.0]
    kernel = [sys.argv[2]] if len(sys.argv) > 2 else ["sigmoid"]
    alpha =   [float(sys.argv[1])] if len(sys.argv)>1 else [0.5]

  
    with mlflow.start_run():
        # mlflow.set_tracking_uri("http://localhost:5000")
    
        
        # setting MLflow tracking server
        # mlflow.set_tracking_uri('http://training.itu.dk:5000/')
        # mlflow.set_experiment('wind_power')
       
        # # Setting the requried environment variables
        # os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://130.226.140.28:5000'
        # os.environ['AWS_ACCESS_KEY_ID'] = 'training-bucket-access-key'
        # os.environ['AWS_SECRET_ACCESS_KEY'] = 'tqvdSsEDnBWTDuGkZYVsRKnTeu'


        #classifiers to be checked for the data
        clf = Lasso()

        #dictionaries with parameters for each clfs
        params={'clf__alpha': alpha}


        # iterate through eveyr classifier

        steps = [('preprocess', preprocess),
                ('clf',clf)]

        pipeline = Pipeline(steps)
        print("Start GridSearch")

        search = GridSearchCV(pipeline, params, 
                        cv=2, 
                        scoring='r2')
        print("End GridSearch")
        
        # fit the model and get the score
        model = search.fit(X_train, y_train)  
        print("End fitting")
        # dict with  best values for coefficients
        cfs = model.best_params_
        print(cfs)
        preds = model.predict(x_test)
        r2 = r2_score(y_test,preds)
        print(r2)

    

        # # save all the params and metric
        mlflow.log_param("alpha",cfs["clf__alpha"])

        mlflow.log_metric("r2",r2)

        mlflow.sklearn.log_model(model,"model")
        

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # model registry does not wirk with file store
        if tracking_url_type_store != 'file':
            # register the model
            # there are other ways to use the Model Registry, which depends on the use case
            mlflow.sklearn.log_model(model,'model', registered_model_name='Lasso')
        else:
            mlflow.sklearn.log_model(model,"model")

        mlflow.end_run()

        df = mlflow.search_runs(filter_string="metrics.r2 > 0.7")
        print(df)