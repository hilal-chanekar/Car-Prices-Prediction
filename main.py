#importing necessary libraries
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model, tree
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import xgboost as xgb
import tensorflow
from keras import models, layers, optimizers, regularizers
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
import random
import catboost as cb
import warnings
warnings.filterwarnings("ignore")
#set random seed to get consistent results
random.seed(11)

def readfile():
    """
        The function reads training and test datasets
    """
    df = pd.read_csv('cars_train.csv')
    df_wo_labels = pd.read_csv('cars_test_without_labels.csv')

    return df, df_wo_labels

def eda(df):
    """
        The function performs preliminary Exploratory Data Analysis
    """
    sns.heatmap(data=df.corr(), annot=True)  #heatmap of corealtion matrix to find co-related columns
    plt.show()
    print(df.corr().to_string()) # 'Kilometers' and 'year' columns are highly co-related with target variable 'price'
    print(df.isna().sum()) #No null values are present in any of the columns

def preprocessing1(df):
    """
        Pre-process the training data: Encoding and Scaling
    """
    df['Present Year'] = 2022
    df['Car Age'] = df['Present Year'] - df['year']   # Make additional feature of determining car age
    df.drop(['Present Year'], inplace=True, axis=1)

    encoder_list = ['manufacturer', 'model', 'fuel', 'title_status', 'transmission', 'drive', 'type', 'paint_color']
    le = LabelEncoder()  # Using Label encoder to encode categorical columns in data

    for i in encoder_list:
        le.fit(df[i])
        df[i] = le.transform(df[i])

    mn = MinMaxScaler()  #Using minmax scaler to scale values of feature between 0 and 1
    scaled_features = pd.DataFrame(mn.fit_transform(df.iloc[:,2:]), columns=df.columns[2:])
    df.drop(df.columns[2:], axis=1, inplace=True)
    # join back the normalized features
    df_scaled = pd.concat([df, scaled_features], axis=1)

    return df_scaled

def preprocessing2(df_wo_labels):
    """
        Pre-process the test data: Encoding and Scaling
    """
    df_wo_labels['Present Year'] = 2022
    df_wo_labels['Car Age'] = df_wo_labels['Present Year'] - df_wo_labels['year']
    df_wo_labels.drop(['Present Year'], inplace=True, axis=1)

    encoder_list = ['manufacturer', 'model', 'fuel', 'title_status', 'transmission', 'drive', 'type', 'paint_color']
    le = LabelEncoder()

    for i in encoder_list:
        le.fit(df_wo_labels[i])
        df_wo_labels[i] = le.transform(df_wo_labels[i])

    mn = MinMaxScaler()
    scaled_features = pd.DataFrame(mn.fit_transform(df_wo_labels.iloc[:,1:]), columns=df_wo_labels.columns[1:])
    df_wo_labels.drop(df_wo_labels.columns[1:], axis=1, inplace=True)
    # join back the normalized features
    df_scaled_wo_labels = pd.concat([df_wo_labels, scaled_features], axis=1)
    #print(df_scaled_wo_labels.head(100).to_string())

    return df_scaled_wo_labels

def linearregression(df):
    """
        Model simple linear regression with training data
    """
    print("Running Linear Regression\n")
    x = df.drop('price', axis=1)
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)  # using 80-20 train-test split

    reg = linear_model.LinearRegression()   #model linear regression
    reg.fit(x_train, y_train)   #fit the training data to the model

    y_pred = reg.predict(x_test)   # predict the prices of test data

    mae = mean_absolute_error(y_test, y_pred)  #4002.1614094637066
    r2 = r2_score(y_test, y_pred)   #0.0589499
    return print("Linear Regression MAE: ",mae)

def decisiontree(df):
    """
        Model Decision tree with training data
        Model crashes due to memory limitations
    """
    print("Running Decision Tree\n")
    x = df.drop('price', axis=1)
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    clf = tree.DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=6, min_samples_leaf=5).fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    return print("Decision Tree MAE:", mae)

def xgboost_default(df):
    """
        Model XGBoost regressor with training data: Using default parameters
    """
    print("Running XGBoost(default)")
    x = df.drop('price', axis=1)
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    xgb_reg = xgb.XGBRegressor()
    xgb_reg.fit(x_train, y_train)
    y_pred = xgb_reg.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)  # Default xgb model MAE: 2949.0312
    return print("XGB(default) MAE: ", mae)

def xgboost_gridsearch(df):
    """
        Model XGBoost regressor with training data: Using GridSearch to find best model parameters
        This does not improve the default XGB model, rather slightly reduces the performance
    """
    print("Running XGBoost GridSearch")
    x = df.drop('price', axis=1)
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    xgb_reg = xgb.XGBRegressor()
    # Using different depths of tree, leaning rates and estimators
    parameters = {'max_depth': [2, 4, 6],'learning_rate': [0.1, 0.5, 1.0],'n_estimators': [50, 100, 200]}
    grid_search = GridSearchCV(xgb_reg, parameters,cv=5) # Using 5 fold cross validation

    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_  # After grid search is finished, fit a model with best parameters

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(x_train, y_train)

    y_pred = final_model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)  # xgb model(with gridsearch) MAE: 2974.3740
    return print("XGB(GridSearch) MAE: ", mae)

def catboost_default(df):
    """
        Model CatBoost regressor which is specifically optimised for data having many categorical features: Using default parameters
    """
    print("Running CatBoost")
    x = df.drop('price', axis=1)
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    train_dataset = cb.Pool(x_train, y_train)

    model = cb.CatBoostRegressor(loss_function='MAE')  #defining CatBoost regressor with MAE as loss function to improve in each iteration
    model.fit(train_dataset)
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)  # CatBoost MAE: 2361.6369, 2375.6728
    print("CatBoost MAE: ", mae)

    return 0

def catboost_final(df,df_wo_labels):
    """
        Model CatBoost regressor with training data
        Using given test data instead of splitting train dataset
        Yeilds best result in terms of MAE
    """
    print("Running CatBoost")
    x = df.drop('price', axis=1)
    y = df['price']

    train_dataset = cb.Pool(x, y)

    model = cb.CatBoostRegressor(loss_function='MAE')
    model.fit(train_dataset)
    y_pred = model.predict(df_wo_labels)  # predict prices for given test data without label info

    output = pd.DataFrame(data={"ID": df_wo_labels["ID"], "predicted_price": y_pred})

    output.to_csv("predictions_Nikte_Omkar_Chanekar_Hilal.csv", index=False, sep=',')  #Export ID and predicted price in csv file

    return 0

def catboost_feature_selection(df):
    """
        Model CatBoost regressor with only selected features from training data which have high corelation to price
        Does not improve MAE, Yeilds worse MAE than default CatBoost regressor
    """
    print("Running CatBoost with selected features")
    x = df[['year','kilometers']]   #selecting Year and kilometers have significantly high corelation with price
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    train_dataset = cb.Pool(x_train, y_train)

    model = cb.CatBoostRegressor(loss_function='MAE')
    model.fit(train_dataset)
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)  #MAE: 2849
    return print("CatBoost feature selected MAE: ", mae)

def nn(df):
    """
        Model Neural Network with training data
        Activation criteria: relu
        Optimizer: Adam
        Experimenting with different number of layers and nodes: 5 layers and 512 nodes yeilds best MAE
    """
    print("Running Neural Network")
    x = df.drop('price', axis=1)
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    nn2 = models.Sequential()
    nn2.add(layers.Dense(512, input_shape=(x_train.shape[1],), activation='relu'))
    nn2.add(layers.Dense(512, activation='relu'))
    nn2.add(layers.Dense(512, activation='relu'))
    nn2.add(layers.Dense(512, activation='relu'))
    nn2.add(layers.Dense(512, activation='relu'))
    nn2.add(layers.Dense(512, activation='relu'))
    nn2.add(layers.Dense(512, activation='relu'))
    nn2.add(layers.Dense(1, activation='linear'))

    # Compiling the model
    nn2.compile(loss='mean_square_error',optimizer='adam',metrics=['mean_square_error'])

    y_pred = nn2.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    # 5 layers * 1024 nodes : MAE: 5320.9361
    # 5layers * 512 nodes : MAE: 3934.8446
    # 5layers * 256 nodes : MAE: 6856.1503
    # 7layers * 512 nodes : MAE: 3997.0515
    return print("NN MAE: ", mae)

def runmodel(model):
    """
        Takes input as Number to run corresponding model
        1: Linear regression
        2: Decision Tree
        3: XGBoost with default parameters
        4: XGBoost with GridSearch
        5: Neural Network
        6: CatBoost
        7: CatBoost with feature selection
    """
    df, df_wo_labels = readfile()  # read training data and test data(without labels)

    df_processed = preprocessing1(df)  # preprocessing
    df_wo_labels_processed = preprocessing2(df_wo_labels)

    # Choose which model to run
    if(model == 1):
        output = linearregression(df_processed)  # MAE:4002.1614
    elif(model == 2):
        output = decisiontree(df_processed) # memory error
    elif (model == 3):
        output = xgboost_default(df_processed) #MAE:2949.0312
    elif (model == 4):
        output = xgboost_gridsearch(df_processed) #MAE:2974.3740
    elif (model == 5):
        output = nn(df_processed) #MAE: 3934.8446
    elif (model == 6):
        output = catboost_final(df_processed,df_wo_labels_processed) #MAE: 2379.2033
    elif (model == 7):
        output = catboost_feature_selection(df_processed) #MAE: 2849.68324

    return output


runmodel(6) # CatBoost model described in catboost_final function yeilds best MAE of 2379.2033

