import pandas as pd
import numpy as np
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy.stats import ttest_1samp


def random_assignment(df: pd.DataFrame, sample_size: int, sample_count: int) -> list:
    '''
    Take rideshare data (df), make sample_count number of chunks with sample_size rows each,
    then randomly assign predictions of tip or no tip. Calculate f1-score and append to list
    to output distribution of f1-scores.

    Parameters
    ----------
    df : dataframe
        pandas dataframe of rideshare data
    sample_size : int
        number of rows in each chunk
    sample_count : int
        number of samples to create

    Returns
    -------
    distribution : list
        list of f1-scores from samples
    '''
    print('\ncreating random assignment distribution')
    distribution = []

    for _ in range(sample_count):
        df_narrow = copy.deepcopy(df[['tips']].sample(sample_size))
        df_narrow['flag_tip'] = df_narrow.apply(lambda x: 1 if x['tips'] > 0 else 0, axis=1)
        df_narrow['predict_tip'] = np.random.randint(2, size=sample_size)
        f1_score = calculate_f1_score(df_narrow['flag_tip'], df_narrow['predict_tip'], True)
        distribution.append(f1_score)

    return distribution


def prepare_training(df: pd.DataFrame, sample_size: int, test_size: float) -> tuple[pd.DataFrame]:
    '''
    Prepare data for training ML model. Feature creation for datetime values. Encoding
    for categorical variables, standard scaler for all values, SMOTE for training values.

    Parameters
    ----------
    df : dataframe
        pandas dataframe of rideshare data
    sample_size : int
        number of rows in test data
    test_size : float
        percentage of training data to use in test
    
    Returns
    -------
    train_x_resample_scaled : dataframe
        training features with SMOTE resampling and standard scaling
    train_y_resample : dataframe
        training lables with SMOTE
    test_x_scaled : dataframe
        test features with standard scaling
    test_y : dataframe
        test labels
    '''
    print('\npreparing training data')
    df_narrow = copy.deepcopy(df.sample(int(sample_size / test_size)))
    df_narrow['flag_tip'] = df_narrow.apply(lambda x: 1 if x['tips'] > 0 else 0, axis=1)
    df_narrow['day_of_week'] = df_narrow['request_datetime'].dt.dayofweek
    df_narrow['24_hour_timestamp'] = pd.to_datetime(
        df_narrow['request_datetime'], format='%m/%d/%Y %I:%M:%S %p')
    df_narrow['hour'] = df_narrow['24_hour_timestamp'].dt.hour
    df_narrow.drop(['request_datetime', 'on_scene_datetime', 'pickup_datetime',
                    'dropoff_datetime', '24_hour_timestamp'], axis=1, inplace=True)

    list_categorical_variables = ['hvfhs_license_num', 'dispatching_base_num',
                                  'originating_base_num', 'PULocationID', 'DOLocationID',
                                  'shared_request_flag', 'shared_match_flag', 'access_a_ride_flag',
                                  'wav_request_flag', 'wav_match_flag']
    df_rideshare_categorical = df_narrow[list_categorical_variables]
    df_rideshare_continuous = df_narrow.drop(list_categorical_variables, axis=1)
    dict_encoder = defaultdict(LabelEncoder)
    df_rideshare_encoded = df_rideshare_categorical.apply(
        lambda x: dict_encoder[x.name].fit_transform(x))
    df_rideshare_merge = pd.merge(left=df_rideshare_continuous, right=df_rideshare_encoded,
                                  how='inner', left_index=True, right_index=True)

    df_label = df_rideshare_merge['flag_tip']
    df_features = df_rideshare_merge.drop(['tips', 'flag_tip'], axis=1)

    train_x, test_x, train_y, test_y = train_test_split(
        df_features, df_label, test_size=.3, shuffle=True, stratify=df_label)

    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    smote = SMOTE(sampling_strategy='auto')
    train_x_resample, train_y_resample = smote.fit_resample(train_x_scaled, train_y)

    return (train_x_resample, train_y_resample, test_x_scaled, test_y)


def train_xgboost(train_feature: pd.DataFrame, train_label: pd.DataFrame,
                  test_feature: pd.DataFrame, test_label: pd.DataFrame) -> float:
    '''
    Train XGBoost ensemble method.

    Parameters
    ----------
    train_feature : DataFrame
        training x values
    train_label : DataFrame
        training y values
    test_feature : DataFrame
        test x values
    test_label : DataFrame
        test y values
    
    Returns
    -------
    f1_score : float
        f1_score of XGBoost of tipping class
    '''
    print('\ntraining xgboost')
    model_xgboost = XGBClassifier(eta=.9, max_depth=25, min_child_weight=25, subsample=0.5)
    model_xgboost.fit(train_feature, train_label)
    pred_y = model_xgboost.predict(test_feature)
    f1_score = calculate_f1_score(test_label, pred_y)

    return f1_score


def train_decision_tree(train_feature: pd.DataFrame, train_label: pd.DataFrame,
                        test_feature: pd.DataFrame, test_label: pd.DataFrame) -> float:
    '''
    Train decision tree. Standard model to test any more complicated models against

    Parameters
    ----------
    train_feature : DataFrame
        training x values
    train_label : DataFrame
        training y values
    test_feature : DataFrame
        test x values
    test_label : DataFrame
        test y values
    
    Returns
    -------
    f1_score : float
        f1_score
    '''
    print('\ntraining decision tree')
    model_tree = DecisionTreeClassifier()
    model_tree.fit(train_feature, train_label)
    pred_y = model_tree.predict(test_feature)
    f1_score = calculate_f1_score(test_label, pred_y)

    return f1_score


def train_random_forest(train_feature: pd.DataFrame, train_label: pd.DataFrame,
                        test_feature: pd.DataFrame, test_label: pd.DataFrame) -> float:
    '''
    Train random forest ensemble method.

    Parameters
    ----------
    train_feature : DataFrame
        training x values
    train_label : DataFrame
        training y values
    test_feature : DataFrame
        test x values
    test_label : DataFrame
        test y values
    
    Returns
    -------
    f1_score : float
        f1_score
    '''
    print('\ntraining random forest')
    model_tree = RandomForestClassifier(min_samples_leaf=250)
    model_tree.fit(train_feature, train_label)
    pred_y = model_tree.predict(test_feature)
    f1_score = calculate_f1_score(test_label, pred_y)

    return f1_score


def calculate_f1_score(y: pd.Series, y_pred: pd.Series, no_print: bool = False) -> float:
    '''
    Calculate f1_score using two series

    Parameters
    ----------
    y : series
        actual labels for y
    y_pred : series
        predicted labels for y
    no_print : bool
        whether to print scores in console

    Returns
    -------
    f1_score : float
        f1_score for predictions
    '''
    df_testing = pd.DataFrame()
    df_testing['flag_tip'] = y
    df_testing['predict_tip'] = y_pred
    recall_base = df_testing[df_testing['flag_tip'] == 1].shape[0]
    precision_base = df_testing[df_testing['predict_tip'] == 1].shape[0]
    top = df_testing[(df_testing['flag_tip'] == 1) & (df_testing['predict_tip'] == 1)].shape[0]
    recall = top / recall_base
    precision = top / precision_base
    f1_score = (2 * recall * precision) / (recall + precision)
    if not no_print:
        print(f'recall: {round(recall, 2)}')
        print(f'precision: {round(precision, 2)}')
        print(f'f1 score: {round(f1_score, 2)}')
    return f1_score


def visualize_distribution(distribution: list, test: list[tuple]) -> None:
    '''
    Visualize the distribution of randomly assigned samples and the ML designated sample.

    Parameters
    ----------
    distribution : list
        list of f1-scores from randomly assigned
    test : list[tuple]
        list of f1-scores from ML models
    
    Returns
    -------
    None
    '''
    plt.hist(distribution, bins=20)
    for stat in test:
        _, p_value = ttest_1samp(distribution, stat[0])
        p_value_round = round(p_value, 5)
        score_round = round(stat[0], 2)
        plt.axvline(x=stat[0], linestyle='dashed', color=f'{stat[2]}',
            label=f'{stat[1]} (p value: {p_value_round}, f1: {score_round})')
    plt.xlabel('f1-scores')
    plt.ylabel('frequency')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    SAMPLE_SIZE = 100_000
    SAMPLE_NUM = 100

    df_rideshare = pd.read_parquet('fhvhv_tripdata_2022-07.parquet')
    df_rideshare.dropna(axis=1, how='all', inplace=True)
    df_rideshare.dropna(axis=0, how='any', inplace=True)

    f1_distribution = random_assignment(df_rideshare, SAMPLE_SIZE, SAMPLE_NUM - 1)

    train_features, train_labels, test_features, test_labels = prepare_training(
        df_rideshare, SAMPLE_SIZE, .3)
    f1_score_xg = train_xgboost(train_features, train_labels, test_features, test_labels)
    f1_score_tree = train_decision_tree(train_features, train_labels, test_features, test_labels)
    f1_score_rf = train_random_forest(train_features, train_labels, test_features, test_labels)

    visualize_distribution(f1_distribution, [
        (f1_score_tree, 'tree', 'red'), (f1_score_xg, 'xgb', 'green'),
        (f1_score_rf, 'rf', 'black')])
    