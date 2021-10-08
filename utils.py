import pickle
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model

base_path = './data'

def load_pickle(path):
    # Load pickle from path
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def standart_scaling(input_dataframe):
    # Relevant feature column
    feat_cols = [key for key in input_dataframe.columns if "hand" in key or "ankle" in key]
    # Stay with relevant features data
    features = input_dataframe[feat_cols]
    # Load StandartScaler
    scaler = load_pickle(f'{base_path}/scaler.pickle')
    # Transform data and return
    scaled_features = scaler.transform(features.values)
    # Replace existing features with scaled features
    input_dataframe[feat_cols] = scaled_features
    return input_dataframe

def window_and_nan(input_dataframe, labeled, timestamps=400):
    # Removing not relevant column
    del input_dataframe["timestamp"]
    x, y = [], []
    # Iterate over subjects, so we dont get overlap samples between subjects
    for subject in list(input_dataframe["subject_id"].unique()):
        print(f'Processing subject #{subject}')
        subject_df = input_dataframe[input_dataframe["subject_id"] == subject]
        for i in range(subject_df.shape[0]-timestamps-1):
            if i % 10000 == 0:
                print(f'finished ({i} / {subject_df.shape[0]}) samples')
            curr_df = subject_df.iloc[i:i+timestamps]
            # If any one of the values from the features is NaN - continue
            if curr_df.isnull().sum().sum() > 0:
                continue
            # If in the middle of the 400 samples, the activity changes, than continue. (and subject just to be sure)
            if curr_df["activityID"].nunique() > 1 or curr_df["subject_id"].nunique() > 1:
                continue
            # If this data is labeled, then extract the activity number
            if labeled:
                y.append(curr_df["activityID"].values[0])

            # Delete irrelevant columns once i'm done with them
            del curr_df["subject_id"]
            del curr_df["activityID"]

            # append the 400x6 matrix
            x.append(np.array(curr_df.values))

    return np.array([np.array(val) for val in x]), np.array(y)


def label_and_onehot(Y):
    # Load labelencoder
    labelencoder = load_pickle(f'{base_path}/labelencoder.pickle')
    # Load OneHotencoder
    onehot = load_pickle(f'{base_path}/OneHotEncoder.pickle')
    # transform labelencoder
    Y = labelencoder.transform(Y)
    # transform OneHot
    return onehot.transform(Y.reshape(-1, 1)).toarray()


def from_input_dataframe_to_model_ready_data(input_dataframe, labeled=True):
    # Stay with relevant activities
    relevant_activities = [4, 5, 7]
    input_dataframe = input_dataframe[input_dataframe["activityID"].isin(relevant_activities)]

    # Scaling input features with Standart Scaler
    input_dataframe = standart_scaling(input_dataframe)

    # 4 second window & remove NaNs
    X, Y = window_and_nan(input_dataframe, labeled)

    # LabelEncoding than OneHot
    Y = label_and_onehot(Y)

    return X, Y

def get_model(X_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_shape[1], X_shape[2])))
    model.add(Dense(100,activation="relu"))
    model.add(Dense(3,activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', 'AUC'])
    return model


def load_LSTM_model():
    return load_model(f"{base_path}/LSTM_model.h5")

def load_pickle(path):
    # Load pickle from path
    with open(path, 'rb') as handle:
        return pickle.load(handle)

# X, Y = from_input_dataframe_to_model_ready_data(df)