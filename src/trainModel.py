import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


def main():

    training_file_path = "f1TrainingData.csv"
    hot_shot_columns = ['Year', 'Driver', 'Team']
    features_to_drop = ['RaceID', 'RacePos']

    df = pd.read_csv(training_file_path)

    # Unique race identifiers
    races = df['RaceID'].unique()

    # Preprocessing categorical data (year, driver, team)
    encoded_df = pd.get_dummies(df, columns=hot_shot_columns)
    
    numerical_columns = encoded_df.columns.drop('RaceID')
    encoded_df[numerical_columns] = encoded_df[numerical_columns].astype(float)

    # X=features (input data), and y=labels (output data)
    X = []
    y = []

    # Splititng the csv up into separate races
    for race in races:
        # Grouping races by RaceId
        race_data = encoded_df[encoded_df['RaceID'] == race]
        
        # Extracting features by ignoring RaceID, and RacePos
        features = race_data.drop(columns=features_to_drop).values
        labels = race_data['RacePos'].values

        X.append(features)
        y.append(labels)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Input shape depends on number of features
    input_shape = (features.shape[1], )

    model = Sequential([
        Flatten(input_shape=X_train.shape[1:]),
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(20, activation='linear')
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

    model_name = input("NN Model File Name:")

    model.save(f"./models/{model_name}.h5")

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()