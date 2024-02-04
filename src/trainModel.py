import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def main():

    training_file_path = "f1TrainingData.csv"

    df = pd.read_csv(training_file_path)

    # Unique race identifiers
    races = df['RaceID'].unique()

    X = []
    y = []

    for race in races:
        race_data = df[df['RaceID'] == race]
        quali_positions = race_data['QualiPos'].tolist()
        race_positions = race_data['RacePos'].tolist()

        X.append(quali_positions)
        y.append(race_positions)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(20,)),
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