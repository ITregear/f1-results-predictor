import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
import sklearn.metrics as metrics
from scipy.stats import spearmanr, kendalltau

from gatherHistoricalData import getRaceData


def main():

    training_file_path = "f1TrainingData.csv"
    hot_shot_columns = ['Year', 'Driver', 'Team']
    features_to_drop = ['RaceID', 'RacePos']

    training_df = pd.read_csv(training_file_path)

    encoded_training = pd.get_dummies(training_df, columns=hot_shot_columns)

    model = load_model("./models/2021-2023_driverteamyear.h5")
    
    race_to_predict = getRaceData(2023, 3)
    encoded_race = pd.get_dummies(race_to_predict, columns=hot_shot_columns)
    encoded_race = encoded_race.reindex(columns=encoded_training.columns, fill_value=0)
    race_features = encoded_race.drop(columns=features_to_drop, errors='ignore')

    race_features = np.array([race_features], dtype='float')

    print([race_features.shape])

    # Creating variables for quali_positions, actual_positions, predicted_positions
    quali_positions = race_to_predict['QualiPos'].values.reshape(1, -1)
    actual_positions = race_to_predict.sort_values(by='QualiPos')['RacePos'].values  # Ensure this matches the order of quali_positions
    predicted_positions_raw = model.predict(race_features)
    predicted_positions = np.argsort(predicted_positions_raw[0]) + 1  # This will be used for ranking, not direct comparison

    # Creating variables for driver orders
    quali_driver_order = race_to_predict.sort_values(by='QualiPos')['Driver'].values
    actual_driver_order = race_to_predict.sort_values(by='RacePos')['Driver'].values
    driver_mapping = {pos: driver for pos, driver in zip(race_to_predict['QualiPos'], race_to_predict['Driver'])}
    predicted_driver_order = [driver_mapping[pos] for pos in predicted_positions]

    # Metrics
    '''
    Metrics to aim for:
    MAE = Mean Absolute Error, close to 0
    RMSE = Root Mean Squared Error, close to 0
    Spearman's Rank, close to 1
    Kendapll's Tau, close to 1
    '''
    mae = metrics.mean_absolute_error(actual_positions, predicted_positions)
    rmse = np.sqrt(metrics.mean_squared_error(actual_positions, predicted_positions))
    spearman_corr, _ = spearmanr(actual_positions, predicted_positions)
    kendall_tau, _ = kendalltau(actual_positions, predicted_positions)

    # Print race prediction
    # Print race prediction comparison
    print(race_to_predict['RaceID'][0])
    print("Quali vs Predicted vs Actual")
    for i, (quali_driver, predicted_driver, actual_driver) in enumerate(zip(quali_driver_order, predicted_driver_order, actual_driver_order), start=1):
        print(f"{i:2}. {quali_driver:20}\t{predicted_driver:20}\t{actual_driver:20}")
    
    print(f"\nMetrics:\nMAE: {mae}\nRMSE: {rmse}\nSpearman's Rank Correlation: {spearman_corr}\nKendall's Tau: {kendall_tau}")



if __name__ == "__main__":
    main()
