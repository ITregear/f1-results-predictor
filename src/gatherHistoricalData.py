import fastf1
import pandas as pd

fastf1.Cache.enable_cache("./cache")


def getNumberOfEvents(year):
    schedule = fastf1.get_event_schedule(year)

    race_events = schedule['EventName'].unique()
    number_of_races = len(race_events)

    return number_of_races


def getRaceData(race_year, round_number):

    quali_session = fastf1.get_session(race_year, round_number, 'Q')
    race_session = fastf1.get_session(race_year, round_number, 'R')

    quali_session.load()
    race_session.load()

    quali_results = quali_session.results
    race_results = race_session.results

    # Create a mapping of driver IDs to their race positions
    race_pos_map = {row['DriverId']: row['Position'] for index, row in race_results.iterrows()}
    
    # Now use the mapping to fetch race positions for drivers in the qualifying results
    event_data = {
        'RaceID': f"{quali_session.event['EventName'].replace(' ', '')}{race_year}",
        'Year': race_year,
        'Driver': [row['DriverId'] for _, row in quali_results.iterrows()],
        'Team': [row['TeamId'] for _, row in quali_results.iterrows()],
        'QualiPos': [row['Position'] for _, row in quali_results.iterrows()],
        'RacePos': [race_pos_map.get(row['DriverId']) for _, row in quali_results.iterrows()],
    }

    event_df = pd.DataFrame(event_data)

    # Convert RacePos to integers, handling None values by filling with a default value before conversion
    event_df['QualiPos'] = event_df['QualiPos'].fillna(-1).astype(int)
    event_df['RacePos'] = event_df['RacePos'].fillna(-1).astype(int)

    # Optionally, remove drivers who are not present for both sessions
    event_df.dropna(subset=['RacePos'], inplace=True)

    return event_df


def main():

    start_year = 2021
    end_year = 2023

    file_path = "./f1TrainingData.csv"

    for year in range(start_year, end_year + 1):
        number_of_races = getNumberOfEvents(year)

        for race in range(1, number_of_races):
            print(race)
            event = getRaceData(year, race)

            if year == start_year and race == 1:
                write_header = True
                write_mode = 'w'
            else:
                write_header = False
                write_mode = 'a'
            
            event.to_csv(file_path, mode=write_mode, header=write_header, index=False)


if __name__ == "__main__":
    main()