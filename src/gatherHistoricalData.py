import fastf1

fastf1.Cache.enable_cache("./cache")

quali = fastf1.get_session(2023, 'British Grand Prix', 'Q')
quali.load()

quali_results = quali.results

quali_results['Pos'] = quali_results.reset_index().index
quali_data = quali_results[["Abbreviation", "Pos"]]
quali_data.reset_index(drop=True, inplace=True)

print(quali_data)