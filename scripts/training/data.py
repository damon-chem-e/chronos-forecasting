import argparse
import datasets
import time

def load_data(dataset):
    print(f"Loading dataset {dataset}")
    start_time = time.time()

    ds = datasets.load_dataset("autogluon/chronos_datasets", dataset, keep_in_memory=False, split="train")
    print(f"{len(ds)}")
    ds.set_format("numpy")
    new_col = [dataset["timestamp"] for dataset in ds]
    ds = ds.add_column("start", new_col)
    end_time = time.time()

    print(f"Time taken to load dataset {dataset}: {end_time - start_time} seconds")
    print("\n ------------------------------ \n")
    return ds

if __name__ == '__main__':

    
    all_data = ['dominick', 'electricity_15min', 'ercot', 'exchange_rate', 'm4_daily', 'm4_hourly', 'm4_monthly', 'm4_quarterly', 'm4_weekly', 
    'm4_yearly', 'm5', 'mexico_city_bikes', 'monash_australian_electricity', 'monash_car_parts', 'monash_cif_2016', 'monash_covid_deaths', 
    'monash_electricity_hourly', 'monash_electricity_weekly', 'monash_fred_md', 'monash_hospital', 'monash_kdd_cup_2018', 
    'monash_london_smart_meters', 'monash_m1_monthly', 'monash_m1_quarterly', 'monash_m1_yearly', 'monash_m3_monthly', 'monash_m3_quarterly',
      'monash_m3_yearly', 'monash_nn5_weekly', 'monash_pedestrian_counts', 'monash_rideshare', 'monash_saugeenday', 'monash_temperature_rain', 
      'monash_tourism_monthly', 'monash_tourism_quarterly', 'monash_tourism_yearly', 'monash_traffic', 'monash_weather', 'nn5', 'solar', 'solar_1h', 
      'taxi_1h', 'taxi_30min', 'training_corpus_kernel_synth_1m', 'training_corpus_tsmixup_10m', 'uber_tlc_daily', 'uber_tlc_hourly', 'ushcn_daily', 
      'weatherbench_daily', 'weatherbench_hourly_10m_u_component_of_wind', 'weatherbench_hourly_10m_v_component_of_wind', 
      'weatherbench_hourly_2m_temperature', 'weatherbench_hourly_geopotential', 'weatherbench_hourly_potential_vorticity', 
      'weatherbench_hourly_relative_humidity', 'weatherbench_hourly_specific_humidity', 'weatherbench_hourly_temperature', 
      'weatherbench_hourly_toa_incident_solar_radiation', 'weatherbench_hourly_total_cloud_cover', 'weatherbench_hourly_total_precipitation', 
      'weatherbench_hourly_u_component_of_wind', 'weatherbench_hourly_v_component_of_wind', 'weatherbench_hourly_vorticity', 'weatherbench_weekly', 
      'wiki_daily_100k', 'wind_farms_daily', 'wind_farms_hourly']
    
    pretraining_data = ["wind_farms_daily", "wind_farms_hourly", "weatherbench_daily", "wiki_daily_100k"]
    
    parser = argparse.ArgumentParser(description="Script to load data from huggingface")
    parser.add_argument('--datasets', '-d', nargs="*", type=str, default=pretraining_data, help=f"list of datasets to load: {pretraining_data}")

    # Parse the arguments
    args = parser.parse_args()

    # Assign variables from parsed arguments
    data_list = args.datasets
    print(data_list)
    for data in data_list:
        load_data(data)
    

