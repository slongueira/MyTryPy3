import os
import pandas as pd

def sort_function(string):
    return int(string.split("_")[-1].split(".")[0])


def CSV_merge(folder_path, exp_id):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not files:
        return
 
    files.sort(key=sort_function)

    # Create an empty DataFrame
    combined_DataFrame = pd.DataFrame()

    # Iterate the CSV files found in the folder path
    print("\nMerging...")
    for file in files:
        # Read CSV
        df = pd.read_csv(os.path.join(folder_path, file), header=0, 
                         index_col=False, delimiter=';', decimal='.')

        print(file)

        # Concatenate CSV file
        combined_DataFrame = pd.concat([combined_DataFrame, df], ignore_index=True)
    
    # Save concatenated DataFrame
    rawdata_dir = os.path.dirname(folder_path)
    filename = f'Motor-{exp_id}.csv'
    combined_DataFrame.to_csv(os.path.join(rawdata_dir, filename), index=False)
    
    print("Data saved to location:", os.path.join(rawdata_dir, filename))
    
    return filename


def Pickle_merge(folder_path, exp_id):
    files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

    if not files:
        return
    
    files.sort(key=sort_function)

    # Create an empty DataFrame
    combined_DataFrame = pd.DataFrame()

    # Iterate the Excel files found in the folder path
    print("\nMerging...")
    for file in files:
        # Read Excel
        df = pd.read_pickle(os.path.join(folder_path, file))

        print(file)

        # Concatenate CSV file
        combined_DataFrame = pd.concat([combined_DataFrame, df], ignore_index=True)
    
    # Save concatenated DataFrame
    rawdata_dir = os.path.dirname(folder_path)
    filename = f'DAQ-{exp_id}.pkl'
    combined_DataFrame.to_pickle(os.path.join(rawdata_dir, filename))
    
    print("Data saved to location:", os.path.join(rawdata_dir, filename))
    
    return filename


def Files_merge(folder_path, exp_id):
    motor_file = CSV_merge(folder_path, exp_id)
    daq_file = Pickle_merge(folder_path, exp_id)
    
    return motor_file, daq_file


        
        
        
        
        
        
        
        
        
        
        