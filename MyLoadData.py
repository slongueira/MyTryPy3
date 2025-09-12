import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


# %% --------------------------------------------------------------------------
# INITIAL CONFIGURATION
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.ERROR,
                    format="%(levelname)s: %(message)s"
)

REQ_LEVEL = 25
logging.addLevelName(REQ_LEVEL, "REQ")

def req(self, message, *args, **kwargs):
    if self.isEnabledFor(REQ_LEVEL):
        self._log(REQ_LEVEL, message, args, **kwargs)

logging.Logger.req = req
logger0 = logging.getLogger(__name__)

if __name__ == "__main__":
    logger0.setLevel(logging.INFO)


# %% --------------------------------------------------------------------------
# PARSE CUSTOM TIME TO SECONDS
# -----------------------------------------------------------------------------

def LTIME_to_seconds(LTIME):
    
    conversor = {"d":86400,
                 "h": 3600,
                 "m": 60,
                 "s": 1,
                 "ms": 1e-3,
                 "us": 1e-6,
                 "ns": 1e-9}

    units = re.split(r'\d+', LTIME)[1:]
    numbers_str = re.findall(r'\d+', LTIME)
    numbers = [int(number) for number in numbers_str]
    
    total_time = 0

    for number, unit in zip(numbers, units):
        total_time += number * conversor[unit]
    
    return total_time


# %% --------------------------------------------------------------------------
# RENAME RAWDATA DICTIONARIES
# -----------------------------------------------------------------------------

MotColumnsRenames = {
    'Time(s)': 'Time',
    'MC SW Overview - Actual Position(mm)': 'Position',
    'MC SW Force Control - Measured Force(N)': 'Force',
    'MC SW Force Control - Target Force(N)': 'TargetForce',
    'LINMOT_MOVING_BOOL': 'Bool1',
    'LINMOT_UP_AND_DOWN_BOOL': 'Bool2'
}

DaqColumnsRenames = {
    'Time (s)': 'Time',
    'Signal': 'Voltage',
    'Current': 'Current',
    'LINMOT_ENABLE': 'Bool1',
    'LINMOT_UP_DOWN': 'Bool2'
}


# %% --------------------------------------------------------------------------
# LOAD MOTOR RAWDATA
# -----------------------------------------------------------------------------

def LoadMotorFile(MotorFile):
    '''
    Loads and processes a motor CSV file.
    
    Parameters
    ----------
    MotorFile : str
        Path to the data file.
    
    Returns
    -------
    pd.DataFrame or None
        Processed motor data or None if there is an error.
    '''
    try:
        dfMot = pd.read_csv(MotorFile, header=0, index_col=False,
                            delimiter=',', decimal='.')
    except Exception as e:
        logger0.error(f'Error reading Motor file {MotorFile}: {e}.')
        return None
    
    # Drop non-defined columns
    dropcols = [col for col in dfMot.columns if col not in MotColumnsRenames]
    dfMot = dfMot.drop(columns=dropcols)
    
    # Check if all defined columns exist
    for col in MotColumnsRenames.keys():
        if col not in dfMot.columns:
            logger0.error(f'Column {col} not found in {MotorFile}.')
            return None
    
    # Rename defined columns
    dfMot = dfMot.rename(columns=MotColumnsRenames)
    
    # Ensure columns have the correct data types
    dfMot = dfMot.astype({
        'Position': float,
        'Force': float,
        'TargetForce': float,
        'Bool1': int,
        'Bool2': int
    })
    
    # Corrections Time
    dfMot['Time'] = dfMot['Time'].apply(LTIME_to_seconds).astype(float)
    dfMot['Time'] -= dfMot['Time'].iloc[0]
    
    # Corrections Position
    dfMot['Position'] -= dfMot['Position'].min()
    
    # Corrections Force and TargetForce
    dfMot['Force'] = - dfMot['Force']
    dfMot['TargetForce'] = - dfMot['TargetForce']
    
    # Estimate Velocity and Acceleration
    dfMot['Velocity'] = (dfMot['Position'].diff()/1000) / dfMot['Time'].diff()
    dfMot['Acceleration'] = dfMot['Velocity'].diff() / dfMot['Time'].diff()
    
    # Create State variable
    dfMot['State'] = dfMot['Bool1'] + dfMot['Bool2']
    dfMot = dfMot.drop(columns=['Bool1', 'Bool2'])
    
    return dfMot


# %% --------------------------------------------------------------------------
# LOAD DAQ RAWDATA
# -----------------------------------------------------------------------------

def LoadDAQFile(DaqFile):
    '''
    Loads and processes a DAQ pickle file.
    
    Parameters
    ----------
    DaqFile : str
        Path to the data file.
    
    Returns
    -------
    pd.DataFrame or None
        Processed DAQ data or None if there is an error.
    '''
    try:
        dfDaq = pd.read_pickle(DaqFile)
    except Exception as e:
        logger0.error(f'Error reading DAQ file {DaqFile}: {e}.')
        return None
    
    # Drop non-defined columns:
    dropcols = [col for col in dfDaq.columns if col not in DaqColumnsRenames]
    dfDaq = dfDaq.drop(columns=dropcols)
    
    # Check if all defined columns exist
    for col in DaqColumnsRenames.keys():
        if col not in dfDaq.columns:
            if col == 'Current':
                dfDaq.insert(2, 'Current', None)
                dfDaq['Current'] = np.zeros(len(dfDaq), dtype=float)
                logger0.warning(f'Column Current not found in {DaqFile}. Filled with zeros.')
            else:
                logger0.error(f'Column {col} not found in {DaqFile}.')
                return None
    
    # Rename defined columns
    dfDaq = dfDaq.rename(columns=DaqColumnsRenames)
    
    # Ensure columns have the correct data types
    dfDaq = dfDaq.astype({
        'Time': float,
        'Voltage': float,
        'Current': float,
        'Bool1': int,
        'Bool2': int
    })
    
    # Corrections Time
    dfDaq['Time'] -= dfDaq['Time'].iloc[0]
    
    # Corrections Voltage
    j = dfDaq['Bool1'].eq(1).idxmax()
    dfDaq['Voltage'] -= dfDaq['Voltage'][:j].mean()
    
    # Create State variable
    dfDaq['State'] = dfDaq['Bool1'] + dfDaq['Bool2']
    dfDaq = dfDaq.drop(columns=['Bool1', 'Bool2'])
    
    return dfDaq


# %% --------------------------------------------------------------------------
# FIND CYCLES FUNCTION
# -----------------------------------------------------------------------------

def FindCycles(state_series):
    '''
    Identifies start and end indices of operational cycles based on state changes.
    
    Parameters
    ----------
    state_series : pd.Series
        Series of state values.
    
    Returns
    -------
    list of [start, end]
        List of cycles represented by their star and end indices.
    '''
    cycles = []
    prev_state = state_series.iloc[0]
    start = None
    
    for i, s in enumerate(state_series[1:], start=1):
        if s != prev_state:
            if s == 2:
                if start is not None:
                    cycles.append([start, i - 1])
                start = i
            elif s == 0 and start is not None:
                cycles.append([start, i - 1])
                start = None
            prev_state = s
        
    if start is not None:
        cycles.append([start, len(state_series) - 1])
    
    return cycles


# %% --------------------------------------------------------------------------
# LOAD AND SYNCHRONIZE RAWDATA FILES
# -----------------------------------------------------------------------------

def LoadFiles(MotorFile, DaqFile):
    '''
    Loads and synchronizes Motor and DAQ data files, returning combined cycles data.
    
    Parameters
    ----------
    ExpDef : object
        An object with attributes 'MotorFile' and 'DaqFile' which correspond to file paths.
    
    Returns
    -------
    tuple (pd.DataFrame, list)
        Combined data of all cycles and a list of (cicles_index, DataFrame) tuples.
    '''
    # Load Motor File
    dfMot = LoadMotorFile(MotorFile)
    if dfMot is None:
        return None, None
    
    # Load DAQ File
    dfDaq = LoadDAQFile(DaqFile)
    if dfDaq is None:
        return None, None
    
    # Motor sampling rate
    MotFs = 1 / dfMot['Time'].diff().mean()
    logger0.info(f'Motor sampling rate: {MotFs}.')
    
    # DAQ sampling rate
    DaqFs = 1 / dfDaq['Time'].diff().mean()
    logger0.info(f'DAQ sampling rate: {DaqFs}.')
                    
    # Finding cycles
    MotCycles = FindCycles(dfMot['State'])
    DaqCycles = FindCycles(dfDaq['State'])
    if len(MotCycles) != len(DaqCycles):
        logger0.info(f'Different number of cycles: Motor={len(MotCycles)}, DAQ={len(DaqCycles)}. Using minimum.')
    nCycles = min(len(MotCycles), len(DaqCycles))
    
    Cycles = []
    for idx in range(nCycles):
        # Obtain data from each cycle
        dfcyM = dfMot.iloc[MotCycles[idx][0] : MotCycles[idx][1] + 1].reset_index(drop=True)
        dfcyD = dfDaq.iloc[DaqCycles[idx][0] : DaqCycles[idx][1] + 1].reset_index(drop=True)
        
        # Find the first index where State == 1
        mask = dfcyM['State'].eq(1)
        if mask.any():
            i = mask.idxmax()
        else:
            # Incomplete cycle
            continue
        
        mask = dfcyD['State'].eq(1)
        if mask.any():
            j = mask.idxmax()
        else:
            # Incomplete cycle
            continue
        
        for col in dfcyM.columns:
            if col == 'Time' or col == 'State':
                continue
            
            dfcyD[col] = np.zeros(len(dfcyD), dtype=float)
            
            # Down-phase interpolation
            valid_mask_down = dfcyM[col].iloc[:i].notna()
            xp = dfcyM['Time'].iloc[:i][valid_mask_down].values
            fp = dfcyM[col].iloc[:i][valid_mask_down].values
            x = dfcyD['Time'].iloc[:j].values
            
            if np.all(np.diff(xp) > 0) and len(xp) > 1:
                dfcyD.loc[dfcyD.index[:j], col] = np.interp(x, xp, fp)
            
            # Up-phase interpolation
            valid_mask_up = dfcyM[col].iloc[i:].notna()
            xp = dfcyM['Time'].iloc[i:][valid_mask_up].values
            fp = dfcyM[col].iloc[i:][valid_mask_up].values
            x = dfcyD['Time'].iloc[j:].values
            
            if np.all(np.diff(xp) > 0) and len(xp) > 1:
                dfcyD.loc[dfcyD.index[j:], col] = np.interp(x, xp, fp)
        
        # Keep the resulting cycle
        Cycles.append(dfcyD)
    
    dfData_all = pd.concat(Cycles, ignore_index=True)
    Cycles_list = list(enumerate(Cycles))
    
    return dfData_all, Cycles_list


# %% --------------------------------------------------------------------------
# LOAD RAWDATA FILES AND PLOT POSITION AND VOLTAGE
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    logger0.req('Please select the data files to load.')
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)
    MotorFile = filedialog.askopenfilename(title="Select Motor CSV File")
    if MotorFile:
        DaqFile = filedialog.askopenfilename(title="Select Daq Pickle File")
    
    if MotorFile and DaqFile:
        MotorFile, DaqFile = os.path.normpath(MotorFile), os.path.normpath(DaqFile)
        dfData_all, Cycles_list = LoadFiles(MotorFile, DaqFile)
        
        if dfData_all is not None:
            plt.figure(figsize=(12, 6))
            
            # Plot Position vs Time
            plt.subplot(2, 1, 1)
            plt.plot(dfData_all['Time'], dfData_all['Position'], label='Position', color='blue')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (mm)')
            plt.title('Position vs Time')
            plt.grid(True)
            plt.legend()
            
            # Plot Voltage vs Time
            plt.subplot(2, 1, 2)
            plt.plot(dfData_all['Time'], dfData_all['Voltage'], label='Voltage', color='red')
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.title('Voltage vs Time')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        
    else:
        logger0.info('No files were selected. Operation canceled.')
        
        







