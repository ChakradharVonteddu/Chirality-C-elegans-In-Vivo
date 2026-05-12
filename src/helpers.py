import pandas as pd
import numpy as np
import os


CUR_DIR = os.getcwd()
# raw data
ANTERIOR_ANGLE_PATH = os.path.join(CUR_DIR, "raw_data", "anterior.xlsx")
CORTICALFLOW_PATH = os.path.join(CUR_DIR, "raw_data", "corticalflow.xlsx")
DORSAL_ANGLE_PATH = os.path.join(CUR_DIR, "raw_data", "dorsal.xlsx")
DISTANCE_DATAPATH = os.path.join(CUR_DIR, "distances.xlsx")
ANGLES_DATAPATH = os.path.join(CUR_DIR,"angles", "dorsal.xlsx")


def process_rawdf(df: pd.DataFrame, time_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    """
    df = df.drop(index=[0]).reset_index(drop=True)
    time = df[time_name]
    df = df.drop(columns=[time_name])
    assert len(df.columns) == 10 * 2, "data format incorrect when processing raw df"
    df_left = df.iloc[:, 0:10]
    df_right = df.iloc[:, 10:10*2]
    return (df_left, df_right, time)


def get_data():
    # read xlsx files
    dorsal_xls = pd.ExcelFile(DORSAL_ANGLE_PATH)
    anterior_xls = pd.ExcelFile(ANTERIOR_ANGLE_PATH)
    # remove first two rows
    dorsal = pd.read_excel(dorsal_xls, "dorsal")
    anterior = pd.read_excel(anterior_xls, "anterior")
    ABa_dorsal, ABp_dorsal, dorsal_t = process_rawdf(dorsal, "Time(s)")
    ABa_ant, ABp_ant, anterior_t = process_rawdf(anterior, "Time(s)")

    return (ABa_dorsal, ABp_dorsal, dorsal_t, ABa_ant, ABp_ant, anterior_t)

def get_cortical_data():
    corticalflow_xls = pd.ExcelFile(CORTICALFLOW_PATH)
    corticalflow = pd.read_excel(corticalflow_xls, "corticalflow")
    return process_rawdf(corticalflow, "Time (s)")

def column_average(df: pd.DataFrame) -> np.ndarray:
    """
    return a column-wise average of a pandas dataframe
    """
    col_sum = 0
    num_cols = len(df.columns)
    for column in range(num_cols):
        col_sum += df.iloc[:,column].to_numpy()
    col_average = col_sum / num_cols
    return col_average