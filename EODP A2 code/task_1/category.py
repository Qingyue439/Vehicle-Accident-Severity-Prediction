# for step 1.1 there were 3 injury levels, and so i assign them to the level of severity accordingly
import pandas as pd

def task1():
    df = pd.read_csv("accident.csv")

    # filter dataframe to only take 
    df = df[df["SEVERITY"].isin([1, 2, 3, 4])]

    # define severity of each injury level
    severity_map = {
        1. "Fatal"
        2: "Serious",
        3: "Other",
        4: "None"
    }

    # map the numeric values into our description into a new column
    df["SEVERITY_DESC"] = df["SEVERITY"].map(severity_map)   

    result_df = df[["ACCIDENT_NO", "SEVERITY_DESC"]].copy()
    
    return result_df

