import pandas as pd
import json
import matplotlib.pyplot as plt
import os

# create file pathway
os.makedirs('output', exist_ok=True)

# load data
vehicles = pd.read_csv('datasets/vehicle.csv')
accidents = pd.read_csv('datasets/accident.csv')
person = pd.read_csv('datasets/person.csv')
filtered_vehicle = pd.read_csv('datasets/filtered_vehicle.csv')

# merge the target columns
accident_subset = accidents[['ACCIDENT_NO', 'SEVERITY', 'ACCIDENT_DATE']]
filtered_merged_1 = pd.merge(filtered_vehicle, accident_subset, on='ACCIDENT_NO', how='inner')


# save result as a new file
filtered_merged_1.to_csv('output/vehicle_severity.csv', index=False)

