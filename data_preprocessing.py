# Import necessary libraries
import numpy as np  # NumPy is used for numerical operations, especially with arrays (lists of numbers)
import pandas as pd  # Pandas is used for handling data in a structured format, like tables or spreadsheets
from sklearn.model_selection import train_test_split  # This helps split data into training and testing sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # StandardScaler scales numerical data, LabelEncoder encodes categorical data

# pd.DataFrame: We create a table (DataFrame) using the Pandas library to organize our sample data.
data = pd.DataFrame({
    'Age': [25, 32, 47, np.nan, 35, 52, np.nan, 42],  # Age of individuals, with some missing values (np.nan)
    'Income': [50000, 54000, 61000, 58000, 52000, np.nan, 45000, 60000],  # Income of individuals, also with a missing value
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],  # Gender of individuals
    'Purchased': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']  # Whether the person made a purchase
})