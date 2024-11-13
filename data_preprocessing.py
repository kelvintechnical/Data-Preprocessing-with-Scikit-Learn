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

# Fill missing values in the 'Age' column with the average (mean) age
data['Age'].fillna(data['Age'].mean(), inplace=True)

# 'data['Age'].mean()' calculates the mean (average) of the 'Age' column to use as the replacement value.

# 'inplace=True' directly updates the 'data' DataFrame without creating a copy.
data['Income'].fillna(data['Income'].mean(), inplace=True)

print(f"\nDataset after handling missing values:\n", data)
#Display the dataset after each handing missing values

label_encoder = LabelEncoder()   # Create a LabelEncoder object, which will convert categories to numbers
data['Gender'] = label_encoder.fit_transform(data['Gender'])

 # Use the LabelEncoder to transform 'Gender' values into numbers

# 'fit_transform()' finds unique values in 'Gender' and assigns a unique number to each category (e.g., 0 for Male, 1 for Female)

data['Purchased'] = label_encoder.fit_transform(data['Purchased'])

print(f"Dataset after encoding categorical data:\n", data)

scaler = StandardScaler() # Create a StandardScaler object, which scales data to have a mean of 0 and standard deviation of 1
data[['Age', 'Income']] = scaler.fit_transform(data[['Age', 'Income']])

# Apply the scaler to the 'Age' and 'Income' columns
# 'fit_transform()' calculates the scaling values (mean and standard deviation) and applies them to 'Age' and 'Income'
# This scales the values in these columns so they have a mean of 0 and are easier for models to use

print(f"\nDataset after scaling numerical data:\n", data)

x = data[['Age', 'Income', 'Gender']]
y = data['Purchased']

# 'X' represents the features (input data) for the model

 # 'y' represents the target variable (output) that we want to predict

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 'train_test_split()' splits X and y into training and testing sets

# 'test_size=0.2' means 20% of the data will go into the test set, and 80% will be used for training
# 'random_state=0' ensures the split is reproducible, so we get the same split each time we run the code


# Display the training and testing sets
print("\nTraining features (X_train):\n", x_train)  # Prints the features (input data) in the training set
print("\nTesting features (X_test):\n", x_test)  # Prints the features in the testing set
print("\nTraining target (y_train):\n", y_train)  # Prints the target (output) in the training set
print("\nTesting target (y_test):\n", y_test)  # Prints the target in the testing set
