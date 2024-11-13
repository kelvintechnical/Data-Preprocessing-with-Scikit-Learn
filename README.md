<h1>Python Data Preprocessing with Scikit-Learn</h1>

<img src="img/data-preprocessing-logo.png" alt="Data Preprocessing Logo">

<a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.7%2B-blue" alt="Python Badge">
</a>

<p><code>Data Preprocessing</code> is an essential step for preparing data in machine learning projects. This project demonstrates how to handle missing values, encode categorical data, scale numerical data, and split the dataset into training and testing sets, all with Python's Scikit-Learn library.</p>

<img src="img/demo.gif" alt="GIF demo">

<h2>Usage</h2>
<pre><code>
Usage: data_preprocessing.py [OPTIONS]

  Perform essential data preprocessing steps, including handling missing values, encoding categorical data, scaling numerical values, and splitting the data.
  Developed by [Your Name] -> (Github: [YourUsername])

Options:
  -help, -h   Show this message and exit.
</code></pre>

<h2>Installation</h2>
<ol>
    <li>Clone the repository:
        <pre><code>$ git clone https://github.com/yourusername/data-preprocessing-with-scikit-learn.git
$ cd data-preprocessing-with-scikit-learn</code></pre>
    </li>
    <li>Install dependencies:
        <pre><code>$ pip install -r requirements.txt</code></pre>
    </li>
</ol>

<h2>Data Preprocessing Steps</h2>
<ol>
    <li><strong>Handling Missing Values</strong>
        <ul>
            <li>Identify missing values in the data.</li>
            <li>Fill missing values in numerical columns (like "Age" and "Income") with the column's mean, allowing for smoother, more complete data.</li>
        </ul>
    </li>
    <li><strong>Encoding Categorical Data</strong>
        <ul>
            <li>Convert categorical text data into numerical data. For example, "Gender" values ("Male" and "Female") are encoded as numbers.</li>
            <li>Similarly, encode "Yes" and "No" values in the "Purchased" column to make the data compatible with machine learning algorithms.</li>
        </ul>
    </li>
    <li><strong>Scaling Numerical Data</strong>
        <ul>
            <li>Standardize numerical data to make sure values have a mean of 0 and a standard deviation of 1. This helps models interpret the data evenly and improves model performance.</li>
        </ul>
    </li>
    <li><strong>Splitting the Data</strong>
        <ul>
            <li>Divide the preprocessed data into training and testing sets. The training set allows the model to learn from the data, while the testing set evaluates how well the model performs on unseen data.</li>
        </ul>
    </li>
</ol>

<h2>Example Code</h2>
<pre><code># Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Sample dataset
data = pd.DataFrame({
    'Age': [25, 32, 47, np.nan, 35, 52, np.nan, 42],
    'Income': [50000, 54000, 61000, 58000, 52000, np.nan, 45000, 60000],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
    'Purchased': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
})

# Handle missing values
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Income'].fillna(data['Income'].mean(), inplace=True)

# Encode categorical data
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Purchased'] = label_encoder.fit_transform(data['Purchased'])

# Scale numerical data
scaler = StandardScaler()
data[['Age', 'Income']] = scaler.fit_transform(data[['Age', 'Income']])

# Split data into features and target
X = data[['Age', 'Income', 'Gender']]
y = data['Purchased']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
</code></pre>

<h2>How to Contribute</h2>
<ol>
    <li>Clone the repo and create a new branch:
        <pre><code>$ git clone https://github.com/yourusername/data-preprocessing-with-scikit-learn.git
$ git checkout -b new-feature</code></pre>
    </li>
    <li>Make changes, run, and test them.</li>
    <li>Submit a Pull Request with a detailed description of your changes.</li>
</ol>

<h2>Acknowledgments</h2>
<ul>
    <li>Inspired by the data preprocessing methods commonly used in machine learning.</li>
    <li>The Scikit-Learn library provides powerful tools for these preprocessing techniques.</li>
</ul>

<h2>Connect with Me</h2>
<p>Follow me on social media for more machine learning projects and insights!</p>

<a href="https://www.instagram.com/kelvinintech" target="_blank" style="text-decoration: none;">
   <button style="background-color: #E4405F; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px;">
       Follow Me on Instagram
   </button>
</a>

<a href="https://x.com/kelvintechnical" target="_blank" style="text-decoration: none;">
   <button style="background-color: #1DA1F2; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px;">
       Follow Me on X
   </button>
</a>

<h2>Donations</h2>
<p>This is free, open-source software. If you'd like to support my future projects or say thanks, consider donating BTC at <code>1FnJ8hRRNUtUavngswUD21dsFNezYLX5y9</code>.</p>
