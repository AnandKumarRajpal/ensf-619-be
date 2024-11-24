# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv('data.csv')
data['GLUCOSE'] = data['GLUCOSE'].str[:-2]
data = data.drop('Serial', axis=1)

# Step 2: Use all features
X = data.drop(columns=["GLUCOSE"])  # Retain all features
y = data["GLUCOSE"]

# Step 3: Optional - Add Interaction Terms
data['Interaction_Current_Potential'] = data['Current (uA)'] * data['Potential(V)']
X['Interaction_Current_Potential'] = data['Interaction_Current_Potential']

# Step 4: Split the data into training and testing sets
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Step 5: Train the Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=2000,       # Number of trees
    max_depth=25,            # Deeper trees
    min_samples_split=2,     # Minimum samples required to split a node
    min_samples_leaf=1,      # Minimum samples required at each leaf node
    max_features='sqrt',     # Use a subset of features for splits
    random_state=42
)

rf_model.fit(X_train, y_train)

import joblib
 # Save the model
joblib.dump(rf_model, 'logreg_model.joblib')
joblib.dump(rf_model, 'model.pkl')