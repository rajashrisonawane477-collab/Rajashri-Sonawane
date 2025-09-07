# Rajashri-Sonawane
Task 1
# Linear Regression to Predict House Prices with Visualization

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Data
data = pd.DataFrame({
    'sqft': [2000, 1500, 2500, 1800, 2200],
    'bedrooms': [3, 2, 4, 3, 4],
    'bathrooms': [2, 1, 3, 2, 3],
    'price': [500000, 350000, 600000, 400000, 550000]
})

print("Data Preview:\n", data)

# Step 3: Split Features and Target
X = data[['sqft', 'bedrooms', 'bathrooms']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error:", mse)
print("R-squared Score:", r2)

# Step 7: Predict Price of a New House
new_house = np.array([[2500, 4, 3]])  # Example: 2500 sqft, 4 bedrooms, 3 bathrooms
predicted_price = model.predict(new_house)
print("\nPredicted Price for New House:", predicted_price[0])

# Step 8: Visualization - Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
