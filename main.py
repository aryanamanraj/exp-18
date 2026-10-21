
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Create sample student dataset
data = {
    "Study_Hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        2, 3, 4, 5, 6, 7, 8, 9, 10, 1
    ],
    "Attendance": [
        50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
        60, 65, 70, 75, 80, 85, 90, 95, 98, 45
    ],
    "Previous_Score": [
        40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
        42, 48, 53, 58, 63, 68, 73, 78, 88, 35
    ],
    "Final_Score": [
        35, 40, 45, 50, 55, 60, 68, 75, 82, 90,
        42, 46, 52, 57, 64, 70, 76, 84, 92, 30
    ]
}

df = pd.DataFrame(data)

print("Student Performance Dataset:")
print(df)

# Step 2: Select input features and target
X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Final_Score"]

# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Step 5: Predict test scores
y_pred = model.predict(X_test)

print("\nActual Scores:")
print(y_test.to_numpy())

print("\nPredicted Scores:")
print(y_pred.round(2))

# Step 6: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 3))

# Step 7: Predict performance for a new student
new_student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [80],
    "Previous_Score": [70]
})

predicted_score = model.predict(new_student)[0]

print("\nNew Student Details:")
print(new_student)

print("Predicted Final Score:", round(predicted_score, 2))

# Step 8: Plot actual vs predicted scores
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Student Performance Prediction")
plt.grid(True)
plt.show()
