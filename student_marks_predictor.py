import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [20, 35, 50, 65, 80]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()

model.fit(X, y)

predicted_marks = model.predict(pd.DataFrame({"Hours":[6]}))

print("Predicted Marks:", predicted_marks[0])
