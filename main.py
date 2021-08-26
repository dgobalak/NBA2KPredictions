import pickle
from sklearn import preprocessing
import numpy as np


# Basic script to use ML model. A web application is planned for the future.

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

pts = float(input("Enter the player's points per game: "))
fp = float(input("Enter the player's fantasy points per game: "))
fga = float(input("Enter the player's attempted field goals per game: "))
ftm = float(input("Enter the player's made free throws per game: "))
minutes = float(input("Enter the player's minutes per game: "))

x = np.array([[pts, fp, fga, ftm, minutes]])
scaled_x = scaler.transform(x)

prediction = model.predict(scaled_x)[0]
prediction = prediction if prediction <= 100 else 100
print(f"The predicted NBA 2K rating is: {round(prediction)}")
