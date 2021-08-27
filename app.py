from flask import Flask, render_template, request
import os
import pickle
import numpy as np

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# pts = float(input("Enter the player's points per game: "))
# fp = float(input("Enter the player's fantasy points per game: "))
# fga = float(input("Enter the player's attempted field goals per game: "))
# ftm = float(input("Enter the player's made free throws per game: "))
# minutes = float(input("Enter the player's minutes per game: "))

# x = np.array([[pts, fp, fga, ftm, minutes]])
# scaled_x = scaler.transform(x)

# prediction = model.predict(scaled_x)[0]
# prediction = prediction if prediction <= 100 else 100
# print(f"The predicted NBA 2K rating is: {round(prediction)}")


#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')

@app.route('/')
def home():
    return render_template('index.html')



# Error handlers.

@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404


#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.run()
