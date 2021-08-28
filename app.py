from flask import Flask, render_template, request
import os
import pickle
import numpy as np


#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')

@app.route('/')
def home():
    prediction = 0
    return render_template('index.html', prediction=str(prediction))


@app.route('/getRating', methods=['GET', 'POST'])
def getRating():
    prediction = 0
    if request.method == 'POST':
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        ppg = request.form.get('points')
        fp = request.form.get('fantasy_points')
        fga = request.form.get('field_goals')
        ftm  =request.form.get('free_throws')
        minutes = request.form.get('minutes')

        x = np.array([[ppg, fp, fga, ftm, minutes]])
        scaled_x = scaler.transform(x)

        prediction = model.predict(scaled_x)[0]
        prediction = round(prediction) if prediction <= 100 else 100

    return render_template('index.html', prediction=str(prediction))


# Error handlers.

@app.errorhandler(500)
def internal_error(error):
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
