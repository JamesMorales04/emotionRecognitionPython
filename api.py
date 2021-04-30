from flask import Flask, jsonify
from emotionPrediction import Prediction

app = Flask(__name__)


@app.route('/api/vid/<string:route>', methods=['GET'])
def videoAnalsis(route):
    predictionVideo=Prediction()
    predictionVideo.videoPrediction(route)
    return "hola"


@app.route('/api/img/<string:route>', methods=['GET'])
def imageAnalsis(route):
    predictionImage=Prediction()
    predictionImage.imagePrediction(route,None)
    return "hola"


@app.route('/api/live', methods=['GET'])
def liveAnalsis(route):
    predictionLive=Prediction()
    predictionLive.liveCamPredict(route)
    return "hola"


if __name__ == "__main__":
    app.run(debug=True)
