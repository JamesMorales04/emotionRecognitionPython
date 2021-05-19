from flask import Flask, jsonify
from emotionPrediction import Prediction

app = Flask(__name__)


@app.route('/api/vid/<string:route>', methods=['GET'])
def videoAnalsis(route):
    predictionVideo = Prediction()
    predictionVideo.videoPrediction(route)
    return "holaa"


@app.route('/api/img/<string:route>', methods=['GET'])
def imageAnalsis(route):
    predictionImage = Prediction()
    predictionImage.imagePrediction(route, None, 0)
    return "hola"


@app.route('/api/live', methods=['GET'])
def liveAnalsis():
    predictionLive = Prediction()
    predictionLive.liveCamPredict()
    return "hola"


@app.route('/api/summary', methods=['GET'])
def summary():
    predictionSummary = Prediction()
    predictionSummary.showAccuracy()
    return "hola"


@app.route('/api/graphic', methods=['GET'])
def graphic():
    predictionGraphic = Prediction()
    return predictionGraphic.trainingGraphics()


if __name__ == "__main__":
    app.run(debug=True)
