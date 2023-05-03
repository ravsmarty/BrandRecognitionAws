import sys, os
from BrandRecognition.pipeline.training_pipeline import TrainPipeline
from BrandRecognition.exception import BrandException
from BrandRecognition.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from ultralytics import YOLO
from BrandRecognition.constant.application import APP_HOST, APP_PORT


app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
@cross_origin()
def predictRoute():
    try:
        image = request.json["image"]
        decodeImage(image, clApp.filename)
        model = YOLO("custom_model_weights/best.pt")
        model.predict(
            source="data/inputImage.jpg",
            show=False,
            save=True,
            show_labels=True,
            show_conf=True,
            conf=0.5,
            save_txt=False,
            save_crop=False,
            line_thickness=2,
        )
        opencodedbase64 = encodeImageIntoBase64("runs/detect/predict/inputImage.jpg")
        result = {"image": opencodedbase64.decode("utf-8")}
        os.system("rm -rf runs")

    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)
