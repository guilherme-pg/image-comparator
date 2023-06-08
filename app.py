
from flask import Flask, render_template, request
from image_comparator import process_data


app = Flask(__name__)


@app.route("/",  methods=['GET'])
def home():

    return render_template("home.html")


@app.route("/process_form", methods=['POST'])
def process_form():
    image_1 = request.files['image_1']
    image_2 = request.files['image_2']
    # TO IMPROVE: drag and drop - handle the upload in the input

    metrics = process_data(image_1, image_2)

    return render_template("comparison.html", rendering=metrics)

'''
# METRICS GUIDE
@app.route("/metrics")
def metrics():
    return render_template("metrics.html")
'''

if __name__ == "__main__":
    app.run(port=8080, debug=True)
