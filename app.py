
from flask import Flask, render_template, request
from image_comparator import process_data


app = Flask(__name__)


@app.route("/",  methods=['GET'])
def home():

    return render_template("home.html")


@app.route("/process_form", methods=['POST'])
def process_form():
    if 'image_1' not in request.files or 'image_2' not in request.files:
        return "Erro: Ambas as imagens devem ser enviadas.", 400

    image_1 = request.files['image_1']
    image_2 = request.files['image_2']

    try:
        metrics = process_data(image_1, image_2)
    except Exception as e:
        return f"Erro ao processar imagens: {str(e)}", 500

    return render_template("comparison.html", rendering=metrics)

'''
# METRICS GUIDE
@app.route("/metrics")
def metrics():
    return render_template("metrics.html")
'''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
