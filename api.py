from flask import Flask
from flask import request, send_file
from run import process_run
import base64

app = Flask(__name__)

@app.route("/image_synthesis", methods=['POST'])
def image_synthesis():
    data = request.json
    response = process_run(
        data["prompt"], 
        data["steps"],
        data["width"], 
        data["height"], 
        data["images"], 
        data["scale"]
    )
    
    return response

app.run(host="0.0.0.0", port=8001)