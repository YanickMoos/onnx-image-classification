from flask import Flask, request, jsonify, send_file
import numpy as np
import onnxruntime
import cv2
import json
import os
import sys

app = Flask(__name__, static_url_path='/', static_folder='web')

# === Modellpfad ermitteln ===
script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = sys.argv[1] if len(sys.argv) > 1 else "efficientnet-lite4-11.onnx"
model_path = os.path.join(script_dir, model_filename)

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")
print(f"⚙️ Using model: {model_path}")

# === ONNX Modell laden ===
ort_session = onnxruntime.InferenceSession(model_path)

# === Labels laden ===
labels_path = os.path.join(script_dir, "labels_map.txt")
if not os.path.isfile(labels_path):
    raise FileNotFoundError(f"❌ Labels file not found: {labels_path}")

with open(labels_path, "r") as f:
    labels = json.load(f)

# === Bildvorverarbeitung ===
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    return cv2.resize(img, (w, h), interpolation=inter_pol)

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    return img[top:bottom, left:right]

# === Web-Endpunkte ===
@app.route("/")
def indexPage():
    return send_file("web/index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    content = request.files.get('0', '').read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = pre_process_edgetpu(img, (224, 224, 3))
    img_batch = np.expand_dims(img, axis=0)

    results = ort_session.run(["Softmax:0"], {"images:0": img_batch})[0]
    top5_indices = reversed(results[0].argsort()[-5:])
    result_list = [{"class": labels[str(i)], "value": float(results[0][i])} for i in top5_indices]

    return jsonify(result_list)

# === App starten ===
if __name__ == "__main__":
    app.run(debug=True)
