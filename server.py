import cv2
import logging
import numpy as np
import threading

from flask import Flask, request, jsonify

# init flask app
app = Flask(__name__)


def setup_logging() -> logging.Logger:
    """Creates a logger"""
    lg = logging.getLogger("app_logger")

    if not lg.handlers:
        logging.basicConfig(level=logging.DEBUG)
        log_format = logging.Formatter("%(levelname)s - %(message)s")
        file_handler = logging.FileHandler("app.log")
        file_handler.setFormatter(log_format)

        lg = logging.getLogger("app_logger")
        lg.addHandler(file_handler)
        lg.propagate = False

    return lg


# create logger
lg = setup_logging()

# --- CONFIGURATION ---
MODEL_PATH = "yolo/yolov8n_br_flip_blur_results/weights/best.onnx"
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

net = cv2.dnn.readNetFromONNX(MODEL_PATH)
infer_lock = threading.Lock()
state_lock = threading.Lock()

classes = ["box", "orange", "green", "blue", "black"]
robot_targets = {"orange": None, "green": None, "blue": None, "black": None}


def process_detections(outputs, img_width, img_height):
    data = outputs[0]
    if data.shape[0] < data.shape[1]:
        data = data.T

    boxes, confidences, class_ids = [], [], []

    x_factor = img_width / INPUT_WIDTH
    y_factor = img_height / INPUT_HEIGHT

    for row in data:
        scores = row[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONF_THRESHOLD:
            cx, cy, w, h = row[0:4]
            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():  # type: ignore
            results.append(
                {
                    "label": classes[class_ids[i]],
                    "confidence": confidences[i],
                    "box": boxes[i],
                }
            )
    return results


def red_box_group(label):
    """Returns true if given label is in a chain that can see red box"""
    checked = set()
    current = label

    while current is not None and current not in checked:
        if current == "box":
            return True
        checked.add(current)
        current = robot_targets.get(current)

    return False


def get_bot_color(route_color=None):
    """Get bot color for backwards compatibility"""
    if route_color:
        return route_color.strip().lower()

    # Backward compatibility with the old client style.
    if "color" in request.form:
        return request.form["color"].strip().lower()
    if "color" in request.files:
        return request.files["color"].read().decode("utf-8").strip().lower()

    return None


# Per robot endpoints:
#   /infer/orange
#   /infer/green
#   /infer/blue
#   /infer/black
@app.route("/infer", methods=["POST"])
@app.route("/infer/<bot_color>", methods=["POST"])
def infer(bot_color=None):
    try:
        bot_color = get_bot_color(bot_color)
        if bot_color not in robot_targets:
            return jsonify({"error": f"unknown or missing bot color: {bot_color}"}), 400

        # read image
        file = request.files["image"].read()
        np_img = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        h, w, _ = frame.shape  # type: ignore

        blob = cv2.dnn.blobFromImage(
            frame,  # type: ignore
            1 / 255.0,
            (INPUT_WIDTH, INPUT_HEIGHT),
            swapRB=True,
            crop=False,  # type: ignore
        )

        # safety, only one robot can use shared opencv DNN object at a time
        with infer_lock:
            net.setInput(blob)
            outputs = net.forward()

        detections = process_detections(outputs, w, h)
        lg.info(f"{bot_color} detected: {detections}")

        # Initialize target
        target = None
        target_label = None

        with state_lock:
            for d in detections:
                label = d["label"]

                # Do not let a robot follow itself.
                if label == bot_color:
                    continue

                # if red box is found, that is the target
                if label == "box":
                    target = d
                    target_label = "box"
                    break
                elif red_box_group(label):
                    target = d
                    target_label = label

            # save the target for the current robot, can be none if it has no target
            robot_targets[bot_color] = target_label  # type: ignore

        if target:
            # calculate steering
            box_center_x = target["box"][0] + (target["box"][2] / 2)
            error_x = (box_center_x / w) - 0.5

            # send steering info to robot
            return jsonify(
                {
                    "status": "found",
                    "robot": bot_color,
                    "label": target["label"],
                    "steering_bias": round(error_x, 3),
                    "box_area": target["box"][2] * target["box"][3],
                }
            )

        # if no target is found, tell the robot to search
        return jsonify({"status": "searching", "robot": bot_color, "label": None})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/robots", methods=["GET"])
def robots():
    with state_lock:
        return jsonify(robot_targets)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
