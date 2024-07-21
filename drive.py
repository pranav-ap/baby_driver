import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
import base64
import cv2
import numpy as np
import random
import threading

from ego_car import EgoCar


# Initialize Socket.IO server
sio = socketio.Server()
app = Flask(__name__)

ego_car = EgoCar()
lanes_c = None

previous_color = None
current_color = None

latest_sid = None
vision_output = None
steer_angle = 0

t = None


def thread_fun(data):
    # Decode image from base64 format

    original = cv2.imdecode(
        np.frombuffer(base64.b64decode(data["image"]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Detect Things

    original_l, original_c, original_r = ego_car.splitter(original)
    detections_c, obstacles, traffic_lights_counter = ego_car.detect_things(original_c)

    # Obstacles

    ego_car.check_obstacle_stop(detections_c, obstacles)

    # Traffic Lights

    global current_color

    if not ego_car.is_blocked_by_obstacle:
        current_color = ego_car.check_traffic_light_stop(traffic_lights_counter)

    # Lane Detection

    global lanes_c
    lanes_c = None

    need_to_stop = (
        ego_car.is_blocked_by_obstacle
        or (previous_color == "red" and current_color != "green")
        or current_color == "red"
    )

    if not need_to_stop and random.random() < 0.5:
        global steer_angle

        lanes_c, curvature, steer_left, offset = ego_car.find_lanes(original_c)
        lanes_c = cv2.cvtColor(lanes_c, cv2.COLOR_RGB2BGR)

        steer_angle = np.clip(
            (offset["offset_val"]) + (curvature / 10000) * 0.01, -1, 1
        )

        steer_angle = float(format(steer_angle, ".6f"))
        steer_angle = abs(steer_angle)

        steer_angle = -steer_angle if steer_left else steer_angle
        # steer_angle = -steer_angle if offset["offset_dir"] == "left" else steer_angle

        # print("steer_angle", steer_angle)

    # show the recieved images on the screen

    global vision_output

    if lanes_c is not None:
        vision_output = lanes_c
    elif detections_c is not None:
        vision_output = np.array(detections_c)
    else:
        vision_output = np.array(original_c)


@sio.on("send_image")
def on_image(sid, data):
    global t

    if t is None or not t.is_alive():
        t = threading.Thread(target=thread_fun, args=(data,), daemon=True)
        t.start()

    global vision_output

    if vision_output is not None:
        cv2.imshow("Vision Output", vision_output)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()


# listen for the event "vehicle_data"
@sio.on("vehicle_data")
def vehicle_command(sid, data):
    if data:
        speed = float(data["velocity"])
        old_steer = float(data["steering_angle"])

        global previous_color, current_color, steer_angle

        need_to_stop = (
            ego_car.is_blocked_by_obstacle
            or (previous_color == "red" and current_color != "green")
            or current_color == "red"
        )

        previous_color = current_color

        brake, throttle = (1, 0) if need_to_stop else (0, 1)

        if old_steer != steer_angle:
            steer_angle = (old_steer * 0.4 + steer_angle * 0.6) / 2
            steer_angle = float(format(steer_angle, ".6f"))
            steer_angle = np.clip(steer_angle, -1, 1)

        speed_limit = 21 if -0.2 < steer_angle < 0.2 else 15
        if speed > speed_limit:
            throttle = 0

        # print("Send to Unity : TBS ", throttle, brake, steer_angle)
        send_control(steer_angle, throttle, brake)


@sio.event
def connect(sid, environ):
    cv2.namedWindow("Vision Output", cv2.WINDOW_NORMAL)
    print("Client connected")
    send_control(0, 0, 0)


# Define a data sending function to send processed data back to unity client
def send_control(steering_angle, throttle, brake):
    sio.emit(
        "control_command",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
            "brake": brake.__str__(),
        },
        skip_sid=True,
    )


@sio.event
def disconnect(sid):
    # implement this function, if disconnected
    print("Client disconnected")


app = socketio.Middleware(sio, app)
# Connect to Socket.IO client
if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
