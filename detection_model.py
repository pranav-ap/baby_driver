import numpy as np
from torch import nn
from ultralytics import YOLO
from PIL import Image
import cv2
from collections import Counter


class DetectionModel(nn.Module):
    LABEL = {
        0: "biker",
        1: "car",
        2: "pedestrian",
        3: "trafficLight",
        4: "trafficLight-Green",
        5: "trafficLight-GreenLeft",
        6: "trafficLight-Red",
        7: "trafficLight-RedLeft",
        8: "trafficLight-Yellow",
        9: "trafficLight-YellowLeft",
        10: "truck",
    }

    def __init__(self):
        super().__init__()
        self.model = YOLO("models/best 5.pt")

    def forward(self, x):
        results = self.model.predict(
            x, classes=[1, 4, 5, 6, 7, 8, 9, 10], verbose=False
        )
        return results

    def predict2(self, x):
        results = self.forward(x)

        if len(results) == 0:
            return None

        result = results[0]

        image = result.plot()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # comment for show()
        image = Image.fromarray(image[..., ::-1])

        obstacles = []
        traffic_lights_counter = Counter(red=0, yellow=0, green=0)

        for cls, coord in zip(result.boxes.cls, result.boxes.xyxy):
            if cls in [1, 10]:
                obstacles.append(("vehicle", coord))
            elif cls in [4, 5]:
                traffic_lights_counter["green"] += 1
            elif cls in [6, 7]:
                traffic_lights_counter["red"] += 1
            elif cls in [8, 9]:
                traffic_lights_counter["yellow"] += 1

        return image, obstacles, traffic_lights_counter
