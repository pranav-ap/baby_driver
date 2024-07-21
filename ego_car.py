from PIL import Image
from detection_model import DetectionModel
from lane_follower.lane_follower import (
    LaneFollowerClassic,
)


WIDTH_SPACE = 160


class EgoCar:
    def __init__(self):
        self.detector = DetectionModel()
        self.lane_follower_classic = LaneFollowerClassic()
        self.is_blocked_by_obstacle = False

    def splitter(self, image):
        image = Image.fromarray(image)

        single_width = image.width // 3

        image_l = image.crop((0, 0, single_width, image.height))
        image_c = image.crop((single_width, 0, single_width * 2, image.height))
        image_r = image.crop((single_width * 2, 0, single_width * 3, image.height))

        return image_l, image_c, image_r

    def detect_things(self, image):
        # detect the traffic lights and vehicles
        det_result = self.detector.predict2(image)
        return det_result

    def check_traffic_light_stop(self, traffic_lights_counter):
        most_common_color = traffic_lights_counter.most_common(1)
        color, count = most_common_color[0]

        if count != 0:
            return color

        return None

    def check_obstacle_stop(self, image, obstacles):
        self.is_blocked_by_obstacle = False

        for cls, obs_box in obstacles:
            # find bbox rectangle center x, y
            x1, y1, x2, y2 = obs_box
            x, y = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            if y2 > image.height - 110 and WIDTH_SPACE - 25 < x < WIDTH_SPACE * 2 + 25:
                self.is_blocked_by_obstacle = True
                break

    def find_lanes(self, image):
        result = self.lane_follower_classic.predict(image)
        return result
