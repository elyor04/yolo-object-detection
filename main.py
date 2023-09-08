import cv2 as cv
import numpy as np
import os.path as path
from time import time
from wget import download
from random import randint


class DetectionModel:
    def __init__(self, dataDir: str = "data") -> None:
        self.DATA_DIR = dataDir
        self.MODEL_LINK = "https://pjreddie.com/media/files/yolov3.weights"

        self.WEIGHTS_PATH = path.join(self.DATA_DIR, "yolov3.weights")
        self.CONFIG_PATH = path.join(self.DATA_DIR, "yolov3.cfg")
        self.LABELS_PATH = path.join(self.DATA_DIR, "coco.names")

        self.model = None
        self.classNames = None
        self.outLayersNames = None

        self.colors = dict()

    def prepare(self) -> None:
        if not path.exists(self.WEIGHTS_PATH):
            download(self.MODEL_LINK, self.WEIGHTS_PATH)
        self.model = cv.dnn.readNet(self.WEIGHTS_PATH, self.CONFIG_PATH)

        with open(self.LABELS_PATH, "rt") as f:
            self.classNames = f.read().splitlines()
        self.outLayersNames = self.model.getUnconnectedOutLayersNames()

    def detect(
        self, image: np.ndarray, confThreshold: float = 0.6, nmsThreshold: float = 0.3
    ) -> tuple[list, list, list]:
        blob = cv.dnn.blobFromImage(
            image, 1.0 / 255, (320, 320), swapRB=True, crop=False
        )
        self.model.setInput(blob)
        outputs = self.model.forward(self.outLayersNames)

        height, width = image.shape[:2]
        boxes = []
        classIds = []
        confidences = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]

                if confidence > confThreshold:
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(detection[0] * width - w / 2), int(
                        detection[1] * height - h / 2
                    )
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidences.append(float(confidence))

        indexes = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        boxes = [boxes[i] for i in indexes]
        classIds = [classIds[i] for i in indexes]
        confidences = [confidences[i] for i in indexes]
        return (boxes, classIds, confidences)

    def visualize(
        self,
        image: np.ndarray,
        boxes: list,
        classIds: list,
        scores: list,
        classNames: list,
    ) -> np.ndarray:
        height, width = image.shape[:2]

        for (xmin, ymin, wd, hg), classId, score in zip(boxes, classIds, scores):
            xmax, ymax = xmin + wd, ymin + hg
            name = classNames[classId].capitalize()

            if name in self.colors:
                color = self.colors[name]
            else:
                color = [randint(0, 255) for _ in range(3)]
                while (True not in [(pxl > 210) for pxl in color]) or (
                    color in self.colors.values()
                ):
                    color = [randint(0, 255) for _ in range(3)]
                self.colors[name] = color

            name = f"{name} {round(score * 100)}%"
            font = cv.FONT_HERSHEY_COMPLEX_SMALL

            gts = cv.getTextSize(name, font, 2.0, 2)
            gtx = gts[0][0] + xmin
            gty = gts[0][1] + ymin
            x, y = min(gtx + 3, width), min(gty + 4, height)

            cv.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
            cv.rectangle(image, (xmin, ymin), (x, y), color, -1)
            cv.putText(image, name, (xmin, gty), font, 2.0, (0, 0, 0), 2)

        return image

    def detectAndVisualize(self, image: np.ndarray) -> None:
        boxes, classIds, scores = self.detect(image)
        self.visualize(image, boxes, classIds, scores, self.classNames)


def main() -> None:
    dm = DetectionModel()
    dm.prepare()

    cam = cv.VideoCapture(0)
    prevTime = 0

    while True:
        img = cam.read()[1]
        dm.detectAndVisualize(img)

        currTime = time()
        fps = f"FPS: {round(1 / (currTime - prevTime), 1)}"
        prevTime = currTime

        cv.putText(
            img, fps, (5, 35), cv.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (255, 0, 0), 2
        )
        cv.imshow("Object Detection", img)

        if cv.waitKey(2) == 27:
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
