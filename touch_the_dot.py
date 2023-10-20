import sys
import numpy as np
import cv2
import random
import mediapipe as mp


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def get_index_tip_coordinates(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                index_tip = hand_landmark.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                return int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
        return None, None


class TouchGame:
    def __init__(self):
        self.circle_coords = None
        self.circle_found = True
        self.detector = HandDetector()

    def draw_random_circle(self, frame):
        h, w = frame.shape[:2]
        self.circle_coords = random.randint(
            10, w - 10), random.randint(10, h - 10)
        return self.circle_coords

    def check_touch(self, index_tip):
        if index_tip[0] is not None and self.circle_coords:
            distance = np.sqrt((index_tip[0] - self.circle_coords[0])
                               ** 2 + (index_tip[1] - self.circle_coords[1]) ** 2)
            if distance < 50:
                self.circle_found = True
                return True
        return False

    def game_loop(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open camera")
            sys.exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # We draw a new circle only if the previous one was touched.
            # We must ensure 'circle_found' is only True when a circle is actually found by the index tip.
            if self.circle_found:
                self.circle_coords = self.draw_random_circle(
                    frame)  # Store the new circle's coordinates
                self.circle_found = False  # Reset the flag

            results = self.detector.process_frame(frame)
            frame = self.detector.draw_landmarks(frame, results)
            index_tip = self.detector.get_index_tip_coordinates(frame, results)

            touch_detected = self.check_touch(index_tip)

            if self.circle_coords:  # Ensure there are circle coordinates before trying to draw
                cv2.circle(frame, self.circle_coords, 5, (255, 0, 0), -1)

            cv2.imshow("Frame", frame)

            if touch_detected:
                self.circle_found = True  # Set the flag to draw a new circle in the next iteration

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = TouchGame()
    game.game_loop()
