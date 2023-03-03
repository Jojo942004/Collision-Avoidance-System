import cv2

class TextHandler:

    def __init__(self, bg_color = (0, 0, 0), color = (255, 255, 255), text_type = cv2.FONT_HERSHEY_SIMPLEX, line_type = cv2.LINE_AA):
        self.bg_color = bg_color
        self.color = color
        self.text_type = text_type
        self.line_type = line_type

    def rectangle(self, frame, point1, point2):
        cv2.rectangle(frame, point1, point2, self.bg_color, 3)
        cv2.rectangle(frame, point1, point2, self.color, 1)

    def circle(self, frame, coords, radius=2, color=(0, 0, 0), thickness=2):
        cv2.circle(frame, coords, radius, color, thickness)

    def putText(self, frame, text, coords,):
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.color, 1, self.line_type)
    
    