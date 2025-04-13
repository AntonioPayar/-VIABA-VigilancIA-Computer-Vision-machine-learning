
class Persona:
    def __init__(self, id, confidence, color, lower_right, lower_left):
        self.track_id = id
        self.clase = "persona"
        self.confidence = confidence
        self.color = color
        self.lower_right = lower_right
        self.lower_left = lower_left