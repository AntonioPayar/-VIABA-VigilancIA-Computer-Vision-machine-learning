class Persona:
    def __init__(
        self, id, confidence, color, lower_right, lower_left, x1, y1, x2, y2
    ):
        self.track_id = id
        self.clase = "persona"
        self.confidence = confidence
        self.color = color
        self.lower_right = lower_right
        self.lower_left = lower_left
        self.x1_imagen_original = x1
        self.y1_imagen_original = y1
        self.x2_imagen_original = x2
        self.y2_imagen_original = y2
    
    def setNombre(self, nombre):
        self.nombre = nombre
