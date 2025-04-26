class Persona:
    def __init__(self, id, confidence, color, lower_right, lower_left, x1, y1, x2, y2):
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
        self.contador = 0
        self.img_cara = None
        self.sumaContador()

    def sumaContador(self):
        self.contador = self.contador + 1

    def setNombre(self, nombre):
        self.nombre = nombre
        self.sumaContador()

    def setContador(self, contador):
        self.contador = contador


class Vehiculo:
    def __init__(self, id, confidence, color, lower_right, lower_left, x1, y1, x2, y2):
        self.track_id = id
        self.clase = "vehiculo"
        self.confidence = confidence
        self.color = color
        self.lower_right = lower_right
        self.lower_left = lower_left
        self.x1_imagen_original = x1
        self.y1_imagen_original = y1
        self.x2_imagen_original = x2
        self.y2_imagen_original = y2

    def setMatricula(self, matricula):
        self.matricula = matricula
