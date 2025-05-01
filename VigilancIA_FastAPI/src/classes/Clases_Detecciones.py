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
        self.caraBase64 = None
        self.sumaContador()

    def sumaContador(self):
        self.contador = self.contador + 1

    def setNombre(self, nombre):
        self.nombre = nombre
        self.sumaContador()

    def setContador(self, contador):
        self.contador = contador

    def setCaraBase64(self, img_cara):
        self.caraBase64 = img_cara

    def actualizarPersona(self, persona_antigua):
        self.contador = persona_antigua.contador
        self.setNombre(persona_antigua.nombre)
        self.caraBase64 = persona_antigua.caraBase64


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
        self.contador = 0
        self.MatriculaBase64 = None
        self.sumaContador()

    def sumaContador(self):
        self.contador = self.contador + 1

    def setMatricula(self, matricula):
        self.matricula = matricula
        self.sumaContador()

    def setContador(self, contador):
        self.contador = contador

    def setMatricula64(self, img_cara):
        self.MatriculaBase64 = img_cara

    def actualizarVehiculo(self, coche_antiguo):
        self.contador = coche_antiguo.contador
        self.setMatricula(coche_antiguo.matricula)
        self.MatriculaBase64 = coche_antiguo.MatriculaBase64
