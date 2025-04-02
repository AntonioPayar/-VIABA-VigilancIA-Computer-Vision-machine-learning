import joblib
from detectar_matricula import detectar_matricula

class MatriculaDetector:
    def predict(self, image_input):
        return detectar_matricula(image_input)

model = MatriculaDetector()
joblib.dump(model, "models/matricula-detection/matricula_model.joblib")
print("Matricula detection model saved as models/matricula-detection/matricula_model.joblib")
