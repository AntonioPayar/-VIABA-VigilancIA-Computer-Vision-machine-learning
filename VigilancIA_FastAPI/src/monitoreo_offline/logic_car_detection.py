
def comprobar_detecciones_coches(detecciones_trackeadas, x1, y1, x2, y2):
    tolerancia = 5
    for dict_coche in detecciones_trackeadas:
        if (
            abs(dict_coche["x1_imagen_original"] - x1) <= tolerancia
            and abs(dict_coche["y1_imagen_original"] - y1) <= tolerancia
            and abs(dict_coche["x2_imagen_original"] - x2) <= tolerancia
            and abs(dict_coche["y2_imagen_original"] - y2) <= tolerancia
        ):
            return dict_coche["color"], dict_coche["matricula"]
    return "red", "None"  # Color por defecto si no se encuentra coincidencia