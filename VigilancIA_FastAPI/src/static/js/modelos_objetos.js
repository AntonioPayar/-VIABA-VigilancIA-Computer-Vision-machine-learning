// personas.js

export class Persona {
  constructor(id, nombre, color, hora_detectada) {
    this.id = id;
    this.nombre = nombre;
    this.color = color;
    this.ultimo_punto = null;
    this.hora_detectada = hora_detectada;
    this.lista_de_puntos_creados = [];
    this.lista_de_lineas_creadas = [];
  }

  setUltimoPunto(punto) {
    this.ultimo_punto = punto;
  }

  agregarPunto(marker) {
    this.lista_de_puntos_creados.push(marker);
  }

  agregarLinea(linea) {
    this.lista_de_lineas_creadas.push(linea);
  }

  eliminarPuntos() {
    this.lista_de_puntos_creados.forEach((marker) => {
      camera.parentElement.removeChild(marker);
    });
    this.lista_de_puntos_creados = [];
  }

  eliminarLineas() {
    this.lista_de_lineas_creadas.forEach((linea) => {
      camera.parentElement.removeChild(linea);
    });
    this.lista_de_lineas_creadas = [];
  }
}

export function agregarPersonaATabla(persona) {
  const tabla = document.getElementById("tabla-personas");

  const fila = document.createElement("tr");
  fila.setAttribute("id", `persona-fila-${persona.id}`);

  const celdaNombre = document.createElement("td");
  celdaNombre.textContent = persona.nombre;

  const celdaHora = document.createElement("td");
  celdaHora.textContent = persona.hora_detectada;

  const celdaColor = document.createElement("td");
  celdaColor.textContent = persona.color;

  const celdaEliminar = document.createElement("td");
  const botonEliminar = document.createElement("button");
  botonEliminar.textContent = "Eliminar Tracking";
  botonEliminar.onclick = function () {
    persona.eliminarPuntos();
    persona.eliminarLineas();
    fila.remove();
  };
  celdaEliminar.appendChild(botonEliminar);

  fila.appendChild(celdaNombre);
  fila.appendChild(celdaHora);
  fila.appendChild(celdaColor);
  fila.appendChild(celdaEliminar);

  tabla.appendChild(fila);
}
