// personas.js

export class Persona {
  constructor(id, nombre, color, hora_detectada, contador, caraBase64) {
    this.id = id;
    this.nombre = nombre;
    this.color = color;
    this.ultimo_punto = { x: null, y: null };
    this.hora_detectada = hora_detectada;
    this.contador = contador;
    this.lista_de_puntos_creados = [];
    this.lista_de_lineas_creadas = [];
    this.caraBase64 = caraBase64;
  }

  setUltimoPunto(x, y) {
    this.ultimo_punto = { x, y };
  }

  setcaraBase64(caraBase64) {
    this.caraBase64 = caraBase64;
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

function ocultarTracking(persona) {
  // Ocultar todos los puntos creados
  persona.lista_de_puntos_creados.forEach((punto) => {
    punto.style.display = "none"; // Ocultar el punto
  });

  // Ocultar todas las líneas creadas
  persona.lista_de_lineas_creadas.forEach((linea) => {
    linea.style.display = "none"; // Ocultar la línea
  });

  console.log(`Tracking ocultado para la persona con ID: ${persona.id}`);
}

export function agregarPersonaATabla(persona) {
  const tabla = document.getElementById("tabla-personas");

  // Verificar si la fila ya existe
  let filaExistente = document.getElementById(`persona-fila-${persona.id}`);
  if (filaExistente) {
    // Actualizar la fila existente con la nueva información
    filaExistente.querySelector("td:nth-child(1)").textContent = persona.nombre;
    filaExistente.querySelector("td:nth-child(2) button").onclick =
      function () {
        mostrarImagen(persona.caraBase64);
      };
    filaExistente.querySelector("td:nth-child(3)").textContent =
      persona.hora_detectada;
    filaExistente.querySelector("td:nth-child(4)").textContent = persona.color;
    return;
  }

  // Verificar si hay más de 10 filas (excluyendo el encabezado)
  const filas = tabla.getElementsByTagName("tr");
  if (filas.length > 10) {
    // Reutilizar la fila más antigua (segunda fila, índice 1)
    const filaAntigua = filas[1];
    filaAntigua.setAttribute("id", `persona-fila-${persona.id}`);
    filaAntigua.querySelector("td:nth-child(1)").textContent = persona.nombre;
    filaAntigua.querySelector("td:nth-child(2) button").onclick = function () {
      mostrarImagen(persona.caraBase64);
    };
    filaAntigua.querySelector("td:nth-child(3)").textContent =
      persona.hora_detectada;
    filaAntigua.querySelector("td:nth-child(4)").textContent = persona.color;

    // Actualizar el botón de eliminar para la nueva persona
    const botonEliminar = filaAntigua.querySelector("td:nth-child(5) button");
    botonEliminar.onclick = function () {
      ocultarTracking(persona);
      filaAntigua.remove();
    };
    return;
  }

  // Crear una nueva fila si hay menos de 10 filas
  const fila = document.createElement("tr");
  fila.setAttribute("id", `persona-fila-${persona.id}`);

  const celdaNombre = document.createElement("td");
  celdaNombre.textContent = persona.nombre;

  const celdaImagen = document.createElement("td");
  const botonImagen = document.createElement("button");
  botonImagen.textContent = "Ver Imagen";
  botonImagen.onclick = function () {
    mostrarImagen(persona.caraBase64);
  };
  celdaImagen.appendChild(botonImagen);

  const celdaHora = document.createElement("td");
  celdaHora.textContent = persona.hora_detectada;

  const celdaColor = document.createElement("td");
  celdaColor.textContent = persona.color;

  const celdaEliminar = document.createElement("td");
  const botonEliminar = document.createElement("button");
  botonEliminar.textContent = "Eliminar Tracking";
  botonEliminar.onclick = function () {
    ocultarTracking(persona);
    fila.remove();
  };
  celdaEliminar.appendChild(botonEliminar);

  fila.appendChild(celdaNombre);
  fila.appendChild(celdaImagen);
  fila.appendChild(celdaHora);
  fila.appendChild(celdaColor);
  fila.appendChild(celdaEliminar);

  tabla.appendChild(fila);
}

function mostrarImagen(imagenBase64) {
  if (imagenBase64) {
    const nuevaVentana = window.open("", "_blank");
    nuevaVentana.document.write(
      `<img src="data:image/jpeg;base64,${imagenBase64}" alt="Imagen de la persona" style="max-width:100%; height:auto;">`
    );
  } else {
    alert("No hay imagen disponible para esta persona.");
  }
}
