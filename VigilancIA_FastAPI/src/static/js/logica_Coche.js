// monitoring_page.js

export class Coche {
  constructor(id, matricula, color, hora_detectada, contador, matriculaBase64) {
    this.id = id;
    this.matricula = matricula;
    this.color = color;
    this.ultimo_punto = { x: null, y: null };
    this.hora_detectada = hora_detectada;
    this.contador = contador;
    this.lista_de_puntos_creados = [];
    this.lista_de_lineas_creadas = [];
    this.MatriculaBase64 = matriculaBase64;
  }

  setUltimoPunto(x, y) {
    this.ultimo_punto = { x, y };
  }

  setMatriculaBase64(caraBase64) {
    this.MatriculaBase64 = caraBase64;
  }

  agregarPunto(marker) {
    this.lista_de_puntos_creados.push(marker);
  }

  agregarLinea(linea) {
    this.lista_de_lineas_creadas.push(linea);
  }

  eliminarPuntos() {
    this.lista_de_puntos_creados.forEach((marker) => {
      marker.parentElement.removeChild(marker);
    });
    this.lista_de_puntos_creados = [];
  }

  eliminarLineas() {
    this.lista_de_lineas_creadas.forEach((linea) => {
      linea.parentElement.removeChild(linea);
    });
    this.lista_de_lineas_creadas = [];
  }
}

function ocultarTracking(coche) {
  coche.lista_de_puntos_creados.forEach((punto) => {
    punto.style.display = "none";
  });

  coche.lista_de_lineas_creadas.forEach((linea) => {
    linea.style.display = "none";
  });

  console.log(`Tracking ocultado para el coche con ID: ${coche.id}`);
}

export function agregarCocheATabla(coche) {
  const tabla = document.getElementById("tabla-vehiculos");

  // Verificar si la fila ya existe
  let filaExistente = document.getElementById(`coche-fila-${coche.id}`);
  if (filaExistente) {
    filaExistente.querySelector("td:nth-child(1)").textContent =
      coche.matricula;
    filaExistente.querySelector("td:nth-child(2) button").onclick =
      function () {
        mostrarImagen(coche.MatriculaBase64);
      };
    filaExistente.querySelector("td:nth-child(3)").textContent =
      coche.hora_detectada;
    filaExistente.querySelector("td:nth-child(4)").textContent = coche.color;
    return;
  }

  // Verificar si hay más de 10 filas (excluyendo encabezado)
  const filas = tabla.getElementsByTagName("tr");
  if (filas.length > 10) {
    const filaAntigua = filas[1];
    filaAntigua.setAttribute("id", `coche-fila-${coche.id}`);
    filaAntigua.querySelector("td:nth-child(1)").textContent = coche.matricula;
    filaAntigua.querySelector("td:nth-child(2) button").onclick = function () {
      mostrarImagen(coche.MatriculaBase64);
    };
    filaAntigua.querySelector("td:nth-child(3)").textContent =
      coche.hora_detectada;
    filaAntigua.querySelector("td:nth-child(4)").textContent = coche.color;

    const botonEliminar = filaAntigua.querySelector("td:nth-child(5) button");
    botonEliminar.onclick = function () {
      ocultarTracking(coche);
      filaAntigua.remove();
    };
    return;
  }

  // Crear nueva fila
  const fila = document.createElement("tr");
  fila.setAttribute("id", `coche-fila-${coche.id}`);

  const celdaMatricula = document.createElement("td");
  celdaMatricula.textContent = coche.matricula;

  const celdaImagen = document.createElement("td");
  const botonImagen = document.createElement("button");
  botonImagen.textContent = "Ver Imagen";
  botonImagen.onclick = function () {
    mostrarImagen(coche.MatriculaBase64);
  };
  celdaImagen.appendChild(botonImagen);

  const celdaHora = document.createElement("td");
  celdaHora.textContent = coche.hora_detectada;

  const celdaColor = document.createElement("td");
  celdaColor.textContent = coche.color;

  const celdaEliminar = document.createElement("td");
  const botonEliminar = document.createElement("button");
  botonEliminar.textContent = "Eliminar Tracking";
  botonEliminar.onclick = function () {
    ocultarTracking(coche);
    fila.remove();
  };
  celdaEliminar.appendChild(botonEliminar);

  fila.appendChild(celdaMatricula);
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
      `<img src="data:image/jpeg;base64,${imagenBase64}" alt="Imagen del coche" style="max-width:100%; height:auto;">`
    );
  } else {
    alert("No hay imagen disponible para este coche.");
  }
}
