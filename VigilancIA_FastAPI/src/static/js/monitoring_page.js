import { Persona, agregarPersonaATabla } from "./logica_Persona.js";
import { Coche, agregarCocheATabla } from "./logica_Coche.js";
// Variable global para guardar el último punto seleccionado
let personas = [];
let coches = [];
let contador_personas = 0;
let contador_coches = 0;
let cameraYoloImage = document.getElementById("camera_yolo");

async function eliminarTodoPlano() {
  const plano = document.getElementById("plano");
  const divs = plano.querySelectorAll("div");

  divs.forEach((div) => {
    plano.removeChild(div);
  });
}

function iniciarCamara() {
  const video = document.getElementById("camera1");

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        video.srcObject = stream;
        video.play();
        video.addEventListener("loadedmetadata", () => {
          comenzarCaptura(video);
        });
      })
      .catch((error) => {
        console.error("No se pudo acceder a la cámara:", error);
        alert("No se pudo acceder a la cámara.");
      });
  } else {
    alert("Tu navegador no soporta acceso a la cámara.");
  }
}

// Función para dibujar las bounding boxes sobre el canvas
function posicionarMapa(context, boundingBoxes) {
  if (boundingBoxes && boundingBoxes.length > 0) {
    boundingBoxes.forEach((deteccion) => {
      // Verificamos si la persona detectada ya existe
      let persona;
      let last_point;
      if (!personas[deteccion.track_id]) {
        contador_personas++;

        persona = new Persona(
          deteccion.track_id,
          deteccion.nombre,
          deteccion.color,
          obtenerHoraActual(),
          contador_personas,
          deteccion.caraBase64
        );

        last_point = { x: null, y: null };
      } else {
        persona = personas[deteccion.track_id];
        persona.setcaraBase64(deteccion.caraBase64);
        last_point = persona.ultimo_punto;
      }

      const lowerLeftX = deteccion.lower_left.x;
      const lowerLeftY = deteccion.lower_left.y;
      const lowerRightX = deteccion.lower_right.x;
      const lowerRightY = deteccion.lower_right.y;

      const { marker, linea } = pintarPunto(
        "camera2",
        lowerLeftX,
        lowerLeftY,
        persona.color,
        last_point
      );

      persona.setUltimoPunto(lowerLeftX, lowerLeftY); // Actualizar el último punto en la persona
      persona.agregarPunto(marker); // Guardar referencia al marcador en la persona
      persona.agregarLinea(linea); // Guardar referencia a la línea en la persona
      personas[deteccion.track_id] = persona; // Guardar la persona en el array

      agregarPersonaATabla(persona);
    });
  } else {
    console.log("No se recibieron bounding boxes o el array está vacío.");
  }
}

function posicionarMapaCoche(context, boundingBoxes) {
  if (boundingBoxes && boundingBoxes.length > 0) {
    boundingBoxes.forEach((deteccion) => {
      // Verificamos si la persona detectada ya existe
      let coche;
      let last_point;
      if (!coches[deteccion.track_id]) {
        contador_coches++;
        coche = new Coche(
          deteccion.track_id,
          deteccion.matricula,
          deteccion.color,
          obtenerHoraActual(),
          contador_coches,
          deteccion.MatriculaBase64
        );
        last_point = { x: null, y: null };
      } else {
        coche = coches[deteccion.track_id];
        coche.setMatriculaBase64(deteccion.MatriculaBase64);
        last_point = coche.ultimo_punto;
      }

      const lowerLeftX = deteccion.lower_left.x;
      const lowerLeftY = deteccion.lower_left.y;
      const lowerRightX = deteccion.lower_right.x;
      const lowerRightY = deteccion.lower_right.y;

      const { marker, linea } = pintarPunto(
        "camera2",
        lowerLeftX,
        lowerLeftY,
        coche.color,
        last_point
      );

      coche.setUltimoPunto(lowerLeftX, lowerLeftY); // Actualizar el último punto en la coche
      coche.agregarPunto(marker); // Guardar referencia al marcador en la coche
      coche.agregarLinea(linea); // Guardar referencia a la línea en la coche
      coche[deteccion.track_id] = coche; // Guardar la coche en el array

      agregarCocheATabla(coche);
    });
  } else {
    console.log("No se recibieron bounding boxes o el array está vacío.");
  }
}

function comenzarCaptura(video) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  function capturarYEnviar() {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64Image = canvas.toDataURL("image/jpeg", 0.8).split(",")[1]; // Reducir la calidad a 0.8

    // Enviar la imagen al servidor
    enviarFotogramas(
      { image_base64: base64Image },
      "http://localhost:8000/getCamaraBounding"
    )
      .then((response) => {
        const boundingBoxes = response.bounding_boxes;
        const imageBounding = response.image_bounding;
        const bounding_boxes_coches = response.bounding_boxes_coches;

        const image = new Image();
        image.src = `data:image/jpeg;base64,${imageBounding}`;

        image.onload = () => {
          // Reemplaza la imagen actual en el elemento camera_yolo
          cameraYoloImage.src = image.src;

          context.drawImage(image, 0, 0); // Dibuja la imagen en el canvas (si es necesario)
          if (boundingBoxes && boundingBoxes.length > 0) {
            posicionarMapa(context, boundingBoxes);
          }
          if (bounding_boxes_coches && bounding_boxes_coches.length > 0) {
            posicionarMapaCoche(context, bounding_boxes_coches);
          }
        };
      })
      .catch((error) => {
        console.error("Error al recibir los bounding boxes:", error);
      });

    requestAnimationFrame(capturarYEnviar); // Usar requestAnimationFrame para FPS máximos
  }

  function enviarFotogramas(datos, url) {
    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(datos),
    }).then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json(); // Espera un array de objetos JSON
    });
  }

  capturarYEnviar();
}

iniciarCamara();

function pintarLinea(cameraId, x1, y1, x2, y2, color) {
  const camera = document.getElementById(cameraId);

  if (!camera) {
    console.error(`No se encontró una cámara con el ID: ${cameraId}`);
    return;
  }

  const container = camera.parentElement;
  container.style.position = "relative";

  // Calcular distancia y ángulo
  const deltaX = x2 - x1;
  const deltaY = y2 - y1;
  const length = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
  const angle = Math.atan2(deltaY, deltaX) * (180 / Math.PI); // en grados

  // Crear la línea como un div rotado
  const linea = document.createElement("div");
  linea.style.position = "absolute";
  linea.style.height = `12px`;
  linea.style.width = `${length}px`;
  linea.style.backgroundColor = color;
  linea.style.left = `${x1}px`;
  linea.style.top = `${y1}px`;
  linea.style.transform = `rotate(${angle}deg)`;
  linea.style.transformOrigin = "0 0"; // Anclar al inicio de la línea

  container.appendChild(linea);
  return linea; // Devolver la línea creada
}

function pintarPunto(cameraId, x, y, color, ultimo_punto) {
  const camera = document.getElementById(cameraId);

  if (!camera) {
    console.error(`No se encontró una cámara con el ID: ${cameraId}`);
    return;
  }

  // Crear un marcador (punto)
  const marker = document.createElement("div");
  marker.style.position = "absolute";
  marker.style.width = "15px"; // Tamaño del marcador
  marker.style.height = "15px";
  marker.style.backgroundColor = color;
  marker.style.borderRadius = "50%";
  marker.style.left = `${x}px`; // Posición relativa al contenedor
  marker.style.top = `${y}px`; // Posición relativa al contenedor
  marker.style.transform = "translate(-50%, -50%)"; // Centrar el marcador

  // Asegurarse de que el contenedor tenga posición relativa
  const container = camera.parentElement;
  container.style.position = "relative";

  // Agregar el marcador al contenedor de la imagen
  container.appendChild(marker);

  let linea;
  // Pintar línea desde el último punto al nuevo
  if (ultimo_punto.x !== null && ultimo_punto.y !== null) {
    linea = pintarLinea(cameraId, ultimo_punto.x, ultimo_punto.y, x, y, color);
  }
  return { marker, linea }; // Correcto
}

function obtenerHoraActual() {
  const ahora = new Date();
  const horas = ahora.getHours();
  const minutos = ahora.getMinutes();
  const segundos = ahora.getSeconds();
  const milisegundos = ahora.getMilliseconds();

  return `${horas.toString().padStart(2, "0")}:${minutos
    .toString()
    .padStart(2, "0")}:${segundos.toString().padStart(2, "0")}.${milisegundos
    .toString()
    .padStart(3, "0")}`;
}

// Exponer la función al ámbito global
window.eliminarTodoPlano = eliminarTodoPlano;
