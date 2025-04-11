// Variable global para guardar el último punto seleccionado
let ultimoPunto = { x: null, y: null };

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

function comenzarCaptura(video) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  function capturarYEnviar() {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64Image = canvas.toDataURL("image/jpeg").split(",")[1];

    enviarFotogramas({ image_base64: base64Image }, "http://localhost:8000/getCamaraBounding");

    setTimeout(capturarYEnviar, 200); // 5 fps (ajustable)
  }

  capturarYEnviar();
}

iniciarCamara();

async function guardarPuntos() {
  const cameras = [
    { id: "camera1", prefix: "cam1" },
    { id: "camera2", prefix: "cam2" },
  ];
  const esquinas = [
    {
      name: "esquina-superior-izquierda",
      label: "la esquina superior izquierda",
    },
    { name: "esquina-superior-derecha", label: "la esquina superior derecha" },
    {
      name: "esquina-inferior-izquierda",
      label: "la esquina inferior izquierda",
    },
    { name: "esquina-inferior-derecha", label: "la esquina inferior derecha" },
  ];

  for (const esquina of esquinas) {
    for (const camera of cameras) {
      alert(
        `Seleccione el pixel correspondiente a ${esquina.label} de la ${
          camera.id === "camera1" ? "cámara de seguridad" : "plano desde arriba"
        }`
      );
      const punto = await obtenerPunto(
        document.getElementById(camera.id),
        "red"
      );
      // Guardar las coordenadas en el HTML
      document.getElementById(
        `${camera.prefix}-${esquina.name}_x`
      ).innerHTML = `${punto.x}`;
      document.getElementById(
        `${camera.prefix}-${esquina.name}_y`
      ).innerHTML = `${punto.y}`;
    }
  }
}

async function VerificacionPuntos() {
  const punto = await obtenerPunto(document.getElementById("camera1"), "green");

  // Datos que se enviarán en el cuerpo del POST
  const datos = {
    x: String(punto.x),
    y: String(punto.y),
  };

  // URL del endpoint de la API
  const url = "http://localhost:8000/getPixelPlano";
  try {
    // Llamar a enviarResultados y esperar la respuesta
    const json_recibido = await enviarResultados(datos, url);
    pintarPunto("camera2", json_recibido.x, json_recibido.y);
  } catch (error) {
    console.error("Error en VerificacionPuntos:", error);
    alert("Ocurrió un error al procesar la solicitud.");
  }
}

// Función para obtener el punto donde el usuario hace clic en una imagen
function obtenerPunto(camera, color) {
  return new Promise((resolve) => {
    function handleClick(event) {
      const rect = event.target.getBoundingClientRect(); // Obtener posición de la imagen
      const x = event.clientX - rect.left; // Coordenada X relativa a la imagen
      const y = event.clientY - rect.top; // Coordenada Y relativa a la imagen

      // Crear un marcador (punto)
      const marker = document.createElement("div");
      marker.style.position = "absolute";
      marker.style.width = "25px";
      marker.style.height = "25px";
      marker.style.backgroundColor = color;
      marker.style.borderRadius = "50%";
      marker.style.left = `${x + rect.left}px`; // Posición absoluta en la página
      marker.style.top = `${y + rect.top}px`; // Posición absoluta en la página
      marker.style.transform = "translate(-50%, -50%)"; // Centrar el marcador

      // Agregar el marcador al contenedor de la imagen
      document.body.appendChild(marker);

      // Eliminar el evento de clic después de seleccionar el punto
      camera.removeEventListener("click", handleClick);

      // Resolver la promesa con las coordenadas
      resolve({ x, y });
    }

    // Agregar el evento de clic a la cámara
    camera.addEventListener("click", handleClick);
  });
}

function pintarLinea(cameraId, x1, y1, x2, y2, color = "red") {
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
}

function pintarPunto(cameraId, x, y, color = "blue") {
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

  // Pintar línea desde el último punto al nuevo
  if (ultimoPunto.x !== null && ultimoPunto.y !== null) {
    pintarLinea(cameraId, ultimoPunto.x, ultimoPunto.y, x, y);
  }

  // Actualizar variable global
  ultimoPunto = { x, y };
}

function verificarYEnviar() {
  const datos = {
    camara1: {
      esquinaSuperiorIzquierda: {
        x: document.getElementById("cam1-esquina-superior-izquierda_x")
          .innerHTML,
        y: document.getElementById("cam1-esquina-superior-izquierda_y")
          .innerHTML,
      },
      esquinaSuperiorDerecha: {
        x: document.getElementById("cam1-esquina-superior-derecha_x").innerHTML,
        y: document.getElementById("cam1-esquina-superior-derecha_y").innerHTML,
      },
      esquinaInferiorIzquierda: {
        x: document.getElementById("cam1-esquina-inferior-izquierda_x")
          .innerHTML,
        y: document.getElementById("cam1-esquina-inferior-izquierda_y")
          .innerHTML,
      },
      esquinaInferiorDerecha: {
        x: document.getElementById("cam1-esquina-inferior-derecha_x").innerHTML,
        y: document.getElementById("cam1-esquina-inferior-derecha_y").innerHTML,
      },
    },
    camara2: {
      esquinaSuperiorIzquierda: {
        x: document.getElementById("cam2-esquina-superior-izquierda_x")
          .innerHTML,
        y: document.getElementById("cam2-esquina-superior-izquierda_y")
          .innerHTML,
      },
      esquinaSuperiorDerecha: {
        x: document.getElementById("cam2-esquina-superior-derecha_x").innerHTML,
        y: document.getElementById("cam2-esquina-superior-derecha_y").innerHTML,
      },
      esquinaInferiorIzquierda: {
        x: document.getElementById("cam2-esquina-inferior-izquierda_x")
          .innerHTML,
        y: document.getElementById("cam2-esquina-inferior-izquierda_y")
          .innerHTML,
      },
      esquinaInferiorDerecha: {
        x: document.getElementById("cam2-esquina-inferior-derecha_x").innerHTML,
        y: document.getElementById("cam2-esquina-inferior-derecha_y").innerHTML,
      },
    },
  };

  // URL del endpoint de la API
  const url = "http://localhost:8000/setConfiguration";

  // Seleccionar todas las celdas de la tabla que contienen coordenadas
  const celdas = document.querySelectorAll('td[id^="cam"]');

  // Verificar si alguna celda contiene el valor "✓"
  for (const celda of celdas) {
    if (celda.innerHTML.trim() === "✓") {
      alert(
        "Error: Todas las celdas deben tener valores antes de enviar los datos."
      );
      return; // Detener la ejecución si se encuentra una celda con "✓"
    }
  }

  // Si todas las celdas tienen valores válidos, llamar a la función para enviar los datos
  enviarResultados(datos, url);
  alert("Datos enviados correctamente.");
}

async function enviarResultados(datos, url) {
  try {
    // Configuración de la solicitud POST
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(datos), // Convertir el objeto a JSON
    });

    // Verificar si la respuesta es exitosa
    if (!response.ok) {
      throw new Error(`Error en la solicitud: ${response.status}`);
    }

    // Parsear la respuesta como JSON
    const data = await response.json();
    console.log("Respuesta de la API:", data);
    return data; // Devolver la respuesta parseada
  } catch (error) {
    console.error("Error al enviar los datos:", error);
    throw error; // Propagar el error para manejarlo en VerificacionPuntos
  }
}

function enviarFotogramas(datos, url) {
  fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(datos),
  }).catch((error) => {
    console.error("Error al enviar los datos:", error);
  });
}

