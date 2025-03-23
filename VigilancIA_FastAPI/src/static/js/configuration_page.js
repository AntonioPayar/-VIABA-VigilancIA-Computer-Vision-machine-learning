async function guardarPuntos() {
    const cameras = [
        { id: 'camera1', prefix: 'cam1' },
        { id: 'camera2', prefix: 'cam2' }
    ];
    const esquinas = [
        { name: 'esquina-superior-izquierda', label: 'la esquina superior izquierda' },
        { name: 'esquina-superior-derecha', label: 'la esquina superior derecha' },
        { name: 'esquina-inferior-izquierda', label: 'la esquina inferior izquierda' },
        { name: 'esquina-inferior-derecha', label: 'la esquina inferior derecha' }
    ];

    for (const esquina of esquinas) {
        for (const camera of cameras) {
            alert(`Seleccione el pixel correspondiente a ${esquina.label} de la ${camera.id === 'camera1' ? 'cámara de seguridad' : 'plano desde arriba'}`);
            const punto = await obtenerPunto(document.getElementById(camera.id));
            //console.log(`Coordenadas seleccionadas en ${camera.id}: X=${punto.x}, Y=${punto.y}`);
            //alert(`Coordenadas seleccionadas en ${camera.id}: X=${punto.x}, Y=${punto.y}`);
            document.getElementById(`${camera.prefix}-${esquina.name}_x`).innerHTML = `${punto.x}`;
            document.getElementById(`${camera.prefix}-${esquina.name}_y`).innerHTML = `${punto.y}`;
        }
    }
}

// Función para obtener el punto donde el usuario hace clic en una imagen
function obtenerPunto(camera) {
    return new Promise((resolve) => {
        function handleClick(event) {
            const rect = event.target.getBoundingClientRect(); // Obtener posición de la imagen
            const x = event.clientX - rect.left; // Coordenada X relativa a la imagen
            const y = event.clientY - rect.top;  // Coordenada Y relativa a la imagen

            // Crear un marcador (punto)
            const marker = document.createElement('div');
            marker.style.position = 'absolute';
            marker.style.width = '50px';
            marker.style.height = '50px';
            marker.style.backgroundColor = 'red';
            marker.style.borderRadius = '50%';
            marker.style.left = `${x + rect.left}px`; // Posición absoluta en la página
            marker.style.top = `${y + rect.top}px`; // Posición absoluta en la página
            marker.style.transform = 'translate(-50%, -50%)'; // Centrar el marcador

            // Agregar el marcador al contenedor de la imagen
            document.body.appendChild(marker);

            // Eliminar el evento de clic después de seleccionar el punto
            camera.removeEventListener('click', handleClick);

            // Resolver la promesa con las coordenadas
            resolve({ x, y });
        }

        // Agregar el evento de clic a la cámara
        camera.addEventListener('click', handleClick);
    });
}


// JavaScript
function verificarYEnviar() {
    // Seleccionar todas las celdas de la tabla que contienen coordenadas
    const celdas = document.querySelectorAll('td[id^="cam"]');

    // Verificar si alguna celda contiene el valor "✓"
    for (const celda of celdas) {
        if (celda.innerHTML.trim() === '✓') {
            alert('Error: Todas las celdas deben tener valores antes de enviar los datos.');
            return; // Detener la ejecución si se encuentra una celda con "✓"
        }
    }

    // Si todas las celdas tienen valores válidos, llamar a la función para enviar los datos
    enviarResultados();
}

function enviarResultados() {
    // Datos que se enviarán en el cuerpo del POST
    const datos = {
        camara1: {
            esquinaSuperiorIzquierda : {
                x: document.getElementById('cam1-esquina-superior-izquierda_x').innerHTML,
                y: document.getElementById('cam1-esquina-superior-izquierda_y').innerHTML
            },
            esquinaSuperiorDerecha : {
                x: document.getElementById('cam1-esquina-superior-derecha_x').innerHTML,
                y: document.getElementById('cam1-esquina-superior-derecha_y').innerHTML
            },
            esquinaInferiorIzquierda : {
                x: document.getElementById('cam1-esquina-inferior-izquierda_x').innerHTML,
                y: document.getElementById('cam1-esquina-inferior-izquierda_y').innerHTML
            },
            esquinaInferiorDerecha : {
                x: document.getElementById('cam1-esquina-inferior-derecha_x').innerHTML,
                y: document.getElementById('cam1-esquina-inferior-derecha_y').innerHTML
            }
        },
        camara2: {
            esquinaSuperiorIzquierda : {
                x: document.getElementById('cam2-esquina-superior-izquierda_x').innerHTML,
                y: document.getElementById('cam2-esquina-superior-izquierda_y').innerHTML
            },
            esquinaSuperiorDerecha : {
                x: document.getElementById('cam2-esquina-superior-derecha_x').innerHTML,
                y: document.getElementById('cam2-esquina-superior-derecha_y').innerHTML
            },
            esquinaInferiorIzquierda : {
                x: document.getElementById('cam2-esquina-inferior-izquierda_x').innerHTML,
                y: document.getElementById('cam2-esquina-inferior-izquierda_y').innerHTML
            },
            esquinaInferiorDerecha : {
                x: document.getElementById('cam2-esquina-inferior-derecha_x').innerHTML,
                y: document.getElementById('cam2-esquina-inferior-derecha_y').innerHTML
            }
        }
    };

    // URL del endpoint de la API
    const url = 'http://localhost:8000/getConfiguration';

    // Configuración de la solicitud POST
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(datos) // Convertir el objeto a JSON
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Error en la solicitud: ${response.status}`);
        }
        return response.json(); // Parsear la respuesta como JSON
    })
    .then(data => {
        console.log('Respuesta de la API:', data);
        alert('Datos enviados correctamente');
    })
    .catch(error => {
        console.error('Error al enviar los datos:', error);
        alert('Hubo un error al enviar los datos');
    });
}