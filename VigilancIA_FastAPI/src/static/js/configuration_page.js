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
            const punto = await obtenerPunto(document.getElementById(camera.id),'red');
            //console.log(`Coordenadas seleccionadas en ${camera.id}: X=${punto.x}, Y=${punto.y}`);
            //alert(`Coordenadas seleccionadas en ${camera.id}: X=${punto.x}, Y=${punto.y}`);
            document.getElementById(`${camera.prefix}-${esquina.name}_x`).innerHTML = `${punto.x}`;
            document.getElementById(`${camera.prefix}-${esquina.name}_y`).innerHTML = `${punto.y}`;
        }
    }
}

async function VerificacionPuntos() {
    const punto = await obtenerPunto(document.getElementById('camera1'),'green');
    alert(`Coordenadas seleccionadas en: X=${punto.x}, Y=${punto.y}`);
    // Datos que se enviarán en el cuerpo del POST
    const datos = {
        x: String(punto.x),
        y: String(punto.y)
    };

    // URL del endpoint de la API
    const url = 'http://localhost:8000/getPixelPlano';
    try {
        // Llamar a enviarResultados y esperar la respuesta
        const json_recibido = await enviarResultados(datos, url);
        pintarPuntoPlano(json_recibido.x, json_recibido.y);
        alert(`Respuesta recibida: X=${json_recibido.x}, Y=${json_recibido.y}`);
    } catch (error) {
        console.error('Error en VerificacionPuntos:', error);
        alert('Ocurrió un error al procesar la solicitud.');
    }
}

// Función para obtener el punto donde el usuario hace clic en una imagen
function obtenerPunto(camera,color) {
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
            marker.style.backgroundColor = color;
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

function pintarPuntoPlano(x = null, y = null) {
    const camera = document.getElementById('camera2'); // Siempre usar 'camera2'
    const color = 'blue'; // Color fijo

    return new Promise((resolve) => {
        function pintarPunto(x, y) {
            // Crear un marcador (punto)
            const marker = document.createElement('div');
            marker.style.position = 'absolute';
            marker.style.width = '50px';
            marker.style.height = '50px';
            marker.style.backgroundColor = color;
            marker.style.borderRadius = '50%';
            marker.style.left = `${x}px`; // Posición relativa al contenedor
            marker.style.top = `${y}px`; // Posición relativa al contenedor
            marker.style.transform = 'translate(-50%, -50%)'; // Centrar el marcador

            // Agregar el marcador al contenedor de la cámara
            camera.appendChild(marker);

            // Resolver la promesa con las coordenadas
            resolve({ x, y });
        }

        if (x !== null && y !== null) {
            // Si se proporcionan coordenadas, pintar directamente
            pintarPunto(x, y);
        } else {
            // Si no se proporcionan coordenadas, esperar al clic
            function handleClick(event) {
                const rect = camera.getBoundingClientRect(); // Obtener posición del elemento camera2
                const clickX = event.clientX - rect.left; // Coordenada X relativa al elemento
                const clickY = event.clientY - rect.top;  // Coordenada Y relativa al elemento

                // Pintar el punto en las coordenadas del clic
                pintarPunto(clickX, clickY);

                // Eliminar el evento de clic después de seleccionar el punto
                camera.removeEventListener('click', handleClick);
            }

            // Agregar el evento de clic al elemento camera2
            camera.addEventListener('click', handleClick);
        }
    });
}


function verificarYEnviar() {
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
    enviarResultados(datos,url);
    alert('Datos enviados correctamente.');
}

async function enviarResultados(datos, url) {
    try {
        // Configuración de la solicitud POST
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(datos) // Convertir el objeto a JSON
        });

        // Verificar si la respuesta es exitosa
        if (!response.ok) {
            throw new Error(`Error en la solicitud: ${response.status}`);
        }

        // Parsear la respuesta como JSON
        const data = await response.json();
        console.log('Respuesta de la API:', data);
        return data; // Devolver la respuesta parseada
    } catch (error) {
        console.error('Error al enviar los datos:', error);
        throw error; // Propagar el error para manejarlo en VerificacionPuntos
    }
}