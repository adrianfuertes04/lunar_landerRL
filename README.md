# Lunar Lander v3 (Python) con Entrenamiento IA

Este proyecto implementa una versión del clásico juego Lunar Lander utilizando Python y Pygame, con la adición de un módulo de entrenamiento basado en Redes Neuronales Multicapa (MLP) y Algoritmos Genéticos.

## Descripción

Lunar Lander v3 es un juego en el que el jugador controla una nave espacial que debe aterrizar con seguridad en una plataforma lunar. Además, este proyecto incluye un sistema de inteligencia artificial que puede aprender a jugar el juego automáticamente.

## Características

* Gráficos simples utilizando Pygame.
* Física básica para simular la gravedad y el movimiento de la nave.
* Control del combustible y la velocidad de la nave.
* Detección de colisiones.
* Interfaz de usuario básica para mostrar información del juego.
* **Entrenamiento con Redes Neuronales Multicapa (MLP):**
    * Utiliza MLP para aprender a controlar la nave.
    * Entrenamiento basado en recompensas por aterrizajes exitosos.
* **Optimización con Algoritmos Genéticos:**
    * Ajusta los pesos y sesgos de la MLP para mejorar el rendimiento.
    * Implementa selección, cruce y mutación para evolucionar la red neuronal.

## Requisitos

* Python 3.x
* Pygame (`pip install pygame`)
* NumPy (`pip install numpy`)

## Cómo ejecutar

1.  Asegúrate de tener Python y las dependencias instaladas.
2.  Descarga o clona este repositorio.
3.  Navega hasta el directorio del proyecto en tu terminal.

## Controles

* Flecha arriba: Aumentar el empuje del motor.
* Flecha izquierda/derecha: Rotar la nave.

## Estructura del proyecto

* `Gymnasium alumno.py`: El archivo principal del juego y el entrenamiento.
* `MLP.py`: Implementación de la Red Neuronal Multicapa.

## Contribución

Las contribuciones son bienvenidas. Si encuentras errores o tienes ideas para mejorar el juego o el entrenamiento de la IA, no dudes en crear un pull request.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para obtener más información.

## Notas adicionales

* El entrenamiento puede requerir un tiempo considerable dependiendo de la complejidad de la red neuronal y los parámetros del algoritmo genético.
* Se pueden ajustar los parámetros de entrenamiento en los archivos `MLP.py` y  para experimentar con diferentes configuraciones.
