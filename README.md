## Índice
- [Índice](#índice)
- [Librerías](#librerías)
- [Importar el dataset](#importar-el-dataset)
- [Modelo](#modelo)
  - [Capa de entrada](#capa-de-entrada)
  - [Capas ocultas](#capas-ocultas)
  - [Capa de salida](#capa-de-salida)
  - [Compilación](#compilación)
- [Entrenamiento](#entrenamiento)
- [Validación y aplicación](#validación-y-aplicación)
  - [Aplicación en tiempo real](#aplicación-en-tiempo-real)

---
## Librerías
Vamos a necesitar algunas librerias para poder crear, entrenar y guardar el modelo.

Se usará **numpy** para dar una nueva forma a una matriz sin cambiar los datos.
```python
import numpy as np
```
La librería **matplotlib.pyplot** permitirá una vez entrenado el modelo visualizar un gráfico con la precisión y error del modelo tanto en entrenamiento como en validación.
```python
import matplotlib.pyplot as plt
```

Para leer y redimensionar las imagenes del dataset se usará **cv2**.
```python
import cv2
```
Para acceder a recursos del sistema operativo se usara la libreria **os**.
```python
import os
```
La librería **random** ayudará a crear posciones aleatorias para que el modelo no se entrene con las imagenes en orden si no en deorden y no encuentre patrones tan facilemente.
```python
import random
```
**Tensorflow** es una librería muy completa de la que nos ayudaremos para crear el modelo, definir sus capas, los parámetros de entrenamiento y guardado. 
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
```
## Importar el dataset
La función se ejecuta para poder importar el dataset a una variable y porder entrenar al modelo.
```python
create_training_data()
```
Ahora vamos a instanciar un objeto de tipo **Sequential**.
```python
model = Sequential()
```
El modelo tiene 1 capa de entrada, 2 capas ocultas y una capa de salida.

## Modelo
### Capa de entrada 
Esta capa nos permite la entrada de información a la red y no se realiza ningún procesamiento en ella. Con el objetivo de permitir el ingreso de la información se realiza un modelo secuencial sequential(), la cual es apropiada ya que tenemos un tensor de entrada. Cuando realizamos una arquitectura secuencial, es necesario aplicar capas de forma incremental add() la cual tiene una entra de una convolución 2D con 64 neuronas, el número de neuronas denota en la profundidad de la red, mientras más neuronas se tenga, más profunda será la red, además de adicionar el argumento input_shape el cual nos permite obtener el valor de los pixeles, para posteriormente ser procesados.

Los datos obtenidos requieren de una rectificación en la red neuronal para lo cual se aplica activación(‘relu’), la modificación de estos parámetros no permite utilizar umbrales distintos de cero.

Por consiguiente, se realiza una agrupación máxima de datos espaciales 2D MaxPooling2D, la cual permite la representación de los datos, tomando los valores máximos definida por pool_size(2,2) el cual tomará el valor máximo sobre una ventada de agrupación 2x2.
```python
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
```
### Capas ocultas 
Para la primera capa oculta usaremos las mimsas funciones y parámetros,
```python
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
```

Como segunda capa oculta ocupamos
model.flatten() para pasar el tensor multidimensinal a unidimensional (aplanar), de esta manera podemos pasar a una capa densa de Keras “Dense(64)” completamente conectada con una dimensionalidad de salida de 64 nodos. Luego ocupamos otra capa Dense(1)  
 
```python
model.add(Flatten())
model.add(Dense(64)) 
```
### Capa de salida 
En la capa de salida utilizamos la función de activación Sigmoid. Anteriormente en la segunda capa oculta se ocupó otra capa Dense(1), le ponemos 1 debido a que la función de activación que ocupamos en la capa de salida es de clasificación binaria “Sigmoid” (El valor de salida será 0 o 1) y solo necesitará una neurona de salida 


```python
model.add(Dense(1))
model.add(Activation('sigmoid')) 
```
Antes de empezar con el entrenamiento se debe configurar ciertos parámetros previos. 
1)	Optimizador u optimizer: se hará uso del optimizador basado en gradientes de Adam. Este optimizador es una versión más refinada del RMSprop [5].
2)	Función de pérdida o loss: se ha elegido binary-crossentropy como función de pérdida ya que la red neuronal solo debe distinguir entre 2 categorías [5].
3)	Lista de métricas o metrics: dado que solo queremos clasificar y hallar la precisión con la que lo hace, sólo pediremos que devuelva la precisión o acurracy [5].
### Compilación
```python
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```
 Es hora de empezar con el entrenamiento y para ello describamos brevemente las métricas usadas. Como primer parámetro se debe proporcionar todas las imágenes del dataset y que debe ser de tipo Numpy array. El segundo parámetro y contiene los valores esperados o target de la solución deseada en un array Numpy. Batch_size es el tercer parámetro usado, si no se define, el valor por defecto es 32.
Como penúltimo parámetro tenemos a epochs que define una iteración a todo el contenido del dataset. Por último, tenemos a validation_split que es un valor flotante de entre 0 y 1 que se traduce al porcentaje de datos que se aparta del dataset para evaluar el modelo cada vez que termina una iteración y devuelve los valores de val_loss y val_acurracy. Para el presente estudio vamos a dejar un 2% para que realice esta operación. Este argumento no se admite cuando x es un conjunto de datos, generador o instancia de keras.utils.Sequence.

## Entrenamiento
```python
epochs=5
history = model.fit(X, y, batch_size=64, epochs=epochs, validation_split=0.02) 
model.save('class.model')
```
Por último hacemos uso de la librería 

```python
import matplotlib.pyplot as plt
```
para graficar los resultados del modelo.

![](modelo.jpg)

## Validación y aplicación

### Aplicación en tiempo real

Primero importamos las mismas librerías que para el entrenamiento salvo por 

```python
from plyer import notification
import time
```
Que harán posible el análisis casi en tiempo real y las notificaciones en pantalla.

Creamos una variable que contenga las categorías posibles dentro de la predicción así como se hizo en el entrenamiento.

```python
CATEGORIES = ["present", "no_present"]
```

En un bucle y con la ayuda de la cámara web se va a estar ejecutando la función **predict** del modelo cada segundo.

Las funciones usadas son las siguientes:

```python
makeModel(image_name)
```
Esta función nos dirá por consola si el estudiante se encuentra presente ante la clase virtual o no. 

Primero se carga el modelo guardado anteriormente en el entrenamiento 
```python
models = tf.keras.models.load_model("class.model")
```
y se llama a la función de predicción del modelo que proporciona Tensorflow a demás de ejecutarse la función prepare que redimensiona y ajusta la imagen a analizar desde la cámara web.
```python
prediction = models.predict([prepare(image)])
```
Una función extra que se ha desarrollado con la librería **plyer.notification** permite desplegar una alerta o notificación en el ordenador cuando el estudiante no se encuentre presente y le llame la atención para que regrese.

Se debe definir el título, menasaje, icono y tiempo en pantalla, una libirerñia muy limpia y sencilla de usar.

```python
def notif(predic):
    if predic == "no_present":
        notification.notify(
            title='Class status',
            message='Estudiante ausente de la clase.',
            app_icon='./icons/sad.ico',
            timeout=10,
        )
```
![](./icons/sad.ico)

Es la primera versión del análisis de reconocimiento de acciones de los estudiantes en clases virtuales para que los docentes tomen medidas antes posibles distracciones, cansancio, aburrimiento por una mala planificación y metodologías de la clase.






