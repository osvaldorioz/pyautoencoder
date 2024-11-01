from fastapi import FastAPI
import autoencoder_module
import time
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

# Definir el modelo para el vector
class Vector(BaseModel):
    vector: List[float]

@app.post("/autoencoder")
async def auto(entradas: int, 
             nodos_ocultos: int,
             epochs: int, 
             learning_rate: float,
             data: Matrix,
             muestra: Vector):
    start = time.time()
    
    # Inicializar el modelo con 2 clusters y un máximo de 100 iteraciones
    ae = autoencoder_module.Autoencoder(entradas, nodos_ocultos)

    # Entrenar el autoencoder
    result = ae.train(data.matrix, epochs, learning_rate)
    
    # Probar la red con una muestra de entrada
    output = ae.forward(muestra.vector)

    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "Resultado": result,
        "Salida": output
    }
    jj = json.dumps(j1)

    return jj