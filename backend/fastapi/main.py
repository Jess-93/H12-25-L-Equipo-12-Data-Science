from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

class FlightModel(BaseModel):
    aerolinea: str
    origen: str
    destino: str
    fechaPartida: datetime
    distanciaKm: float 



@app.post('/predict')
async def generar_prediccion(flight : FlightModel):
    print('vuelo: ', flight)
    return {
        "prevision": 'On Time (desde fastapi)',
        "probabilidad": 0.99
    }