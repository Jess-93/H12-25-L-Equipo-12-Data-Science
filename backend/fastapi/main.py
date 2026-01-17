from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import papermill as pm
import scrapbook as sb

app = FastAPI()

class FlightModel(BaseModel):
    airline: str
    origin_airport: str
    destination_airport: str
    scheduled_departure: int
    distance: int
    year: int
    month: int
    day: int


@app.post('/predict')
async def generar_prediccion(flight : FlightModel):
    print('vuelo: ', flight)
    predicion = predecir(flight)
    print('prediccion: ', predicion)
    # return {
    #     "prevision": 'On Time (desde fastapi)',
    #     "probabilidad": 0.99
    # }

    return predicion

def predecir(flight):
    pm.execute_notebook(
        input_path='../../data-science/notebook_prediccion.ipynb',
        output_path='analisis_ejecutado.ipynb',
        parameters={
            'json_ticket': {
                "airline": "AA",
                "origin_airport": "DFW",
                "destination_airport": "LAX",
                "scheduled_departure": 900,
                "distance": 1235,
                "year": 2025,
                "month": 11,
                "day": 10
                }
        }
    )

    notebook = sb.read_notebook('analisis_ejecutado.ipynb')

    prediccion = notebook.scraps['predict_api'].data
    return prediccion