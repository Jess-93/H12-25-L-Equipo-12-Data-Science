package com.flightontime.api.controller;

import com.flightontime.api.dto.PredictionResponse;
import com.flightontime.api.dto.FlightRequest;
import com.flightontime.api.service.PredictionService;
import org.springframework.web.bind.annotation.*;
import jakarta.validation.Valid;


@RestController
@RequestMapping("/api/v1/flights") // Ruta base para todas las solicitudes de este controlador
@CrossOrigin(
        origins = "*",
        allowedHeaders = "*",
        methods = {RequestMethod.POST, RequestMethod.OPTIONS} // Permitir solo POST y OPTIONS
)
public class PredictionController {

    /*
    ===========================================================================
    INICIO DE BLOQUE: INTEGRACIÓN CON FAST API (FUTURO)
    ===========================================================================
    */

    // NUEVO: Declaramos el servicio para usarlo más abajo
    private final PredictionService predictionService;

    // NUEVO: Constructor para que Spring inyecte el Prediction Service
    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    /*
    ===========================================================================
    FIN DE BLOQUE
    */

    @PostMapping("/predict")
    public PredictionResponse predecirVuelo(@Valid @RequestBody FlightRequest solicitud) {

//         --- INICIO DE SIMULACIÓN (MOCK ACTUAL) ---
//         MANTENIDO: Simulamos una respuesta mientras Data Science termina la IA
//        PredictionResponse respuesta = new PredictionResponse();
//        respuesta.setPrevision("On Time");
//        respuesta.setProbabilidad(0.95);
//
//        return respuesta;
//         --- FIN DE SIMULACIÓN ---

        /*
        ===========================================================================
        INICIO DE LÓGICA REAL CON FAST API
        ===========================================================================
        */

        // Delegamos la responsabilidad al Service (Lógica de negocio FAST API)
         return predictionService.getPrediction(solicitud);

        /*
        ===========================================================================
        */


    }
}