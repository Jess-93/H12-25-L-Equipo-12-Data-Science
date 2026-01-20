package com.flightontime.api.model;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Entity
@Table(name = "flight_history")
@Data
public class FlightPredictionEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // Datos de la petición
    private String airline;
    private String originAirport;
    private String destinationAirport;
    private Integer scheduledDeparture;
    private Integer distance;
    @Column(name = "flight_year")
    private Integer flightYear; // "Year" es palabra reservada en SQL, recomendable usar flightYear para robustez.
    @Column(name = "flight_month")
    private Integer flightMonth;
    @Column(name = "flight_day")
    private Integer flightDay;

    // Datos de la respuesta de la IA
    private String prevision;
    private Double probabilidad;

    // Auditoría
    private LocalDateTime createdAt = LocalDateTime.now();
}