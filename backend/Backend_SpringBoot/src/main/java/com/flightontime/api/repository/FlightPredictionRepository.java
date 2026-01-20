package com.flightontime.api.repository;

import com.flightontime.api.model.FlightPredictionEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface FlightPredictionRepository extends JpaRepository<FlightPredictionEntity, Long> {
}