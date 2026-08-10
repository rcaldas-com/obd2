package com.obd2.lambda;

import android.Manifest;
import android.content.Context;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.os.Looper;
import android.util.Log;

import androidx.core.app.ActivityCompat;

/**
 * Velocidade via GPS do próprio dispositivo (a multimídia tem GPS embutido),
 * pra substituir a velocidade do OBD2 quando ela vem errada — comum em
 * instalações com Speeduino, já que a ECU original pode não ter mais uma
 * leitura de roda correta ou coerente.
 *
 * Usa LocationManager puro (não Play Services / FusedLocationProvider), já
 * que multimídias automotivas normalmente não têm Google Play Services.
 */
public class GpsSpeedProvider {

    private static final String TAG = "GpsSpeed";
    private static final long MIN_TIME_MS = 500;
    private static final float MIN_DISTANCE_M = 0f;
    // Sem atualização de GPS há mais tempo que isso, considera indisponível
    // — o chamador deve usar outra fonte (OBD2) como reserva.
    private static final long STALE_AFTER_MS = 5000;

    private final Context context;
    private final LocationManager locationManager;
    private volatile Float speedKmh;
    private volatile long lastUpdateAt = 0;
    private boolean listening = false;

    private final LocationListener listener = new LocationListener() {
        @Override
        public void onLocationChanged(Location location) {
            if (location.hasSpeed()) {
                speedKmh = location.getSpeed() * 3.6f; // m/s -> km/h
                lastUpdateAt = System.currentTimeMillis();
            }
        }

        @Override
        public void onStatusChanged(String provider, int status, Bundle extras) {}

        @Override
        public void onProviderEnabled(String provider) {}

        @Override
        public void onProviderDisabled(String provider) {}
    };

    public GpsSpeedProvider(Context context) {
        this.context = context.getApplicationContext();
        locationManager = (LocationManager) this.context.getSystemService(Context.LOCATION_SERVICE);
    }

    public boolean hasPermission() {
        return ActivityCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
                == android.content.pm.PackageManager.PERMISSION_GRANTED;
    }

    public void start() {
        if (listening || locationManager == null || !hasPermission()) return;
        try {
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(
                        LocationManager.GPS_PROVIDER, MIN_TIME_MS, MIN_DISTANCE_M, listener, Looper.getMainLooper());
                listening = true;
            } else {
                Log.w(TAG, "GPS desligado no dispositivo");
            }
        } catch (SecurityException e) {
            Log.w(TAG, "Sem permissão de localização: " + e.getMessage());
        }
    }

    public void stop() {
        if (!listening) return;
        try {
            locationManager.removeUpdates(listener);
        } catch (SecurityException ignored) {}
        listening = false;
    }

    /** Velocidade em km/h vinda do GPS, ou null se não tiver fix recente
     * (mais de {@link #STALE_AFTER_MS} sem atualização, ou permissão/GPS
     * indisponível) — o chamador cai pra outra fonte (OBD2) nesse caso. */
    public Float getSpeedKmh() {
        if (speedKmh == null) return null;
        if (System.currentTimeMillis() - lastUpdateAt > STALE_AFTER_MS) return null;
        return speedKmh;
    }
}
