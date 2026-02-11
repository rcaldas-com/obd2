package com.obd2.lambda;

import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Color;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.hoho.android.usbserial.driver.UsbSerialDriver;
import com.hoho.android.usbserial.driver.UsbSerialProber;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "LambdaMonitor";
    private static final String ACTION_USB_PERMISSION = "com.obd2.lambda.USB_PERMISSION";
    private static final int POLL_INTERVAL_MS = 300;

    // UI Elements
    private TextView tvO2S1Value, tvO2S1Status, tvO2S1Label;
    private TextView tvO2S5Value, tvO2S5Status;
    private TextView tvStft1, tvStft2;
    private TextView tvRpm, tvTiming;
    private TextView tvConnStatus, tvSampleRate;
    private View indicatorBar;
    private Button btnConnect;
    private LambdaChartView chartView;
    private LinearLayout layoutDash, layoutConnect;

    // Logic
    private Elm327Manager elm327;
    private UsbManager usbManager;
    private HandlerThread pollThread;
    private Handler pollHandler;
    private Handler uiHandler;
    private boolean polling = false;
    private int sampleCount = 0;
    private long startTime = 0;

    // CSV Logging
    private BufferedWriter csvWriter;
    private boolean logging = false;

    private final BroadcastReceiver usbPermissionReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (ACTION_USB_PERMISSION.equals(intent.getAction())) {
                synchronized (this) {
                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {
                        doConnect();
                    } else {
                        showStatus("Permissão USB negada");
                    }
                }
            }
        }
    };

    private final BroadcastReceiver usbDetachReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (UsbManager.ACTION_USB_DEVICE_DETACHED.equals(intent.getAction())) {
                stopPolling();
                showStatus("USB desconectado");
                showConnectView();
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Manter tela ligada
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        usbManager = (UsbManager) getSystemService(USB_SERVICE);
        elm327 = new Elm327Manager();
        uiHandler = new Handler(getMainLooper());

        initViews();
        registerReceivers();

        // Verificar se já há um dispositivo USB conectado
        checkExistingUsb();
    }

    private void initViews() {
        layoutConnect = findViewById(R.id.layout_connect);
        layoutDash = findViewById(R.id.layout_dash);
        btnConnect = findViewById(R.id.btn_connect);
        tvConnStatus = findViewById(R.id.tv_conn_status);

        tvO2S1Value = findViewById(R.id.tv_o2s1_value);
        tvO2S1Status = findViewById(R.id.tv_o2s1_status);
        tvO2S1Label = findViewById(R.id.tv_o2s1_label);

        tvO2S5Value = findViewById(R.id.tv_o2s5_value);
        tvO2S5Status = findViewById(R.id.tv_o2s5_status);

        tvStft1 = findViewById(R.id.tv_stft1);
        tvStft2 = findViewById(R.id.tv_stft2);
        tvRpm = findViewById(R.id.tv_rpm);
        tvTiming = findViewById(R.id.tv_timing);
        tvSampleRate = findViewById(R.id.tv_sample_rate);
        indicatorBar = findViewById(R.id.indicator_bar);

        chartView = findViewById(R.id.chart_view);

        btnConnect.setOnClickListener(v -> requestConnection());
        findViewById(R.id.btn_disconnect).setOnClickListener(v -> {
            stopPolling();
            showConnectView();
        });
    }

    private void registerReceivers() {
        IntentFilter permFilter = new IntentFilter(ACTION_USB_PERMISSION);
        registerReceiver(usbPermissionReceiver, permFilter);

        IntentFilter detachFilter = new IntentFilter(UsbManager.ACTION_USB_DEVICE_DETACHED);
        registerReceiver(usbDetachReceiver, detachFilter);
    }

    private void checkExistingUsb() {
        List<UsbSerialDriver> drivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager);
        if (!drivers.isEmpty()) {
            tvConnStatus.setText("Adaptador USB detectado. Toque em Conectar.");
        } else {
            tvConnStatus.setText("Conecte o adaptador ELM327 USB");
        }
    }

    private void requestConnection() {
        List<UsbSerialDriver> drivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager);
        if (drivers.isEmpty()) {
            showStatus("Nenhum adaptador USB encontrado");
            return;
        }

        UsbDevice device = drivers.get(0).getDevice();
        if (usbManager.hasPermission(device)) {
            doConnect();
        } else {
            btnConnect.setEnabled(false);
            btnConnect.setText("Aguardando permissão...");
            PendingIntent pi = PendingIntent.getBroadcast(this, 0,
                    new Intent(ACTION_USB_PERMISSION),
                    PendingIntent.FLAG_IMMUTABLE);
            usbManager.requestPermission(device, pi);
        }
    }

    private void doConnect() {
        btnConnect.setEnabled(false);
        btnConnect.setText("Conectando...");

        new Thread(() -> {
            try {
                String deviceName = elm327.connect(usbManager);
                uiHandler.post(() -> {
                    showDashView();
                    showStatus("Conectado: " + deviceName);
                    startPolling();
                    startCsvLog();
                    // Foreground service para o Android 9 não matar o app
                    startForegroundService(new Intent(MainActivity.this, OBD2ForegroundService.class));
                });
            } catch (IOException e) {
                uiHandler.post(() -> {
                    showStatus("Erro: " + e.getMessage());
                    btnConnect.setEnabled(true);
                    btnConnect.setText("CONECTAR");
                });
            }
        }).start();
    }

    private void startPolling() {
        if (polling) return;
        polling = true;
        sampleCount = 0;
        startTime = System.currentTimeMillis();

        pollThread = new HandlerThread("OBD2Poll");
        pollThread.start();
        pollHandler = new Handler(pollThread.getLooper());
        pollHandler.post(pollRunnable);
    }

    private final Runnable pollRunnable = new Runnable() {
        @Override
        public void run() {
            if (!polling || !elm327.isConnected()) return;

            final Elm327Manager.LambdaData data = elm327.readLambdaData();
            sampleCount++;

            uiHandler.post(() -> updateUI(data));

            if (polling) {
                pollHandler.postDelayed(this, POLL_INTERVAL_MS);
            }
        }
    };

    private void stopPolling() {
        polling = false;
        if (pollThread != null) {
            pollThread.quitSafely();
            pollThread = null;
        }
        elm327.disconnect();
        stopCsvLog();
        // Parar foreground service
        stopService(new Intent(this, OBD2ForegroundService.class));
    }

    private void updateUI(Elm327Manager.LambdaData data) {
        // Sample rate
        long elapsed = System.currentTimeMillis() - startTime;
        if (elapsed > 0) {
            float hz = sampleCount * 1000f / elapsed;
            tvSampleRate.setText(String.format(Locale.US, "%.1f Hz", hz));
        }

        // O2 S1 (principal)
        if (data.o2s1Current != null) {
            tvO2S1Value.setText(String.format(Locale.US, "%.3f mA", data.o2s1Current));
            String status = data.getO2S1Status();
            tvO2S1Status.setText(status);

            int bgColor, borderColor;
            switch (status) {
                case "POBRE":
                    bgColor = Color.parseColor("#1A3A5C");
                    borderColor = Color.parseColor("#2196F3");
                    tvO2S1Value.setTextColor(Color.parseColor("#64B5F6"));
                    break;
                case "RICO":
                    bgColor = Color.parseColor("#5C1A1A");
                    borderColor = Color.parseColor("#F44336");
                    tvO2S1Value.setTextColor(Color.parseColor("#EF5350"));
                    break;
                default: // ESTEQUIO
                    bgColor = Color.parseColor("#1A3C1A");
                    borderColor = Color.parseColor("#4CAF50");
                    tvO2S1Value.setTextColor(Color.parseColor("#81C784"));
                    break;
            }
            indicatorBar.setBackgroundColor(borderColor);
        } else {
            tvO2S1Value.setText("--");
            tvO2S1Status.setText("SEM DADOS");
        }

        // O2 S5
        if (data.o2s5Current != null) {
            tvO2S5Value.setText(String.format(Locale.US, "%.3f mA", data.o2s5Current));
            tvO2S5Status.setText(data.o2s5Current < -0.01f ? "POBRE" :
                    data.o2s5Current <= 0.01f ? "ESTEQUIO" : "RICO");
        } else {
            tvO2S5Value.setText("--");
            tvO2S5Status.setText("--");
        }

        // Fuel trims
        tvStft1.setText(data.stft1 != null ? String.format(Locale.US, "%.1f%%", data.stft1) : "--");
        tvStft2.setText(data.stft2 != null ? String.format(Locale.US, "%.1f%%", data.stft2) : "--");

        // Colorir fuel trims
        colorFuelTrim(tvStft1, data.stft1);
        colorFuelTrim(tvStft2, data.stft2);

        // RPM e Timing
        tvRpm.setText(data.rpm != null ? String.valueOf(data.rpm) : "--");
        tvTiming.setText(data.timingAdvance != null ?
                String.format(Locale.US, "%.1f°", data.timingAdvance) : "--");

        // Chart
        chartView.addData(data.o2s1Current, data.o2s5Current);

        // CSV Log
        writeCsvLine(data);
    }

    private void colorFuelTrim(TextView tv, Float val) {
        if (val == null) {
            tv.setTextColor(Color.parseColor("#AAAAAA"));
        } else if (val > 5f) {
            tv.setTextColor(Color.parseColor("#F44336")); // Vermelho - lean
        } else if (val < -5f) {
            tv.setTextColor(Color.parseColor("#2196F3")); // Azul - rich
        } else {
            tv.setTextColor(Color.parseColor("#4CAF50")); // Verde - ok
        }
    }

    private void showConnectView() {
        layoutConnect.setVisibility(View.VISIBLE);
        layoutDash.setVisibility(View.GONE);
        btnConnect.setEnabled(true);
        btnConnect.setText("CONECTAR");
        chartView.clearData();
    }

    private void showDashView() {
        layoutConnect.setVisibility(View.GONE);
        layoutDash.setVisibility(View.VISIBLE);
    }

    private void showStatus(String msg) {
        tvConnStatus.setText(msg);
        Log.i(TAG, msg);
    }

    // ---- CSV Logging ----

    private void startCsvLog() {
        try {
            File dir = new File(getExternalFilesDir(null), "logs");
            if (!dir.exists()) dir.mkdirs();

            String filename = "lambda_" + new SimpleDateFormat("yyyy-MM-dd_HH-mm", Locale.US)
                    .format(new Date()) + ".csv";
            File file = new File(dir, filename);

            csvWriter = new BufferedWriter(new FileWriter(file));
            csvWriter.write("timestamp,o2s1_current,o2s1_lambda,o2s5_current,o2s5_lambda,stft1,stft2,rpm,timing\n");
            logging = true;

            Log.i(TAG, "Logging para: " + file.getAbsolutePath());
            Toast.makeText(this, "Log: " + filename, Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            Log.e(TAG, "Erro ao criar log CSV", e);
        }
    }

    private void writeCsvLine(Elm327Manager.LambdaData data) {
        if (!logging || csvWriter == null) return;
        try {
            csvWriter.write(String.format(Locale.US, "%d,%s,%s,%s,%s,%s,%s,%s,%s\n",
                    data.timestamp,
                    data.o2s1Current != null ? String.format("%.4f", data.o2s1Current) : "",
                    data.o2s1Lambda != null ? String.format("%.4f", data.o2s1Lambda) : "",
                    data.o2s5Current != null ? String.format("%.4f", data.o2s5Current) : "",
                    data.o2s5Lambda != null ? String.format("%.4f", data.o2s5Lambda) : "",
                    data.stft1 != null ? String.format("%.1f", data.stft1) : "",
                    data.stft2 != null ? String.format("%.1f", data.stft2) : "",
                    data.rpm != null ? data.rpm.toString() : "",
                    data.timingAdvance != null ? String.format("%.1f", data.timingAdvance) : ""
            ));
            csvWriter.flush();
        } catch (IOException e) {
            Log.e(TAG, "Erro ao escrever CSV", e);
        }
    }

    private void stopCsvLog() {
        logging = false;
        if (csvWriter != null) {
            try {
                csvWriter.close();
            } catch (IOException ignored) {}
            csvWriter = null;
        }
    }

    @Override
    protected void onDestroy() {
        stopPolling();
        try { unregisterReceiver(usbPermissionReceiver); } catch (Exception ignored) {}
        try { unregisterReceiver(usbDetachReceiver); } catch (Exception ignored) {}
        super.onDestroy();
    }
}
