package com.obd2.lambda;

import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbManager;
import android.os.Build;
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
    private static final int POLL_INTERVAL_MS = 50;  // Polling rápido - PIDs já controlam ritmo
    // Na tela do gráfico, a voltagem é lida com esta folga pra não roubar banda
    // das leituras de lambda (que precisam ser rápidas pro ajuste em tempo real).
    private static final int VOLTAGE_INTERVAL_MS = 3000;

    // UI Elements
    private TextView tvConnStatus, tvSampleRate, tvBatteryVoltage;
    private Button btnConnect, btnToggleScreen;
    private LambdaChartView chartView;
    private DashboardView dashboardView;
    private View layoutDash;
    private LinearLayout layoutConnect;

    // Screen mode: false = lambda chart (default), true = dashboard
    private boolean showingDashboard = false;

    // Logic
    private Elm327Manager elm327;
    private UsbManager usbManager;
    private HandlerThread pollThread;
    private Handler pollHandler;
    private Handler uiHandler;
    private boolean polling = false;
    private int sampleCount = 0;
    private long startTime = 0;
    private long lastVoltageReadTime = 0;

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

        tvSampleRate = findViewById(R.id.tv_sample_rate);
        tvBatteryVoltage = findViewById(R.id.tv_battery_voltage);

        chartView = findViewById(R.id.chart_view);
        dashboardView = findViewById(R.id.dashboard_view);
        btnToggleScreen = findViewById(R.id.btn_toggle_screen);

        btnConnect.setOnClickListener(v -> requestConnection());
        findViewById(R.id.btn_disconnect).setOnClickListener(v -> {
            stopPolling();
            showConnectView();
        });
        btnToggleScreen.setOnClickListener(v -> toggleScreen());
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
            int pendingFlags = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
                    ? PendingIntent.FLAG_IMMUTABLE : 0;
            PendingIntent pi = PendingIntent.getBroadcast(this, 0,
                    new Intent(ACTION_USB_PERMISSION),
                    pendingFlags);
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
        lastVoltageReadTime = 0;  // lê a voltagem já no primeiro ciclo do gráfico

        pollThread = new HandlerThread("OBD2Poll");
        pollThread.start();
        pollHandler = new Handler(pollThread.getLooper());
        pollHandler.post(pollRunnable);
    }

    private final Runnable pollRunnable = new Runnable() {
        @Override
        public void run() {
            if (!polling || !elm327.isConnected()) return;

            if (showingDashboard) {
                final Elm327Manager.DashboardData data = elm327.readDashboardData();
                sampleCount++;
                uiHandler.post(() -> updateDashboardUI(data));
            } else {
                final Elm327Manager.LambdaData data = elm327.readLambdaData();
                sampleCount++;
                uiHandler.post(() -> updateUI(data));

                // Voltagem em baixa frequência na tela do gráfico: uma leitura
                // ATRV (leve, local do ELM327) a cada VOLTAGE_INTERVAL_MS, pra
                // manter a taxa de lambda alta.
                long now = System.currentTimeMillis();
                if (now - lastVoltageReadTime >= VOLTAGE_INTERVAL_MS) {
                    lastVoltageReadTime = now;
                    final Float v = elm327.readBatteryVoltage();
                    if (v != null) uiHandler.post(() -> setBatteryVoltage(v));
                }
            }

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

        // Chart - só lambda, sem RPM/timing para máxima velocidade
        chartView.addData(data.o2s1Lambda, data.o2s5Lambda, data.o2s1Current, data.o2s5Current);

        // CSV Log
        writeCsvLine(data);
    }

    private void updateDashboardUI(Elm327Manager.DashboardData data) {
        // Voltagem da bateria na barra inferior
        if (data.batteryVoltage != null) {
            setBatteryVoltage(data.batteryVoltage);
        }

        dashboardView.updateData(data);
    }

    private void setBatteryVoltage(float v) {
        tvBatteryVoltage.setText(String.format(Locale.US, "%.1fV", v));
    }

    private void toggleScreen() {
        showingDashboard = !showingDashboard;
        // Reset sample counter ao trocar de tela para Hz correto
        sampleCount = 0;
        startTime = System.currentTimeMillis();

        // Voltagem fica visível nas duas telas; só o Hz é exclusivo do gráfico.
        tvBatteryVoltage.setVisibility(View.VISIBLE);
        if (showingDashboard) {
            chartView.setVisibility(View.GONE);
            dashboardView.setVisibility(View.VISIBLE);
            btnToggleScreen.setText("λ");
            tvSampleRate.setVisibility(View.GONE);
        } else {
            chartView.setVisibility(View.VISIBLE);
            dashboardView.setVisibility(View.GONE);
            btnToggleScreen.setText("⚙");
            tvSampleRate.setVisibility(View.VISIBLE);
        }
    }

    private void showConnectView() {
        layoutConnect.setVisibility(View.VISIBLE);
        layoutDash.setVisibility(View.GONE);
        btnConnect.setEnabled(true);
        btnConnect.setText("CONECTAR");
        chartView.clearData();
        dashboardView.clearData();
        // Reset para tela de lambda como padrão
        showingDashboard = false;
        chartView.setVisibility(View.VISIBLE);
        dashboardView.setVisibility(View.GONE);
        btnToggleScreen.setText("⚙");
    }

    private void showDashView() {
        layoutConnect.setVisibility(View.GONE);
        layoutDash.setVisibility(View.VISIBLE);
        enableFullscreen();

        // Toque na tela re-ativa fullscreen (para quando volta de outro app)
        chartView.setOnClickListener(v -> enableFullscreen());
    }

    private void enableFullscreen() {
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_FULLSCREEN
                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus && layoutDash.getVisibility() == View.VISIBLE) {
            enableFullscreen();
        }
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
