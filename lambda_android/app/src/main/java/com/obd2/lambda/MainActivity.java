package com.obd2.lambda;

import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Color;
import android.graphics.Typeface;
import android.hardware.usb.UsbManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.Spinner;
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "LambdaMonitor";
    private static final String ACTION_USB_PERMISSION = "com.obd2.lambda.USB_PERMISSION";
    private static final int POLL_INTERVAL_MS = 50;  // Polling rápido - PIDs já controlam ritmo
    // Na tela do gráfico, voltagem/temperatura (voltagem + PID 0105) são lidas
    // com esta folga pra não roubar banda das leituras de lambda (que precisam
    // ser rápidas pro ajuste em tempo real). Também é a cadência dos alertas
    // nessa tela — pedido explicitamente em baixa frequência.
    private static final int ALERT_CHECK_INTERVAL_MS = 3000;

    // UI Elements
    private TextView tvConnStatus, tvBatteryVoltage;
    private Button btnConnect, btnToggleScreen;
    private LambdaChartView chartView;
    private DashboardView dashboardView;
    private View layoutDash;
    private LinearLayout layoutConnect;
    private LinearLayout layoutAlerts;
    private View layoutAlertSettings;
    private EditText etVoltageMin, etVoltageHysteresis, etTempMax, etTempHysteresis;
    private List<String> currentAlerts = Collections.emptyList();
    private LinearLayout layoutUsbDevices;
    private Button btnMslLog;
    private TextView tvMslLogStatus;

    // Adicionar alerta personalizado (escolha de PID)
    private View layoutPidPicker;
    private LinearLayout layoutCustomRules;
    private TextView tvPidSearchStatus, tvPidLiveValue;
    private Spinner spinnerPid, spinnerDirection;
    private EditText etCustomThreshold, etCustomClearMargin;
    private List<ObdPid> pidOptions = new ArrayList<>();
    private String previewPidId = null;
    private boolean previewRunning = false;

    // Screen mode: false = lambda chart (default), true = dashboard
    private boolean showingDashboard = false;

    // Logic
    private Elm327Manager elm327;
    private SpeeduinoManager speeduino;
    private DeviceRoleManager deviceRoleManager;
    private MslLogger mslLogger;
    private UsbManager usbManager;
    private HandlerThread pollThread;
    private Handler pollHandler;
    private HandlerThread speeduinoPollThread;
    private Handler speeduinoPollHandler;
    private boolean speeduinoPolling = false;
    private Handler uiHandler;
    private boolean polling = false;
    private long lastAlertCheckTime = 0;
    private AlertManager alertManager;

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
        speeduino = new SpeeduinoManager();
        deviceRoleManager = new DeviceRoleManager(this);
        mslLogger = new MslLogger(this);
        alertManager = new AlertManager(this);
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

        tvBatteryVoltage = findViewById(R.id.tv_battery_voltage);

        chartView = findViewById(R.id.chart_view);
        dashboardView = findViewById(R.id.dashboard_view);
        btnToggleScreen = findViewById(R.id.btn_toggle_screen);

        layoutAlerts = findViewById(R.id.layout_alerts);
        layoutAlertSettings = findViewById(R.id.layout_alert_settings);
        etVoltageMin = findViewById(R.id.et_voltage_min);
        etVoltageHysteresis = findViewById(R.id.et_voltage_hysteresis);
        etTempMax = findViewById(R.id.et_temp_max);
        etTempHysteresis = findViewById(R.id.et_temp_hysteresis);
        layoutCustomRules = findViewById(R.id.layout_custom_rules);
        layoutUsbDevices = findViewById(R.id.layout_usb_devices);
        btnMslLog = findViewById(R.id.btn_msl_log);
        tvMslLogStatus = findViewById(R.id.tv_msl_log_status);

        layoutPidPicker = findViewById(R.id.layout_pid_picker);
        tvPidSearchStatus = findViewById(R.id.tv_pid_search_status);
        tvPidLiveValue = findViewById(R.id.tv_pid_live_value);
        spinnerPid = findViewById(R.id.spinner_pid);
        spinnerDirection = findViewById(R.id.spinner_direction);
        etCustomThreshold = findViewById(R.id.et_custom_threshold);
        etCustomClearMargin = findViewById(R.id.et_custom_clear_margin);

        ArrayAdapter<String> dirAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item,
                new String[]{"Abaixo de", "Acima de"});
        dirAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerDirection.setAdapter(dirAdapter);

        btnConnect.setOnClickListener(v -> requestConnection());
        findViewById(R.id.btn_disconnect).setOnClickListener(v -> {
            stopPolling();
            showConnectView();
        });
        btnToggleScreen.setOnClickListener(v -> toggleScreen());
        findViewById(R.id.btn_open_settings).setOnClickListener(v -> openAlertSettings());
        findViewById(R.id.btn_save_alert_settings).setOnClickListener(v -> saveAlertSettings());
        findViewById(R.id.btn_close_alert_settings).setOnClickListener(v -> closeAlertSettings());
        findViewById(R.id.btn_add_custom_alert).setOnClickListener(v -> openPidPicker());
        findViewById(R.id.btn_cancel_custom_alert).setOnClickListener(v -> closePidPicker());
        findViewById(R.id.btn_confirm_add_custom_alert).setOnClickListener(v -> confirmAddCustomAlert());
        btnMslLog.setOnClickListener(v -> toggleMslLog());

        spinnerPid.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (position >= 0 && position < pidOptions.size()) {
                    startPidPreview(pidOptions.get(position).pid);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
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

        boolean anyRoleAssigned = false;
        for (UsbSerialDriver driver : drivers) {
            String key = deviceRoleManager.keyFor(usbManager, driver, drivers);
            if (!DeviceRoleManager.ROLE_NONE.equals(deviceRoleManager.getRole(key))) {
                anyRoleAssigned = true;
                break;
            }
        }
        if (!anyRoleAssigned) {
            showStatus("Nenhum dispositivo configurado. Abra ☰ → Dispositivos USB.");
            return;
        }

        // Pede permissão pra um dispositivo com papel atribuído por vez —
        // fluxo padrão do Android. Ao conceder, usbPermissionReceiver chama
        // requestConnection() de novo, que segue pro próximo sem permissão
        // até todos estarem prontos, e então cai em doConnect().
        for (UsbSerialDriver driver : drivers) {
            String key = deviceRoleManager.keyFor(usbManager, driver, drivers);
            if (DeviceRoleManager.ROLE_NONE.equals(deviceRoleManager.getRole(key))) continue;
            if (!usbManager.hasPermission(driver.getDevice())) {
                btnConnect.setEnabled(false);
                btnConnect.setText("Aguardando permissão...");
                int pendingFlags = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
                        ? PendingIntent.FLAG_IMMUTABLE : 0;
                PendingIntent pi = PendingIntent.getBroadcast(this, 0,
                        new Intent(ACTION_USB_PERMISSION),
                        pendingFlags);
                usbManager.requestPermission(driver.getDevice(), pi);
                return;
            }
        }

        doConnect();
    }

    private void doConnect() {
        btnConnect.setEnabled(false);
        btnConnect.setText("Conectando...");

        List<UsbSerialDriver> drivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager);
        UsbSerialDriver elmDriver = null;
        UsbSerialDriver speeduinoDriver = null;
        for (UsbSerialDriver driver : drivers) {
            String key = deviceRoleManager.keyFor(usbManager, driver, drivers);
            String role = deviceRoleManager.getRole(key);
            if (DeviceRoleManager.ROLE_ELM327.equals(role)) elmDriver = driver;
            else if (DeviceRoleManager.ROLE_SPEEDUINO.equals(role)) speeduinoDriver = driver;
        }
        final UsbSerialDriver finalElmDriver = elmDriver;
        final UsbSerialDriver finalSpeeduinoDriver = speeduinoDriver;

        if (finalElmDriver == null && finalSpeeduinoDriver == null) {
            showStatus("Nenhum dos dispositivos atribuídos está plugado agora.");
            btnConnect.setEnabled(true);
            btnConnect.setText("CONECTAR");
            return;
        }

        new Thread(() -> {
            String elmDeviceName = null;
            if (finalElmDriver != null) {
                try {
                    elmDeviceName = elm327.connect(usbManager, finalElmDriver);
                } catch (IOException e) {
                    Log.w(TAG, "Falha ao conectar ELM327: " + e.getMessage());
                }
            }
            if (finalSpeeduinoDriver != null) {
                try {
                    speeduino.connect(usbManager, finalSpeeduinoDriver);
                    if (!speeduino.verifySignature()) {
                        Log.w(TAG, "Speeduino conectada mas assinatura não confere — desconectando");
                        speeduino.disconnect();
                    }
                } catch (IOException e) {
                    Log.w(TAG, "Falha ao conectar Speeduino: " + e.getMessage());
                }
            }

            final String finalElmDeviceName = elmDeviceName;
            uiHandler.post(() -> {
                if (elm327.isConnected() || speeduino.isConnected()) {
                    showDashView();
                    StringBuilder status = new StringBuilder();
                    if (elm327.isConnected()) status.append("Conectado: ").append(finalElmDeviceName);
                    else if (finalElmDriver != null) status.append("Falha no ELM327");
                    if (speeduino.isConnected()) status.append(status.length() > 0 ? " · Speeduino OK" : "Speeduino OK");
                    else if (finalSpeeduinoDriver != null) status.append(status.length() > 0 ? " · falha na Speeduino" : "Falha na Speeduino");
                    showStatus(status.toString());

                    if (elm327.isConnected()) {
                        startPolling();
                        startCsvLog();
                    }
                    if (speeduino.isConnected()) {
                        startSpeeduinoPolling();
                    }
                    // Foreground service para o Android 9 não matar o app
                    startForegroundService(new Intent(MainActivity.this, OBD2ForegroundService.class));
                } else {
                    showStatus("Falha ao conectar nos dispositivos atribuídos.");
                    btnConnect.setEnabled(true);
                    btnConnect.setText("CONECTAR");
                }
            });
        }).start();
    }

    private void startPolling() {
        if (polling) return;
        polling = true;
        lastAlertCheckTime = 0;  // lê voltagem/temperatura já no primeiro ciclo do gráfico

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
                // Dashboard já lê voltagem+temperatura toda vez (não faz PIDs de
                // lambda), então os alertas usam esses mesmos dados; só os
                // alertas personalizados (PID à parte) fazem consulta extra.
                final Map<String, Float> customValues = readCustomAlertValues();
                uiHandler.post(() -> {
                    updateDashboardUI(data);
                    evaluateAlerts(data.batteryVoltage, data.coolantTemp, customValues);
                });
            } else {
                final Elm327Manager.LambdaData data = elm327.readLambdaData();
                uiHandler.post(() -> updateUI(data));

                // Voltagem + temperatura + alertas personalizados em baixa
                // frequência na tela do gráfico: consultas leves a cada
                // ALERT_CHECK_INTERVAL_MS, pra manter a taxa de lambda alta e
                // ainda assim os alertas funcionarem independente da tela ativa.
                long now = System.currentTimeMillis();
                if (now - lastAlertCheckTime >= ALERT_CHECK_INTERVAL_MS) {
                    lastAlertCheckTime = now;
                    final Float v = elm327.readBatteryVoltage();
                    final Float temp = elm327.readCoolantTemp();
                    // Ponto da ECU original — só usado como referência no
                    // log .msl (a Speeduino é quem manda de verdade agora),
                    // por isso lido na mesma cadência baixa da bateria/água.
                    mslLogger.updateObd2Advance(elm327.readStockTimingAdvance());
                    final Map<String, Float> customValues = readCustomAlertValues();
                    uiHandler.post(() -> {
                        if (v != null) setBatteryVoltage(v);
                        evaluateAlerts(v, temp, customValues);
                    });
                }
            }

            if (polling) {
                pollHandler.postDelayed(this, POLL_INTERVAL_MS);
            }
        }
    };

    private void startSpeeduinoPolling() {
        if (speeduinoPolling) return;
        speeduinoPolling = true;

        speeduinoPollThread = new HandlerThread("SpeeduinoPoll");
        speeduinoPollThread.start();
        speeduinoPollHandler = new Handler(speeduinoPollThread.getLooper());
        speeduinoPollHandler.post(speeduinoPollRunnable);
    }

    /** Independente do pollRunnable do ELM327 — porta USB própria, sem
     * disputa. Roda sempre (não só na tela do dashboard), pra alimentar o
     * MslLogger continuamente mesmo com o gráfico de lambda em primeiro
     * plano; atualizar a UI do dashboard quando ele não estiver visível não
     * tem custo perceptível (a View simplesmente não desenha enquanto GONE). */
    private final Runnable speeduinoPollRunnable = new Runnable() {
        @Override
        public void run() {
            if (!speeduinoPolling || !speeduino.isConnected()) return;

            final SpeeduinoManager.SpeeduinoData data = speeduino.readOutputChannels();
            mslLogger.updateSpeeduino(data);
            uiHandler.post(() -> dashboardView.updateSpeeduinoData(data));

            if (speeduinoPolling) {
                speeduinoPollHandler.postDelayed(this, POLL_INTERVAL_MS);
            }
        }
    };

    private void stopSpeeduinoPolling() {
        speeduinoPolling = false;
        if (speeduinoPollThread != null) {
            speeduinoPollThread.quitSafely();
            speeduinoPollThread = null;
        }
        speeduino.disconnect();
        dashboardView.clearSpeeduinoData();
    }

    /** Lê o valor atual de cada alerta personalizado configurado. Chamado só
     * do thread de polling (nunca da UI thread) — mesma serialização de todo
     * I/O do ELM327. */
    private Map<String, Float> readCustomAlertValues() {
        List<AlertManager.CustomAlertRule> rules = alertManager.getCustomRules();
        if (rules.isEmpty()) return Collections.emptyMap();
        Map<String, Float> values = new HashMap<>();
        for (AlertManager.CustomAlertRule rule : rules) {
            ObdPid def = ObdPid.get(rule.pid);
            if (def == null) continue;
            Float v = elm327.readGenericPid(def);
            if (v != null) values.put(rule.pid, v);
        }
        return values;
    }

    // Prévia do valor ao vivo na tela "Adicionar Alerta". Postada no MESMO
    // pollHandler (thread único de I/O serial) que o pollRunnable principal —
    // nunca uma thread separada, pra não correr com as consultas em andamento
    // na porta serial. Continua rodando em baixa frequência mesmo com o
    // gráfico ativo em paralelo (interleaving seguro via fila do Handler).
    private final Runnable pidPreviewRunnable = new Runnable() {
        @Override
        public void run() {
            if (!previewRunning || previewPidId == null || !elm327.isConnected()) return;
            ObdPid def = ObdPid.get(previewPidId);
            final Float value = def != null ? elm327.readGenericPid(def) : null;
            uiHandler.post(() -> {
                if (def != null) {
                    tvPidLiveValue.setText(value != null ? def.format(value) : "sem dado");
                }
            });
            if (previewRunning) {
                pollHandler.postDelayed(this, 1500);
            }
        }
    };

    private void stopPolling() {
        polling = false;
        previewRunning = false;
        if (pollThread != null) {
            pollThread.quitSafely();
            pollThread = null;
        }
        elm327.disconnect();
        stopCsvLog();
        stopSpeeduinoPolling();
        if (mslLogger.isRecording()) {
            mslLogger.stop();
            updateMslLogButtonUi();
        }
        // Parar foreground service
        stopService(new Intent(this, OBD2ForegroundService.class));
    }

    private void updateUI(Elm327Manager.LambdaData data) {
        // Chart - só lambda, sem RPM/timing para máxima velocidade
        chartView.addData(data.o2s1Lambda, data.o2s5Lambda, data.o2s1Current, data.o2s5Current);
        mslLogger.updateObd2Lambda(data.o2s1Lambda, data.o2s5Lambda);

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

    // ---- Alertas ----

    private void evaluateAlerts(Float voltage, Float coolantTemp, Map<String, Float> customValues) {
        List<String> alerts = alertManager.evaluate(voltage, coolantTemp, customValues);
        updateAlertBanners(alerts);
    }

    /** Sincroniza as tarjas vermelhas exibidas com a lista de alertas ativos. */
    private void updateAlertBanners(List<String> messages) {
        if (messages.equals(currentAlerts)) return;
        currentAlerts = messages;

        layoutAlerts.removeAllViews();
        float density = getResources().getDisplayMetrics().density;
        for (String msg : messages) {
            TextView banner = new TextView(this);
            banner.setText(msg);
            banner.setTextColor(Color.WHITE);
            banner.setTextSize(22f);
            banner.setTypeface(Typeface.DEFAULT_BOLD);
            banner.setGravity(Gravity.CENTER);
            banner.setBackgroundColor(Color.parseColor("#D32F2F"));
            banner.setPadding(0, (int) (14 * density), 0, (int) (14 * density));

            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
            lp.topMargin = (int) (2 * density);
            lp.bottomMargin = (int) (2 * density);
            banner.setLayoutParams(lp);

            layoutAlerts.addView(banner);
        }
    }

    private void openAlertSettings() {
        etVoltageMin.setText(trimZero(alertManager.getVoltageMin()));
        etVoltageHysteresis.setText(trimZero(alertManager.getVoltageHysteresis()));
        etTempMax.setText(trimZero(alertManager.getTempMax()));
        etTempHysteresis.setText(trimZero(alertManager.getTempHysteresis()));
        renderCustomRulesList();
        renderUsbDevicesList();
        updateMslLogButtonUi();
        layoutDash.setVisibility(View.GONE);
        layoutAlertSettings.setVisibility(View.VISIBLE);
    }

    private void closeAlertSettings() {
        layoutAlertSettings.setVisibility(View.GONE);
        layoutDash.setVisibility(View.VISIBLE);
        enableFullscreen();
    }

    private void saveAlertSettings() {
        try {
            float voltageMin = Float.parseFloat(etVoltageMin.getText().toString().trim().replace(',', '.'));
            float voltageHysteresis = Float.parseFloat(etVoltageHysteresis.getText().toString().trim().replace(',', '.'));
            float tempMax = Float.parseFloat(etTempMax.getText().toString().trim().replace(',', '.'));
            float tempHysteresis = Float.parseFloat(etTempHysteresis.getText().toString().trim().replace(',', '.'));
            alertManager.saveSettings(voltageMin, voltageHysteresis, tempMax, tempHysteresis);
            Toast.makeText(this, "Configurações de alerta salvas", Toast.LENGTH_SHORT).show();
            closeAlertSettings();
        } catch (NumberFormatException e) {
            Toast.makeText(this, "Valores inválidos", Toast.LENGTH_SHORT).show();
        }
    }

    private String trimZero(float v) {
        String s = String.format(Locale.US, "%.1f", v);
        return s.endsWith(".0") ? s.substring(0, s.length() - 2) : s;
    }

    /** Preenche a lista de alertas personalizados já configurados, cada um com
     * um botão pra remover. */
    private void renderCustomRulesList() {
        layoutCustomRules.removeAllViews();
        List<AlertManager.CustomAlertRule> rules = alertManager.getCustomRules();
        float density = getResources().getDisplayMetrics().density;

        if (rules.isEmpty()) {
            TextView empty = new TextView(this);
            empty.setText("Nenhum alerta personalizado ainda.");
            empty.setTextColor(Color.parseColor("#666666"));
            empty.setTextSize(13f);
            layoutCustomRules.addView(empty);
            return;
        }

        for (int i = 0; i < rules.size(); i++) {
            AlertManager.CustomAlertRule rule = rules.get(i);
            final int index = i;

            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setPadding(0, (int) (4 * density), 0, (int) (4 * density));

            TextView label = new TextView(this);
            label.setText(rule.describe());
            label.setTextColor(Color.parseColor("#EEEEEE"));
            label.setTextSize(14f);
            LinearLayout.LayoutParams labelLp = new LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
            row.addView(label, labelLp);

            TextView remove = new TextView(this);
            remove.setText("✕");
            remove.setTextColor(Color.parseColor("#FF5252"));
            remove.setTextSize(18f);
            remove.setPadding((int) (12 * density), 0, (int) (4 * density), 0);
            remove.setOnClickListener(v -> {
                alertManager.removeCustomRule(index);
                renderCustomRulesList();
            });
            row.addView(remove);

            layoutCustomRules.addView(row);
        }
    }

    /** Lista os adaptadores USB-serial plugados agora, cada um com um
     * seletor de papel (Nenhum/ELM327/Speeduino) — não dá pra distinguir
     * automaticamente por VID/PID, então o usuário escolhe uma vez e a
     * escolha fica salva (DeviceRoleManager). */
    private void renderUsbDevicesList() {
        layoutUsbDevices.removeAllViews();
        List<UsbSerialDriver> drivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager);
        float density = getResources().getDisplayMetrics().density;

        if (drivers.isEmpty()) {
            TextView empty = new TextView(this);
            empty.setText("Nenhum adaptador USB detectado agora.");
            empty.setTextColor(Color.parseColor("#666666"));
            empty.setTextSize(13f);
            layoutUsbDevices.addView(empty);
            return;
        }

        String[] roles = {DeviceRoleManager.ROLE_NONE, DeviceRoleManager.ROLE_ELM327, DeviceRoleManager.ROLE_SPEEDUINO};
        String[] roleLabels = {"Nenhum", "ELM327", "Speeduino"};
        ArrayAdapter<String> roleAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, roleLabels);
        roleAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        for (UsbSerialDriver driver : drivers) {
            String key = deviceRoleManager.keyFor(usbManager, driver, drivers);
            String currentRole = deviceRoleManager.getRole(key);

            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setPadding(0, (int) (4 * density), 0, (int) (4 * density));

            TextView label = new TextView(this);
            label.setText(deviceRoleManager.labelFor(usbManager, driver));
            label.setTextColor(Color.parseColor("#EEEEEE"));
            label.setTextSize(13f);
            LinearLayout.LayoutParams labelLp = new LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
            row.addView(label, labelLp);

            Spinner roleSpinner = new Spinner(this);
            roleSpinner.setAdapter(roleAdapter);
            int selection = 0;
            for (int i = 0; i < roles.length; i++) {
                if (roles[i].equals(currentRole)) { selection = i; break; }
            }
            roleSpinner.setSelection(selection, false);
            roleSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                @Override
                public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                    deviceRoleManager.setRole(key, roles[position]);
                }

                @Override
                public void onNothingSelected(AdapterView<?> parent) {}
            });
            row.addView(roleSpinner);

            layoutUsbDevices.addView(row);
        }
    }

    // ---- Log combinado .msl ----

    private void toggleMslLog() {
        if (mslLogger.isRecording()) {
            mslLogger.stop();
            updateMslLogButtonUi();
            return;
        }
        try {
            String filename = mslLogger.start();
            Toast.makeText(this, "Gravando: " + filename, Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            Toast.makeText(this, "Erro ao iniciar log: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
        updateMslLogButtonUi();
    }

    /** Atualiza o botão/status enquanto a tela de Configurações estiver
     * aberta — sem tique automático: o texto reflete o estado só quando a
     * tela é (re)aberta ou o botão é tocado, suficiente pro caso de uso
     * (gravação é um "liga/desliga" ocasional, não precisa de cronômetro
     * ao vivo). */
    private void updateMslLogButtonUi() {
        if (mslLogger.isRecording()) {
            btnMslLog.setText("Parar gravação");
            tvMslLogStatus.setText("Gravando…");
        } else {
            btnMslLog.setText("Gravar log");
            tvMslLogStatus.setText("");
        }
    }

    // ---- Adicionar alerta personalizado (escolha de PID) ----

    private void openPidPicker() {
        spinnerPid.setVisibility(View.GONE);
        tvPidLiveValue.setText("");
        etCustomThreshold.setText("");
        etCustomClearMargin.setText("");
        tvPidSearchStatus.setText("Buscando PIDs suportados pelo veículo...");
        layoutAlertSettings.setVisibility(View.GONE);
        layoutPidPicker.setVisibility(View.VISIBLE);
        searchSupportedPids();
    }

    private void closePidPicker() {
        previewRunning = false;
        previewPidId = null;
        layoutPidPicker.setVisibility(View.GONE);
        layoutAlertSettings.setVisibility(View.VISIBLE);
    }

    private void searchSupportedPids() {
        if (!elm327.isConnected() || pollHandler == null) {
            tvPidSearchStatus.setText("Não conectado ao veículo.");
            return;
        }
        // Posta no MESMO handler de I/O serial do polling principal — nunca
        // uma thread separada, pra não correr com as consultas em andamento.
        pollHandler.post(() -> {
            List<String> supportedIds = elm327.querySupportedPids();
            List<ObdPid> options = new ArrayList<>();
            for (String pid : supportedIds) {
                ObdPid def = ObdPid.get(pid);
                if (def != null) options.add(def);
            }
            uiHandler.post(() -> showPidOptions(options));
        });
    }

    private void showPidOptions(List<ObdPid> options) {
        pidOptions = options;
        if (options.isEmpty()) {
            tvPidSearchStatus.setText("Nenhum PID reconhecido suportado por este veículo.");
            return;
        }
        tvPidSearchStatus.setText("Escolha o sinal para o alerta:");
        List<String> names = new ArrayList<>();
        for (ObdPid def : options) names.add(def.name);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, names);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerPid.setAdapter(adapter);
        spinnerPid.setVisibility(View.VISIBLE);
        startPidPreview(options.get(0).pid);
    }

    private void startPidPreview(String pid) {
        previewPidId = pid;
        tvPidLiveValue.setText("lendo...");
        if (!previewRunning) {
            previewRunning = true;
            pollHandler.post(pidPreviewRunnable);
        }
    }

    private void confirmAddCustomAlert() {
        int position = spinnerPid.getSelectedItemPosition();
        if (position < 0 || position >= pidOptions.size()) {
            Toast.makeText(this, "Escolha um sinal.", Toast.LENGTH_SHORT).show();
            return;
        }
        String thresholdText = etCustomThreshold.getText().toString().trim().replace(',', '.');
        float threshold;
        try {
            threshold = Float.parseFloat(thresholdText);
        } catch (NumberFormatException e) {
            Toast.makeText(this, "Informe o limiar.", Toast.LENGTH_SHORT).show();
            return;
        }
        ObdPid def = pidOptions.get(position);
        boolean above = spinnerDirection.getSelectedItemPosition() == 1; // 0=Abaixo de, 1=Acima de

        // Campo vazio = histerese automática (2% do limiar, mínimo 0.5) —
        // ver AlertManager.CustomAlertRule.effectiveClearMargin().
        Float clearMargin = null;
        String marginText = etCustomClearMargin.getText().toString().trim().replace(',', '.');
        if (!marginText.isEmpty()) {
            try {
                clearMargin = Float.parseFloat(marginText);
            } catch (NumberFormatException e) {
                Toast.makeText(this, "Histerese inválida.", Toast.LENGTH_SHORT).show();
                return;
            }
        }

        alertManager.addCustomRule(new AlertManager.CustomAlertRule(def.pid, def.name, def.unit, threshold, above, clearMargin));
        Toast.makeText(this, "Alerta adicionado", Toast.LENGTH_SHORT).show();
        closePidPicker();
        renderCustomRulesList();
    }

    private void toggleScreen() {
        showingDashboard = !showingDashboard;

        // Voltagem fica visível nas duas telas.
        tvBatteryVoltage.setVisibility(View.VISIBLE);
        if (showingDashboard) {
            chartView.setVisibility(View.GONE);
            dashboardView.setVisibility(View.VISIBLE);
            btnToggleScreen.setText("λ");
        } else {
            chartView.setVisibility(View.VISIBLE);
            dashboardView.setVisibility(View.GONE);
            btnToggleScreen.setText("⚙");
        }
    }

    private void showConnectView() {
        layoutConnect.setVisibility(View.VISIBLE);
        layoutDash.setVisibility(View.GONE);
        btnConnect.setEnabled(true);
        btnConnect.setText("CONECTAR");
        chartView.clearData();
        dashboardView.clearData();
        dashboardView.clearSpeeduinoData();
        updateAlertBanners(Collections.emptyList());
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
        boolean onDashOrSettings = layoutDash.getVisibility() == View.VISIBLE
                || layoutAlertSettings.getVisibility() == View.VISIBLE
                || layoutPidPicker.getVisibility() == View.VISIBLE;
        if (hasFocus && onDashOrSettings) {
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
