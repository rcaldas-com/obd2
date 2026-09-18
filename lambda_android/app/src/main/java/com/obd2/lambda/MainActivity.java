package com.obd2.lambda;

import android.Manifest;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.hardware.usb.UsbManager;
import android.media.AudioManager;
import android.media.ToneGenerator;
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
import androidx.core.app.ActivityCompat;

import com.hoho.android.usbserial.driver.UsbSerialDriver;
import com.hoho.android.usbserial.driver.UsbSerialProber;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
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
    // Rotação/MAP/TPS pela ECU original na tela de ignição (só quando a
    // Speeduino não está disponível) — ver readIgnitionContextStep.
    private static final int IGNITION_CONTEXT_INTERVAL_MS = 500;
    private static final int REQUEST_LOCATION_PERMISSION = 1001;

    // UI Elements
    private TextView tvConnStatus, tvBatteryVoltage, tvRecIndicator;
    private Button btnConnect, btnToggleScreen, btnIgnitionRef, btnPauseLog;
    private LambdaChartView chartView;
    private IgnitionChartView ignitionChartView;
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
    private boolean settingsOpenedFromConnect = false;

    // Adicionar alerta personalizado (escolha de PID)
    private View layoutPidPicker;
    private LinearLayout layoutCustomRules;
    private TextView tvPidSearchStatus, tvPidLiveValue;
    private Spinner spinnerPid, spinnerDirection;
    private EditText etCustomThreshold, etCustomClearMargin;
    private List<ObdPid> pidOptions = new ArrayList<>();
    private String previewPidId = null;
    private boolean previewRunning = false;

    /**
     * Telas que o botão de alternar percorre em ciclo. Cada uma manda numa
     * cadência de leitura diferente do ELM327 (ver pollRunnable): LAMBDA lê só
     * os dois PIDs de lambda, IGNICAO lê só o 010E — em ambos os casos pra
     * manter a taxa alta no que a tela está mostrando —, e DASH lê o pacote
     * lento de informações gerais.
     */
    private enum Screen { LAMBDA, IGNICAO, DASH }

    private Screen screen = Screen.LAMBDA;

    // Logic
    private Elm327Manager elm327;
    private SpeeduinoManager speeduino;
    private DeviceRoleManager deviceRoleManager;
    private MslLogger mslLogger;
    private GpsSpeedProvider gpsSpeedProvider;
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
    private final KnockWatch knockWatch = new KnockWatch();
    /** Última leitura da Speeduino, pra tela de ignição usar rotação/MAP/TPS
     * sem gastar banda do ELM327 (a Speeduino tem porta e thread próprias). */
    private volatile SpeeduinoManager.SpeeduinoData lastSpeeduinoData;
    // Mesmos dados vindos da ECU original, pra tela de ignição continuar
    // inteira quando a Speeduino estiver ocupada com o TunerStudio.
    private Integer oemRpm;
    private Float oemMapKpa;
    private Float oemTpsPct;
    private int ignitionContextStep = 0;
    private long lastIgnitionContextTime = 0;
    private ToneGenerator toneGenerator;

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

    // A multimídia parece suspender antes de desligar de vez (não é sempre
    // um corte abrupto de energia) — esse broadcast padrão do Android avisa
    // antes do desligamento, dando a chance de fechar o log .msl direito
    // (stopPolling() já cuida disso) em vez de confiar só em onDestroy(),
    // que não é garantido disparar a tempo num desligamento do sistema.
    private final BroadcastReceiver shutdownReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            Log.i(TAG, "Desligamento do sistema detectado — parando gravação/polling");
            stopPolling();
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

        gpsSpeedProvider = new GpsSpeedProvider(this);
        if (gpsSpeedProvider.hasPermission()) {
            gpsSpeedProvider.start();
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.ACCESS_FINE_LOCATION}, REQUEST_LOCATION_PERMISSION);
        }

        initViews();
        registerReceivers();

        // Verificar se já há um dispositivo USB conectado
        checkExistingUsb();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_LOCATION_PERMISSION && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            gpsSpeedProvider.start();
        }
    }

    private void initViews() {
        layoutConnect = findViewById(R.id.layout_connect);
        layoutDash = findViewById(R.id.layout_dash);
        btnConnect = findViewById(R.id.btn_connect);
        tvConnStatus = findViewById(R.id.tv_conn_status);

        tvBatteryVoltage = findViewById(R.id.tv_battery_voltage);
        tvRecIndicator = findViewById(R.id.tv_rec_indicator);

        chartView = findViewById(R.id.chart_view);
        ignitionChartView = findViewById(R.id.ignition_chart_view);
        dashboardView = findViewById(R.id.dashboard_view);
        btnToggleScreen = findViewById(R.id.btn_toggle_screen);
        btnIgnitionRef = findViewById(R.id.btn_ignition_ref);
        btnPauseLog = findViewById(R.id.btn_pause_log);

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
        btnIgnitionRef.setOnClickListener(v -> {
            knockWatch.toggleAnchor();
            enableFullscreen();
        });
        btnPauseLog.setOnClickListener(v -> {
            if (mslLogger.isPaused()) {
                mslLogger.resume();
            } else {
                mslLogger.pause();
            }
            updateMslLogButtonUi();
        });
        findViewById(R.id.btn_open_settings).setOnClickListener(v -> openAlertSettings());
        findViewById(R.id.btn_open_settings_from_connect).setOnClickListener(v -> openAlertSettings());
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

        IntentFilter shutdownFilter = new IntentFilter(Intent.ACTION_SHUTDOWN);
        registerReceiver(shutdownReceiver, shutdownFilter);
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
            String key = deviceRoleManager.keyFor(driver, drivers);
            if (!DeviceRoleManager.ROLE_NONE.equals(deviceRoleManager.getRole(key))) {
                anyRoleAssigned = true;
                break;
            }
        }
        // Caso comum (só o ELM327, sem Speeduino): não faz sentido pedir
        // pra configurar nada — só tem um adaptador possível, então assume
        // ELM327 automaticamente, igual o comportamento de sempre antes de
        // existir a atribuição de papéis. Só exige escolha manual quando há
        // 2+ adaptadores (aí sim é ambíguo qual é qual).
        if (!anyRoleAssigned && drivers.size() == 1) {
            String key = deviceRoleManager.keyFor(drivers.get(0), drivers);
            deviceRoleManager.setRole(key, DeviceRoleManager.ROLE_ELM327);
            anyRoleAssigned = true;
        }
        if (!anyRoleAssigned) {
            showStatus("Múltiplos adaptadores detectados — abra ☰ → Dispositivos USB pra escolher qual é qual.");
            return;
        }

        // Pede permissão pra um dispositivo com papel atribuído por vez —
        // fluxo padrão do Android. Ao conceder, usbPermissionReceiver chama
        // requestConnection() de novo, que segue pro próximo sem permissão
        // até todos estarem prontos, e então cai em doConnect().
        for (UsbSerialDriver driver : drivers) {
            String key = deviceRoleManager.keyFor(driver, drivers);
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
            String key = deviceRoleManager.keyFor(driver, drivers);
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
            String speeduinoFailReason = null;
            if (finalSpeeduinoDriver != null) {
                try {
                    speeduino.connect(usbManager, finalSpeeduinoDriver);
                    if (!speeduino.verifySignature()) {
                        speeduinoFailReason = "não respondeu ao handshake";
                        Log.w(TAG, "Speeduino conectada mas assinatura não confere — desconectando");
                        speeduino.disconnect();
                    } else {
                        // Uma vez só por conexão: stoich é config da tune, não
                        // sai no bloco de status ao vivo (ver SpeeduinoManager).
                        speeduino.readStoich();
                    }
                } catch (IOException e) {
                    speeduinoFailReason = e.getMessage();
                    Log.w(TAG, "Falha ao conectar Speeduino: " + e.getMessage());
                }
            }

            final String finalElmDeviceName = elmDeviceName;
            final String finalSpeeduinoFailReason = speeduinoFailReason;
            uiHandler.post(() -> {
                if (elm327.isConnected() || speeduino.isConnected()) {
                    showDashView();
                    StringBuilder status = new StringBuilder();
                    if (elm327.isConnected()) status.append("Conectado: ").append(finalElmDeviceName);
                    else if (finalElmDriver != null) status.append("Falha no ELM327");
                    if (speeduino.isConnected()) {
                        status.append(status.length() > 0 ? " · Speeduino OK" : "Speeduino OK");
                    } else if (finalSpeeduinoDriver != null) {
                        String detail = "Falha na Speeduino" + (finalSpeeduinoFailReason != null ? " (" + finalSpeeduinoFailReason + ")" : "");
                        status.append(status.length() > 0 ? " · " + detail : detail);
                    }
                    showStatus(status.toString());

                    if (elm327.isConnected()) {
                        startPolling();
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

            // Gravando, o ELM327 fica dedicado a lambda banco 1/2 o tempo
            // todo, não importa a tela — é a única fonte que existe (as
            // sondas estão na injeção original, não na Speeduino) e o log
            // precisa disso ao vivo pra corrigir VE, não travado no último
            // valor de quando a tela de lambda foi vista por último. A tela
            // de ponto fica inacessível durante gravação (applyScreen já
            // garante isso), então só resta decidir o que a tela de
            // informações gerais mostra enquanto grava.
            boolean recording = mslLogger.isRecording();

            if (screen == Screen.LAMBDA || recording) {
                final Elm327Manager.LambdaData data = elm327.readLambdaData();
                mslLogger.updateObd2Lambda(data.o2s1Lambda, data.o2s5Lambda);

                if (screen == Screen.LAMBDA) {
                    uiHandler.post(() -> updateUI(data));
                } else if (screen == Screen.DASH) {
                    // Gravando com a tela de informações gerais na frente: nada de
                    // consultar o ELM327 pros PIDs dela (CLT/IAT/velocidade), isso
                    // atrasaria o lambda que o log está contando pra ter ao vivo.
                    // A Speeduino e o GPS já entregam o essencial de graça, em
                    // porta/thread própria, sem custo nenhum aqui.
                    updateDashboardFromSpeeduinoAndGps();
                }
                pollSlowAlerts();
            } else if (screen == Screen.DASH) {
                final Elm327Manager.DashboardData data = elm327.readDashboardData();
                // Dashboard já lê voltagem+temperatura toda vez (não faz PIDs de
                // lambda), então os alertas usam esses mesmos dados; só os
                // alertas personalizados (PID à parte) fazem consulta extra.
                final Map<String, Float> customValues = readCustomAlertValues();
                uiHandler.post(() -> {
                    updateDashboardUI(data);
                    evaluateAlerts(data.batteryVoltage, data.coolantTemp, customValues);
                });
            } else { // Screen.IGNICAO — nunca alcançável gravando (ver applyScreen/toggleScreen)
                // Só o 010E nessa tela: é UMA consulta por volta, então sai
                // ainda mais rápido que a de lambda (que faz duas). O recuo de
                // ponto da original é o evento que se está caçando aqui — a
                // cadência lenta de antes (junto dos alertas, 3s) perdia o
                // evento inteiro entre duas amostras.
                final Float stockAdvance = elm327.readStockTimingAdvance();
                readIgnitionContextStep();
                updateIgnitionScreen(stockAdvance);
                pollSlowAlerts();
            }

            if (polling) {
                pollHandler.postDelayed(this, POLL_INTERVAL_MS);
            }
        }
    };

    /**
     * Tela de informações gerais durante gravação, sem tocar no ELM327: usa
     * o que a Speeduino (porta própria, atualizando sempre) e o GPS já têm.
     * Falta só o que nenhuma das duas tem sem OBD2 (nada crítico aqui —
     * velocidade cai pro GPS, o resto é aproximação razoável do que a tela
     * mostraria via ELM327).
     */
    private void updateDashboardFromSpeeduinoAndGps() {
        SpeeduinoManager.SpeeduinoData sd = lastSpeeduinoData;
        Elm327Manager.DashboardData data = new Elm327Manager.DashboardData();
        data.timestamp = System.currentTimeMillis();
        if (sd != null) {
            data.rpm = sd.rpm;
            data.coolantTemp = sd.coolantC;
            data.intakeAirTemp = sd.iatC;
            data.batteryVoltage = sd.batteryV;
        }
        Float gpsSpeed = gpsSpeedProvider != null ? gpsSpeedProvider.getSpeedKmh() : null;
        if (gpsSpeed != null) data.speed = Math.round(gpsSpeed);

        // Alertas personalizados ficam de fora aqui de propósito: são PID à
        // parte (readCustomAlertValues), voltaria a consultar o ELM327.
        // Voltagem/água continuam avaliados (vêm da Speeduino, de graça).
        uiHandler.post(() -> {
            updateDashboardUI(data);
            evaluateAlerts(data.batteryVoltage, data.coolantTemp, Collections.emptyMap());
        });
    }

    /**
     * Consultas leves de fundo (voltagem, água, alertas personalizados) na
     * cadência lenta — rodam em qualquer tela de gráfico pra não roubar banda
     * do que a tela está mostrando, mas mantendo os alertas de temperatura e
     * bateria vivos durante um teste de pista, que é justamente quando não se
     * pode perder um aviso desses. Roda na thread do pollHandler.
     */
    private void pollSlowAlerts() {
        long now = System.currentTimeMillis();
        if (now - lastAlertCheckTime < ALERT_CHECK_INTERVAL_MS) return;
        lastAlertCheckTime = now;

        final Float v = elm327.readBatteryVoltage();
        final Float temp = elm327.readCoolantTemp();
        final Map<String, Float> customValues = readCustomAlertValues();
        uiHandler.post(() -> {
            if (v != null) setBatteryVoltage(v);
            evaluateAlerts(v, temp, customValues);
        });
    }

    /**
     * Rotação/MAP/TPS pela ECU original, um PID por volta do loop — usados pra
     * saber se a condição está estável quando a Speeduino NÃO está disponível
     * (caso normal durante o ajuste: o TunerStudio no notebook fica com a porta
     * serial dela, que é uma só). Um por volta, e não os três, porque o 010E é
     * o que precisa de taxa alta aqui; estabilidade é um conceito de segundos,
     * então ~3Hz em cada um destes sobra.
     */
    private void readIgnitionContextStep() {
        // Com a Speeduino conectada ela já dá rotação/MAP/TPS de graça (porta
        // própria), então nem consulta a original — o ELM327 fica 100% no 010E.
        if (speeduino.isConnected() && lastSpeeduinoData != null) return;

        // Sem ela, ainda assim espaça as consultas: estabilidade é medida numa
        // janela de segundos, então ~1 leitura de cada a cada 1,5s sobra, e o
        // 010E (que é o sinal que se está caçando) mantém a taxa cheia.
        long now = System.currentTimeMillis();
        if (now - lastIgnitionContextTime < IGNITION_CONTEXT_INTERVAL_MS) return;
        lastIgnitionContextTime = now;

        switch (ignitionContextStep++ % 3) {
            case 0: {
                Float v = elm327.readGenericPid(ObdPid.get("010C"));
                if (v != null) oemRpm = Math.round(v);
                break;
            }
            case 1: {
                Float v = elm327.readGenericPid(ObdPid.get("010B"));
                if (v != null) oemMapKpa = v;
                break;
            }
            default: {
                Float v = elm327.readGenericPid(ObdPid.get("0111"));
                if (v != null) oemTpsPct = v;
                break;
            }
        }
    }

    /**
     * Cruza o ponto recém-lido da original com rotação/MAP/TPS, roda a detecção
     * de recuo e joga tudo na tela. Chamado da thread do pollHandler; só o
     * desenho vai pra thread de UI.
     *
     * A condição é medida pela Speeduino quando ela está conectada (é a carga
     * que os mapas dela usam de verdade), e pela ECU original quando não está —
     * o que mantém a tela inteira funcional com só o ELM327 plugado, que é o
     * cenário de ajustar o ponto pelo TunerStudio com a serial da Speeduino
     * ocupada.
     */
    private void updateIgnitionScreen(Float stockAdvance) {
        SpeeduinoManager.SpeeduinoData sd = speeduino.isConnected() ? lastSpeeduinoData : null;
        Float speeduinoAdvance = sd != null ? sd.advanceDeg : null;

        Integer rpm = sd != null && sd.rpm != null ? sd.rpm : oemRpm;
        Float map = sd != null && sd.mapKpa != null ? sd.mapKpa : oemMapKpa;
        Float tps = sd != null && sd.tpsPct != null ? sd.tpsPct : oemTpsPct;

        final KnockWatch.Status st = knockWatch.update(
                System.currentTimeMillis(), stockAdvance, rpm, map, tps);

        // Copia o que a UI precisa: o Status é reaproveitado a cada volta.
        final KnockWatch.State state = st.state;
        final Float reference = st.reference;
        final Float drop = st.dropDeg;
        final boolean locked = knockWatch.isAnchorLocked();
        final boolean fired = st.eventJustFired;

        uiHandler.post(() -> {
            ignitionChartView.addData(stockAdvance, speeduinoAdvance);
            ignitionChartView.updateStatus(state, reference, drop, locked);
            if (fired) beepKnockEvent();
        });
    }

    /**
     * Bipe curto no instante em que a original recua — o ajuste é feito com o
     * carro em movimento, então o evento precisa chamar atenção sem depender
     * de alguém estar olhando o gráfico naquele segundo. Uma vez por evento
     * (o KnockWatch só marca a transição), nunca por amostra.
     */
    private void beepKnockEvent() {
        try {
            if (toneGenerator == null) {
                toneGenerator = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 80);
            }
            toneGenerator.startTone(ToneGenerator.TONE_PROP_BEEP2, 250);
        } catch (RuntimeException e) {
            // Multimídia de carro nem sempre expõe o stream esperado; o aviso
            // visual (tarja + borda vermelha) já cobre o caso.
            Log.w(TAG, "Sem áudio pro aviso de recuo: " + e.getMessage());
        }
    }

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
            lastSpeeduinoData = data;
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
    }

    private void updateDashboardUI(Elm327Manager.DashboardData data) {
        // Voltagem da bateria na barra inferior
        if (data.batteryVoltage != null) {
            setBatteryVoltage(data.batteryVoltage);
        }

        // GPS do próprio dispositivo quando disponível — o OBD2 às vezes dá
        // velocidade errada nessa instalação; cai pro OBD2 só se não tiver
        // fix de GPS recente.
        Float gpsSpeedKmh = gpsSpeedProvider.getSpeedKmh();
        if (gpsSpeedKmh != null) {
            data.speed = Math.round(gpsSpeedKmh);
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
        // Configurações também pode ser aberta a partir da tela de conexão
        // (antes de conectar, pra atribuir os papéis dos dispositivos USB
        // pela primeira vez) — guarda qual tela estava visível pra voltar
        // pra ela ao fechar, em vez de sempre assumir o dashboard.
        settingsOpenedFromConnect = layoutConnect.getVisibility() == View.VISIBLE;
        layoutConnect.setVisibility(View.GONE);
        layoutDash.setVisibility(View.GONE);
        layoutAlertSettings.setVisibility(View.VISIBLE);
    }

    private void closeAlertSettings() {
        layoutAlertSettings.setVisibility(View.GONE);
        if (settingsOpenedFromConnect) {
            layoutConnect.setVisibility(View.VISIBLE);
        } else {
            layoutDash.setVisibility(View.VISIBLE);
            enableFullscreen();
        }
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
            String key = deviceRoleManager.keyFor(driver, drivers);
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
            applyScreen(); // libera a tela de ponto de novo — atualiza o rótulo do botão de trocar tela
            return;
        }
        try {
            String filename = mslLogger.start();
            Toast.makeText(this, "Gravando: " + filename, Toast.LENGTH_SHORT).show();
            // Ponto não entra mais nesse log (ver MslLogger) e a tela dele
            // disputaria a mesma porta serial que o lambda precisa agora —
            // se estava nela por baixo das Configurações, tira de lá; e o
            // rótulo do botão de trocar tela precisa refletir isso já.
            if (screen == Screen.IGNICAO) {
                screen = Screen.LAMBDA;
            }
            applyScreen();
        } catch (IOException e) {
            Toast.makeText(this, "Erro ao iniciar log: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
        updateMslLogButtonUi();
    }

    /** Atualiza o botão de start/stop (Configurações), o botão de
     * pausar/retomar e o indicador REC/PAUSA — os dois últimos visíveis em
     * qualquer tela, chamado depois de qualquer mudança de estado do
     * MslLogger (start, stop, pause, resume). Sem tique automático: reflete
     * o estado só quando algo muda, suficiente pro caso de uso. */
    private void updateMslLogButtonUi() {
        boolean recording = mslLogger.isRecording();
        boolean paused = mslLogger.isPaused();
        if (recording) {
            btnMslLog.setText("Parar gravação");
            tvMslLogStatus.setText(paused ? "Pausado" : "Gravando…");
        } else {
            btnMslLog.setText("Gravar log");
            tvMslLogStatus.setText("");
        }
        // Visível em cima do dashboard/gráfico também, não só aqui dentro
        // de Configurações — pra não esquecer que está gravando/pausado.
        tvRecIndicator.setVisibility(recording ? View.VISIBLE : View.GONE);
        if (recording) {
            tvRecIndicator.setText(paused ? "II PAUSA" : "● REC");
            tvRecIndicator.setTextColor(paused ? Color.parseColor("#FFD54F") : Color.parseColor("#FF5252"));
        }

        // Botão de pausar/retomar: só existe enquanto grava — não faz
        // sentido pausar uma gravação que não está acontecendo.
        btnPauseLog.setVisibility(recording ? View.VISIBLE : View.GONE);
        btnPauseLog.setText(paused ? ">" : "II");
        btnPauseLog.setTextColor(paused ? Color.parseColor("#81C784") : Color.parseColor("#FF5252"));
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

    /** Ciclo: lambda → ignição → informações gerais → lambda. */
    /** Ciclo: lambda → ponto → informações gerais → lambda — pulando ponto
     * enquanto uma gravação estiver em andamento (ver applyScreen). */
    private void toggleScreen() {
        switch (screen) {
            case LAMBDA:
                screen = mslLogger.isRecording() ? Screen.DASH : Screen.IGNICAO;
                break;
            case IGNICAO: screen = Screen.DASH; break;
            default: screen = Screen.LAMBDA; break;
        }
        applyScreen();
    }

    private void applyScreen() {
        // Voltagem fica visível em todas as telas.
        tvBatteryVoltage.setVisibility(View.VISIBLE);

        chartView.setVisibility(screen == Screen.LAMBDA ? View.VISIBLE : View.GONE);
        ignitionChartView.setVisibility(screen == Screen.IGNICAO ? View.VISIBLE : View.GONE);
        dashboardView.setVisibility(screen == Screen.DASH ? View.VISIBLE : View.GONE);
        btnIgnitionRef.setVisibility(screen == Screen.IGNICAO ? View.VISIBLE : View.GONE);

        // O texto do botão anuncia a PRÓXIMA tela do ciclo. Só glifos que
        // existem em fonte de Android 5 (a multimídia é antiga): "°" é
        // Latin-1, ao contrário de um raio/emoji, que sairia quadradinho.
        // Gravando, o clique a partir de λ pula direto pra informações
        // gerais (ver toggleScreen) — o rótulo tem que anunciar isso, não "°".
        boolean skipIgnicao = mslLogger.isRecording();
        switch (screen) {
            case LAMBDA: btnToggleScreen.setText(skipIgnicao ? "⚙" : "°"); break;
            case IGNICAO: btnToggleScreen.setText("⚙"); break;
            default: btnToggleScreen.setText("λ"); break;
        }

        if (screen == Screen.IGNICAO) {
            // Entrando na tela: o histórico anterior é de outra condição de
            // rodagem (e possivelmente de minutos atrás) — começar limpo evita
            // uma referência fixada em cima de dado velho.
            knockWatch.reset();
            ignitionChartView.clearData();
        }
    }

    private void showConnectView() {
        layoutConnect.setVisibility(View.VISIBLE);
        layoutDash.setVisibility(View.GONE);
        btnConnect.setEnabled(true);
        btnConnect.setText("CONECTAR");
        chartView.clearData();
        ignitionChartView.clearData();
        dashboardView.clearData();
        dashboardView.clearSpeeduinoData();
        knockWatch.reset();
        updateAlertBanners(Collections.emptyList());
        // Reset para tela de lambda como padrão
        screen = Screen.LAMBDA;
        applyScreen();
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

    @Override
    protected void onDestroy() {
        stopPolling();
        gpsSpeedProvider.stop();
        try { unregisterReceiver(usbPermissionReceiver); } catch (Exception ignored) {}
        try { unregisterReceiver(usbDetachReceiver); } catch (Exception ignored) {}
        try { unregisterReceiver(shutdownReceiver); } catch (Exception ignored) {}
        super.onDestroy();
    }
}
