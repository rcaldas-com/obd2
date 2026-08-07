package com.obd2.lambda;

import android.hardware.usb.UsbManager;
import android.util.Log;

import com.hoho.android.usbserial.driver.UsbSerialDriver;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Gerencia comunicação com ELM327 via USB Serial.
 * Suporta chips CH340, FTDI, CP210x, Prolific.
 */
public class Elm327Manager {

    private static final String TAG = "ELM327";
    private static final int BAUD_RATE = 38400;
    private static final int TIMEOUT_MS = 2000;
    private static final int READ_TIMEOUT_MS = 400;  // Reduzido para respostas mais rápidas

    private final UsbSerialSession session = new UsbSerialSession();
    private boolean connected = false;

    public static class LambdaData {
        public Float o2s1Current;    // mA - corrente do sensor O2 banco 1
        public Float o2s5Current;    // mA - corrente do sensor O2 banco 2
        public Float o2s1Lambda;     // lambda ratio banco 1
        public Float o2s5Lambda;     // lambda ratio banco 2
        public Float stft1;          // Short Fuel Trim banco 1 (%)
        public Float stft2;          // Short Fuel Trim banco 2 (%)
        public Integer rpm;
        public Float timingAdvance;
        public long timestamp;

        public String getO2S1Status() {
            if (o2s1Lambda == null) return "SEM DADOS";
            if (o2s1Lambda > 1.02f) return "POBRE";
            if (o2s1Lambda < 0.98f) return "RICO";
            return "ESTEQUIO";
        }
    }

    public static class DashboardData {
        public Integer rpm;
        public Float coolantTemp;     // °C - PID 0105
        public Float intakeAirTemp;   // °C - PID 010F
        public Integer speed;         // km/h - PID 010D
        // Sem ponto de ignição aqui: quem comanda a ignição de verdade agora
        // é a Speeduino, não a ECU original — ver SpeeduinoManager. O ponto
        // original (PID 010E) ainda é lido, só que separado (readStockTimingAdvance),
        // usado apenas como referência no log .msl (MslLogger), não no dashboard.
        public Float tps;             // % - PID 0111 (absoluto)
        public Float batteryVoltage;  // V - AT RV
        public long timestamp;
    }

    /**
     * Conecta ao driver USB já escolhido pelo chamador (a seleção de qual
     * dispositivo é o ELM327, entre os que estiverem plugados, é feita via
     * DeviceRoleManager — ver MainActivity).
     */
    public String connect(UsbManager usbManager, UsbSerialDriver driver) throws IOException {
        String deviceName = session.open(usbManager, driver, BAUD_RATE);

        connected = true;

        // Inicializar ELM327
        initElm327();

        return deviceName;
    }

    /**
     * Inicializa o ELM327 com comandos AT.
     */
    private void initElm327() throws IOException {
        sendCommand("ATZ");          // Reset
        sleep(1500);
        clearBuffer();

        sendCommand("ATE0");         // Echo off
        readResponse();

        sendCommand("ATL0");         // Linefeeds off
        readResponse();

        sendCommand("ATS0");         // Spaces off (respostas mais compactas)
        readResponse();

        sendCommand("ATH0");         // Headers off
        readResponse();

        sendCommand("ATAT2");        // Adaptive timing agressivo - respostas mais rápidas
        readResponse();

        sendCommand("ATST0A");       // Timeout curto (10 * 4ms = 40ms por tentativa)
        readResponse();

        sendCommand("ATSP0");        // Auto protocol
        readResponse();

        // Warm up - primeiro query pode demorar
        sendCommand("0100");
        readResponse();

        Log.i(TAG, "ELM327 inicializado (modo rápido)");
    }

    /**
     * Lê todos os dados de lambda de uma vez.
     */
    public LambdaData readLambdaData() {
        LambdaData data = new LambdaData();
        data.timestamp = System.currentTimeMillis();

        if (!connected || !session.isOpen()) return data;

        // PID 0134 - O2 Sensor 1 Wide Range (Lambda + Current)
        try {
            String resp = queryPid("0134");
            if (resp != null && resp.length() >= 12) {
                // Formato sem espaços: 4134AABBCCDD
                String hex = resp.replace("4134", "").trim();
                if (hex.length() >= 8) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    int b = Integer.parseInt(hex.substring(2, 4), 16);
                    int c = Integer.parseInt(hex.substring(4, 6), 16);
                    int d = Integer.parseInt(hex.substring(6, 8), 16);
                    data.o2s1Current = ((256f * c + d) / 256f) - 128f;
                    // Conversão mA→lambda igual ao script Python
                    if (data.o2s1Current <= 0) {
                        data.o2s1Lambda = 1.0f + (data.o2s1Current * 0.25f);
                    } else {
                        data.o2s1Lambda = 1.0f + (data.o2s1Current * 0.5f);
                    }
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 0134: " + e.getMessage());
        }

        // PID 0138 - O2 Sensor 5 Wide Range
        try {
            String resp = queryPid("0138");
            if (resp != null && resp.length() >= 12) {
                String hex = resp.replace("4138", "").trim();
                if (hex.length() >= 8) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    int b = Integer.parseInt(hex.substring(2, 4), 16);
                    int c = Integer.parseInt(hex.substring(4, 6), 16);
                    int d = Integer.parseInt(hex.substring(6, 8), 16);
                    data.o2s5Current = ((256f * c + d) / 256f) - 128f;
                    // Conversão mA→lambda igual ao script Python
                    if (data.o2s5Current <= 0) {
                        data.o2s5Lambda = 1.0f + (data.o2s5Current * 0.25f);
                    } else {
                        data.o2s5Lambda = 1.0f + (data.o2s5Current * 0.5f);
                    }
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 0138: " + e.getMessage());
        }

        // PIDs 010C (RPM), 0106/0108 (STFT), 010E (Timing) removidos
        // Apenas 2 PIDs (0134 + 0138) = máxima taxa de atualização lambda

        return data;
    }

    /**
     * Lê dados do dashboard: RPM, temp água, temp ar, baro, ponto, TPS.
     * Chamado apenas quando a tela de dashboard está ativa.
     */
    public DashboardData readDashboardData() {
        DashboardData data = new DashboardData();
        data.timestamp = System.currentTimeMillis();

        if (!connected || !session.isOpen()) return data;

        // PID 010C - RPM
        try {
            String resp = queryPid("010C");
            if (resp != null) {
                String hex = resp.replaceAll("^.*410C", "").trim();
                if (hex.length() >= 4) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    int b = Integer.parseInt(hex.substring(2, 4), 16);
                    data.rpm = (256 * a + b) / 4;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 010C: " + e.getMessage());
        }

        data.coolantTemp = readCoolantTemp();

        // PID 010F - Intake Air Temperature
        try {
            String resp = queryPid("010F");
            if (resp != null) {
                String hex = resp.replaceAll("^.*410F", "").trim();
                if (hex.length() >= 2) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    data.intakeAirTemp = a - 40f;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 010F: " + e.getMessage());
        }

        // PID 010D - Vehicle Speed
        try {
            String resp = queryPid("010D");
            if (resp != null) {
                String hex = resp.replaceAll("^.*410D", "").trim();
                if (hex.length() >= 2) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    data.speed = a;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 010D: " + e.getMessage());
        }

        // PID 0111 - Throttle Position (absoluto)
        try {
            String resp = queryPid("0111");
            if (resp != null) {
                String hex = resp.replaceAll("^.*4111", "").trim();
                if (hex.length() >= 2) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    data.tps = (a * 100f) / 255f;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 0111: " + e.getMessage());
        }

        // Voltagem da bateria (comando ELM327 local, leve)
        data.batteryVoltage = readBatteryVoltage();

        return data;
    }

    /**
     * Lê só a temperatura do líquido de arrefecimento (PID 0105, 1 byte). Como
     * é uma consulta única e rápida, pode ser chamada na tela do gráfico em
     * baixa frequência (junto com a voltagem) para os alertas funcionarem
     * independente da tela ativa, sem prejudicar a taxa de leitura do lambda.
     */
    public Float readCoolantTemp() {
        if (!connected || !session.isOpen()) return null;
        try {
            String resp = queryPid("0105");
            if (resp != null) {
                String hex = resp.replaceAll("^.*4105", "").trim();
                if (hex.length() >= 2) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    return a - 40f;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 0105: " + e.getMessage());
        }
        return null;
    }

    /**
     * Lê o ponto de ignição da ECU original (PID 010E) — não usado mais no
     * dashboard (que agora mostra o ponto real da Speeduino), só serve como
     * referência de comparação no log .msl (MslLogger), pra depois replicar
     * manualmente o ponto original nas células do mapa da Speeduino.
     */
    public Float readStockTimingAdvance() {
        if (!connected || !session.isOpen()) return null;
        try {
            String resp = queryPid("010E");
            if (resp != null) {
                String hex = resp.replaceAll("^.*410E", "").trim();
                if (hex.length() >= 2) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    return (a / 2.0f) - 64f;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 010E: " + e.getMessage());
        }
        return null;
    }

    /**
     * Lê só a voltagem da bateria via comando ATRV do ELM327. É um comando
     * LOCAL do adaptador (não consulta a ECU), então é leve — pode ser chamado
     * na tela do gráfico em baixa frequência sem prejudicar a taxa do lambda.
     */
    public Float readBatteryVoltage() {
        if (!connected || !session.isOpen()) return null;
        try {
            sendCommand("ATRV");
            String resp = readResponse();
            if (resp != null) {
                // Resposta tipo "12.6V" / "12.6v" seguida de ">"
                String cleaned = resp.replaceAll("[\\r\\n>\\s]", "").toUpperCase().replace("V", "");
                if (!cleaned.isEmpty()) return Float.parseFloat(cleaned);
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro ATRV: " + e.getMessage());
        }
        return null;
    }

    /**
     * Consulta quais PIDs padrão (Modo 01) a ECU conectada realmente suporta,
     * varrendo os PIDs de "suporte" (0100, 0120, 0140, ...) — cada resposta é
     * um bitmask de 32 PIDs, cujo último bit indica se o próximo bloco existe.
     * Usado nas Configurações para só oferecer PIDs que o veículo confirma ter,
     * em vez de adivinhar.
     */
    public List<String> querySupportedPids() {
        List<String> supported = new ArrayList<>();
        if (!connected || !session.isOpen()) return supported;

        String[] queries = {"0100", "0120", "0140", "0160", "0180", "01A0", "01C0", "01E0"};
        for (String q : queries) {
            String resp = queryPid(q);
            if (resp == null) break;

            String prefix = "41" + q.substring(2);
            String hex = resp.replaceAll("^.*" + prefix, "").trim();
            if (hex.length() < 8) break;

            long mask;
            try {
                mask = Long.parseLong(hex.substring(0, 8), 16);
            } catch (NumberFormatException e) {
                break;
            }

            int baseOffset = Integer.parseInt(q.substring(2), 16);
            for (int bit = 31; bit >= 1; bit--) {
                if (((mask >> bit) & 1) == 1) {
                    int pidNum = baseOffset + (32 - bit);
                    supported.add(String.format("01%02X", pidNum));
                }
            }
            boolean hasNext = (mask & 1) == 1;
            if (!hasNext) break;
        }
        return supported;
    }

    /**
     * Lê e decodifica um PID genérico (ver {@link ObdPid}) — usado pelos
     * alertas personalizados escolhidos pelo usuário nas Configurações.
     */
    public Float readGenericPid(ObdPid def) {
        if (!connected || !session.isOpen() || def == null) return null;
        try {
            String resp = queryPid(def.pid);
            if (resp == null) return null;
            String prefix = "4" + def.pid.substring(2);
            String hex = resp.replaceAll("^.*" + prefix, "").trim();
            if (hex.length() < def.byteCount * 2) return null;
            int[] bytes = new int[def.byteCount];
            for (int i = 0; i < def.byteCount; i++) {
                bytes[i] = Integer.parseInt(hex.substring(i * 2, i * 2 + 2), 16);
            }
            return def.decode(bytes);
        } catch (Exception e) {
            Log.w(TAG, "Erro PID " + def.pid + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Envia um PID OBD2 e retorna a resposta limpa.
     */
    public String queryPid(String pid) {
        try {
            sendCommand(pid);
            String response = readResponse();
            if (response != null) {
                // Limpar a resposta
                response = response.replaceAll("[\\r\\n>\\s]", "").toUpperCase();
                if (response.contains("NODATA") || response.contains("ERROR") ||
                    response.contains("UNABLE") || response.contains("?")) {
                    return null;
                }
                return response;
            }
        } catch (Exception e) {
            Log.w(TAG, "Query " + pid + " falhou: " + e.getMessage());
        }
        return null;
    }

    private void sendCommand(String cmd) throws IOException {
        String toSend = cmd + "\r";
        session.write(toSend.getBytes(StandardCharsets.US_ASCII), TIMEOUT_MS);
    }

    private String readResponse() {
        StringBuilder sb = new StringBuilder();
        byte[] buf = new byte[256];
        long deadline = System.currentTimeMillis() + READ_TIMEOUT_MS;

        try {
            while (System.currentTimeMillis() < deadline) {
                int len = session.read(buf, 200);
                if (len > 0) {
                    sb.append(new String(buf, 0, len, StandardCharsets.US_ASCII));
                    String partial = sb.toString();
                    if (partial.contains(">")) {
                        break;
                    }
                }
            }
        } catch (IOException e) {
            Log.w(TAG, "Read error: " + e.getMessage());
        }
        return sb.toString();
    }

    private void clearBuffer() {
        byte[] buf = new byte[256];
        try {
            while (session.read(buf, 100) > 0) { /* drain */ }
        } catch (IOException ignored) {}
    }

    public void disconnect() {
        connected = false;
        session.close();
    }

    public boolean isConnected() {
        return connected && session.isOpen();
    }

    private void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) {}
    }
}
