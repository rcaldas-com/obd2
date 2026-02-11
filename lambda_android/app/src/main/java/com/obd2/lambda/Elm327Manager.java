package com.obd2.lambda;

import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbManager;
import android.util.Log;

import com.hoho.android.usbserial.driver.UsbSerialDriver;
import com.hoho.android.usbserial.driver.UsbSerialPort;
import com.hoho.android.usbserial.driver.UsbSerialProber;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Gerencia comunicação com ELM327 via USB Serial.
 * Suporta chips CH340, FTDI, CP210x, Prolific.
 */
public class Elm327Manager {

    private static final String TAG = "ELM327";
    private static final int BAUD_RATE = 38400;
    private static final int TIMEOUT_MS = 2000;
    private static final int READ_TIMEOUT_MS = 1000;

    private UsbSerialPort port;
    private UsbDeviceConnection connection;
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
            if (o2s1Current == null) return "SEM DADOS";
            if (o2s1Current < -0.01f) return "POBRE";
            if (o2s1Current <= 0.01f) return "ESTEQUIO";
            return "RICO";
        }
    }

    /**
     * Conecta ao primeiro dispositivo USB serial encontrado.
     */
    public String connect(UsbManager usbManager) throws IOException {
        List<UsbSerialDriver> drivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager);

        if (drivers.isEmpty()) {
            throw new IOException("Nenhum adaptador USB serial encontrado");
        }

        UsbSerialDriver driver = drivers.get(0);
        UsbDevice device = driver.getDevice();
        String deviceName = device.getDeviceName() + " (" + driver.getClass().getSimpleName() + ")";

        connection = usbManager.openDevice(device);
        if (connection == null) {
            throw new IOException("Sem permissão USB. Reconecte o adaptador.");
        }

        port = driver.getPorts().get(0);
        port.open(connection);
        port.setParameters(BAUD_RATE, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE);
        port.setDTR(true);
        port.setRTS(true);

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

        sendCommand("ATSP0");        // Auto protocol
        readResponse();

        // Warm up - primeiro query pode demorar
        sendCommand("0100");
        readResponse();

        Log.i(TAG, "ELM327 inicializado com sucesso");
    }

    /**
     * Lê todos os dados de lambda de uma vez.
     */
    public LambdaData readLambdaData() {
        LambdaData data = new LambdaData();
        data.timestamp = System.currentTimeMillis();

        if (!connected || port == null) return data;

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
                    data.o2s1Lambda = (2.0f / 65536f) * (256 * a + b);
                    data.o2s1Current = ((256f * c + d) / 256f) - 128f;
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
                    data.o2s5Lambda = (2.0f / 65536f) * (256 * a + b);
                    data.o2s5Current = ((256f * c + d) / 256f) - 128f;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 0138: " + e.getMessage());
        }

        // PID 010C - RPM
        try {
            String resp = queryPid("010C");
            if (resp != null && resp.length() >= 8) {
                String hex = resp.replace("410C", "").trim();
                if (hex.length() >= 4) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    int b = Integer.parseInt(hex.substring(2, 4), 16);
                    data.rpm = (256 * a + b) / 4;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 010C: " + e.getMessage());
        }

        // PID 0106 - Short Fuel Trim Bank 1
        try {
            String resp = queryPid("0106");
            if (resp != null && resp.length() >= 6) {
                String hex = resp.replace("4106", "").trim();
                if (hex.length() >= 2) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    data.stft1 = (a / 1.28f) - 100f;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 0106: " + e.getMessage());
        }

        // PID 0108 - Short Fuel Trim Bank 2
        try {
            String resp = queryPid("0108");
            if (resp != null && resp.length() >= 6) {
                String hex = resp.replace("4108", "").trim();
                if (hex.length() >= 2) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    data.stft2 = (a / 1.28f) - 100f;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 0108: " + e.getMessage());
        }

        // PID 010E - Timing Advance
        try {
            String resp = queryPid("010E");
            if (resp != null && resp.length() >= 6) {
                String hex = resp.replace("410E", "").trim();
                if (hex.length() >= 2) {
                    int a = Integer.parseInt(hex.substring(0, 2), 16);
                    data.timingAdvance = (a / 2.0f) - 64f;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Erro PID 010E: " + e.getMessage());
        }

        return data;
    }

    /**
     * Envia um PID OBD2 e retorna a resposta limpa.
     */
    private String queryPid(String pid) {
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
        port.write(toSend.getBytes(StandardCharsets.US_ASCII), TIMEOUT_MS);
    }

    private String readResponse() {
        StringBuilder sb = new StringBuilder();
        byte[] buf = new byte[256];
        long deadline = System.currentTimeMillis() + READ_TIMEOUT_MS;

        try {
            while (System.currentTimeMillis() < deadline) {
                int len = port.read(buf, 200);
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
            while (port.read(buf, 100) > 0) { /* drain */ }
        } catch (IOException ignored) {}
    }

    public void disconnect() {
        connected = false;
        if (port != null) {
            try {
                port.close();
            } catch (IOException ignored) {}
            port = null;
        }
        if (connection != null) {
            connection.close();
            connection = null;
        }
    }

    public boolean isConnected() {
        return connected && port != null;
    }

    private void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) {}
    }
}
