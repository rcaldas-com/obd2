package com.obd2.lambda;

import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbManager;

import com.hoho.android.usbserial.driver.UsbSerialDriver;
import com.hoho.android.usbserial.driver.UsbSerialPort;

import java.io.IOException;

/**
 * Transporte USB-serial puro (abrir/configurar/ler/escrever/fechar), sem
 * nenhum conhecimento de protocolo — usado tanto por Elm327Manager quanto
 * por SpeeduinoManager, cada um com sua própria instância/porta física,
 * independentes entre si.
 */
public class UsbSerialSession {

    private UsbSerialPort port;
    private UsbDeviceConnection connection;

    /** Abre e configura a porta do driver já escolhido pelo chamador (a escolha
     * de qual dispositivo USB usar é responsabilidade de quem chama, via
     * DeviceRoleManager — esta classe não enumera nem filtra dispositivos). */
    public String open(UsbManager usbManager, UsbSerialDriver driver, int baudRate) throws IOException {
        UsbDevice device = driver.getDevice();
        String deviceName = device.getDeviceName() + " (" + driver.getClass().getSimpleName() + ")";

        connection = usbManager.openDevice(device);
        if (connection == null) {
            throw new IOException("Sem permissão USB. Reconecte o adaptador.");
        }

        port = driver.getPorts().get(0);
        port.open(connection);
        port.setParameters(baudRate, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE);
        port.setDTR(true);
        port.setRTS(true);
        return deviceName;
    }

    public int read(byte[] buf, int timeoutMs) throws IOException {
        return port.read(buf, timeoutMs);
    }

    public void write(byte[] data, int timeoutMs) throws IOException {
        port.write(data, timeoutMs);
    }

    public void close() {
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

    public boolean isOpen() {
        return port != null;
    }
}
