package com.obd2.lambda;

import android.content.Context;
import android.content.SharedPreferences;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbManager;

import com.hoho.android.usbserial.driver.UsbSerialDriver;

import java.util.List;

/**
 * Atribuição manual de qual dispositivo USB-serial plugado é o ELM327 e
 * qual é a Speeduino — não dá pra distinguir isso por VID/PID (o filtro em
 * device_filter.xml é por chip, não por papel), e a multimídia do carro tem
 * várias portas fixas, então o usuário escolhe uma vez na tela de
 * Configurações e a escolha fica salva.
 *
 * Chave de identidade: vendorId:productId:serialNumber. Quando o adaptador
 * não expõe serial (comum em clones CH340) ou tem permissão ainda não
 * concedida (getSerialNumber() exige isso a partir do Android 10), cai no
 * fallback vendorId:productId:posição — que quebra se dois adaptadores
 * idênticos forem trocados de porta entre uma sessão e outra; a tela avisa
 * isso explicitamente.
 */
public class DeviceRoleManager {

    private static final String PREFS = "device_roles";

    public static final String ROLE_NONE = "NONE";
    public static final String ROLE_ELM327 = "ELM327";
    public static final String ROLE_SPEEDUINO = "SPEEDUINO";

    private final SharedPreferences prefs;

    public DeviceRoleManager(Context context) {
        prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    /** Chave estável (o quanto for possível) para um driver, na posição em
     * que ele aparece na lista atual de drivers do mesmo vendor:product. */
    public String keyFor(UsbManager usbManager, UsbSerialDriver driver, List<UsbSerialDriver> allDrivers) {
        UsbDevice device = driver.getDevice();
        int vendorId = device.getVendorId();
        int productId = device.getProductId();

        String serial = null;
        if (usbManager.hasPermission(device)) {
            try {
                serial = device.getSerialNumber();
            } catch (SecurityException ignored) {
                // Sem permissão mesmo tentando — segue pro fallback por posição.
            }
        }
        if (serial != null && !serial.trim().isEmpty()) {
            return vendorId + ":" + productId + ":" + serial;
        }

        // Fallback: posição entre os drivers com o mesmo vendor:product.
        int position = 0;
        for (UsbSerialDriver d : allDrivers) {
            UsbDevice dd = d.getDevice();
            if (dd.getVendorId() == vendorId && dd.getProductId() == productId) {
                if (d == driver) break;
                position++;
            }
        }
        return vendorId + ":" + productId + ":pos" + position;
    }

    /** Rótulo amigável pra mostrar na tela de Configurações. */
    public String labelFor(UsbManager usbManager, UsbSerialDriver driver) {
        UsbDevice device = driver.getDevice();
        String chip = driver.getClass().getSimpleName();
        String vidPid = String.format("%04X:%04X", device.getVendorId(), device.getProductId());
        String serial = null;
        if (usbManager.hasPermission(device)) {
            try {
                serial = device.getSerialNumber();
            } catch (SecurityException ignored) {}
        }
        if (serial != null && !serial.trim().isEmpty()) {
            return chip + " (" + vidPid + ", serial " + serial + ")";
        }
        return chip + " (" + vidPid + ")";
    }

    public String getRole(String deviceKey) {
        return prefs.getString(deviceKey, ROLE_NONE);
    }

    public void setRole(String deviceKey, String role) {
        prefs.edit().putString(deviceKey, role).apply();
    }
}
