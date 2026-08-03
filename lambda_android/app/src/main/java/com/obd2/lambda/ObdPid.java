package com.obd2.lambda;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Tabela de PIDs padrão do Modo 01 (SAE J1979) que sabemos decodificar, para
 * o seletor "Adicionar alerta" das Configurações — só aparecem lá os PIDs
 * daqui que a ECU conectada realmente confirma suportar (ver
 * Elm327Manager#querySupportedPids). Não inclui os sensores O2 wideband
 * (0134/0138) nem a voltagem (comando AT, não é PID) — esses já têm caminho
 * próprio no app.
 */
public class ObdPid {

    public interface Decoder {
        float decode(int[] bytes);
    }

    public final String pid;      // ex.: "010C"
    public final String name;
    public final String unit;
    public final int byteCount;   // quantos bytes de dado (A, AB, ...) a resposta tem
    private final Decoder decoder;
    private final boolean integerDisplay;

    private ObdPid(String pid, String name, String unit, int byteCount, boolean integerDisplay, Decoder decoder) {
        this.pid = pid;
        this.name = name;
        this.unit = unit;
        this.byteCount = byteCount;
        this.integerDisplay = integerDisplay;
        this.decoder = decoder;
    }

    public float decode(int[] bytes) {
        return decoder.decode(bytes);
    }

    public String format(float value) {
        String num = integerDisplay
                ? String.format(Locale.US, "%.0f", value)
                : String.format(Locale.US, "%.1f", value);
        return unit.isEmpty() ? num : num + " " + unit;
    }

    private static final Map<String, ObdPid> TABLE = new LinkedHashMap<>();

    private static void add(String pid, String name, String unit, int byteCount, boolean integerDisplay, Decoder decoder) {
        TABLE.put(pid, new ObdPid(pid, name, unit, byteCount, integerDisplay, decoder));
    }

    static {
        add("0104", "Carga do motor", "%", 1, false, b -> b[0] * 100f / 255f);
        add("0105", "Temp. arrefecimento", "°C", 1, true, b -> b[0] - 40f);
        add("010A", "Pressão de combustível", "kPa", 1, true, b -> b[0] * 3f);
        add("010B", "Pressão coletor admissão", "kPa", 1, true, b -> (float) b[0]);
        add("010C", "RPM", "rpm", 2, true, b -> (256f * b[0] + b[1]) / 4f);
        add("010D", "Velocidade", "km/h", 1, true, b -> (float) b[0]);
        add("010E", "Ponto de ignição", "°", 1, false, b -> b[0] / 2f - 64f);
        add("010F", "Temp. ar admissão", "°C", 1, true, b -> b[0] - 40f);
        add("0110", "Fluxo de ar (MAF)", "g/s", 2, false, b -> (256f * b[0] + b[1]) / 100f);
        add("0111", "Posição do acelerador", "%", 1, false, b -> b[0] * 100f / 255f);
        add("012F", "Nível de combustível", "%", 1, false, b -> b[0] * 100f / 255f);
        add("0133", "Pressão barométrica", "kPa", 1, true, b -> (float) b[0]);
        add("0146", "Temp. ambiente", "°C", 1, true, b -> b[0] - 40f);
        add("015C", "Temp. óleo do motor", "°C", 1, true, b -> b[0] - 40f);
        add("015E", "Consumo de combustível", "L/h", 2, false, b -> (256f * b[0] + b[1]) / 20f);
    }

    public static ObdPid get(String pid) {
        return TABLE.get(pid);
    }

    public static Map<String, ObdPid> all() {
        return TABLE;
    }
}
