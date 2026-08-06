package com.obd2.lambda;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Limiares de alerta (voltagem baixa, temperatura alta, + alertas
 * personalizados por PID) e avaliação dos valores lidos contra eles.
 * Persistido em SharedPreferences, editável na tela de Configurações
 * (botão ☰ na barra inferior).
 *
 * Usa histerese configurável: o alerta só desarma quando o valor volta a uma
 * margem além do limiar, não assim que cruza de volta — evita o alerta
 * piscando quando o valor oscila bem em cima do limite.
 */
public class AlertManager {

    private static final String PREFS = "alert_settings";
    private static final String KEY_VOLTAGE_MIN = "voltage_min";
    private static final String KEY_VOLTAGE_HYSTERESIS = "voltage_hysteresis";
    private static final String KEY_TEMP_MAX = "temp_max";
    private static final String KEY_TEMP_HYSTERESIS = "temp_hysteresis";
    private static final String KEY_CUSTOM_RULES = "custom_rules";

    public static final float DEFAULT_VOLTAGE_MIN = 13.0f;
    public static final float DEFAULT_TEMP_MAX = 105.0f;

    public static final float DEFAULT_VOLTAGE_HYSTERESIS = 0.3f;
    public static final float DEFAULT_TEMP_HYSTERESIS = 3f;

    /** Uma regra de alerta personalizada, escolhida por PID nas Configurações. */
    public static class CustomAlertRule {
        public final String pid;      // ex.: "010C"
        public final String name;
        public final String unit;
        public final float threshold;
        public final boolean aboveTriggers; // true: alerta quando valor > limiar; false: quando valor < limiar
        // Distância do limiar pra voltar ao normal; null = automático (2% do
        // limiar, mínimo 0.5) — mantém regras antigas (sem esse campo) funcionando.
        public final Float clearMargin;
        boolean active = false;       // estado da histerese, mantido entre chamadas de evaluate()

        public CustomAlertRule(String pid, String name, String unit, float threshold, boolean aboveTriggers, Float clearMargin) {
            this.pid = pid;
            this.name = name;
            this.unit = unit;
            this.threshold = threshold;
            this.aboveTriggers = aboveTriggers;
            this.clearMargin = clearMargin;
        }

        float effectiveClearMargin() {
            return clearMargin != null ? clearMargin : Math.max(Math.abs(threshold) * 0.02f, 0.5f);
        }

        public String describe() {
            String op = aboveTriggers ? "acima de" : "abaixo de";
            return String.format(Locale.US, "%s %s %.1f%s", name, op, threshold, unit.isEmpty() ? "" : " " + unit);
        }
    }

    private final SharedPreferences prefs;
    private final List<CustomAlertRule> customRules;

    private boolean voltageActive = false;
    private boolean tempActive = false;

    public AlertManager(Context context) {
        prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        customRules = loadCustomRules();
    }

    public float getVoltageMin() {
        return prefs.getFloat(KEY_VOLTAGE_MIN, DEFAULT_VOLTAGE_MIN);
    }

    public float getVoltageHysteresis() {
        return prefs.getFloat(KEY_VOLTAGE_HYSTERESIS, DEFAULT_VOLTAGE_HYSTERESIS);
    }

    public float getTempMax() {
        return prefs.getFloat(KEY_TEMP_MAX, DEFAULT_TEMP_MAX);
    }

    public float getTempHysteresis() {
        return prefs.getFloat(KEY_TEMP_HYSTERESIS, DEFAULT_TEMP_HYSTERESIS);
    }

    public void saveSettings(float voltageMin, float voltageHysteresis, float tempMax, float tempHysteresis) {
        prefs.edit()
                .putFloat(KEY_VOLTAGE_MIN, voltageMin)
                .putFloat(KEY_VOLTAGE_HYSTERESIS, voltageHysteresis)
                .putFloat(KEY_TEMP_MAX, tempMax)
                .putFloat(KEY_TEMP_HYSTERESIS, tempHysteresis)
                .apply();
    }

    /** Cópia da lista de alertas personalizados configurados (para a tela de Configurações). */
    public List<CustomAlertRule> getCustomRules() {
        return new ArrayList<>(customRules);
    }

    public void addCustomRule(CustomAlertRule rule) {
        customRules.add(rule);
        persistCustomRules();
    }

    public void removeCustomRule(int index) {
        if (index >= 0 && index < customRules.size()) {
            customRules.remove(index);
            persistCustomRules();
        }
    }

    private List<CustomAlertRule> loadCustomRules() {
        List<CustomAlertRule> list = new ArrayList<>();
        try {
            JSONArray arr = new JSONArray(prefs.getString(KEY_CUSTOM_RULES, "[]"));
            for (int i = 0; i < arr.length(); i++) {
                JSONObject o = arr.getJSONObject(i);
                Float clearMargin = o.has("clearMargin") ? (float) o.getDouble("clearMargin") : null;
                list.add(new CustomAlertRule(
                        o.getString("pid"), o.getString("name"), o.getString("unit"),
                        (float) o.getDouble("threshold"), o.getBoolean("above"), clearMargin));
            }
        } catch (JSONException ignored) {
            // Preferências corrompidas/antigas: começa vazio em vez de travar.
        }
        return list;
    }

    private void persistCustomRules() {
        JSONArray arr = new JSONArray();
        try {
            for (CustomAlertRule r : customRules) {
                JSONObject o = new JSONObject();
                o.put("pid", r.pid);
                o.put("name", r.name);
                o.put("unit", r.unit);
                o.put("threshold", r.threshold);
                o.put("above", r.aboveTriggers);
                if (r.clearMargin != null) o.put("clearMargin", r.clearMargin);
                arr.put(o);
            }
        } catch (JSONException ignored) {
        }
        prefs.edit().putString(KEY_CUSTOM_RULES, arr.toString()).apply();
    }

    /**
     * Avalia os valores atuais contra os limiares e devolve as mensagens de
     * alerta ativas agora (lista vazia = nada a mostrar). customValues mapeia
     * PID → valor decodificado lido NESTE ciclo; um PID ausente do mapa
     * significa "não leu agora" e mantém o estado anterior sem atualizar.
     */
    public List<String> evaluate(Float voltage, Float coolantTemp, Map<String, Float> customValues) {
        List<String> alerts = new ArrayList<>();

        if (voltage != null) {
            float min = getVoltageMin();
            if (voltage < min) voltageActive = true;
            else if (voltage > min + getVoltageHysteresis()) voltageActive = false;
            if (voltageActive) {
                alerts.add(String.format(Locale.US, "TENSÃO BAIXA: %.1fV", voltage));
            }
        }

        if (coolantTemp != null) {
            float max = getTempMax();
            if (coolantTemp > max) tempActive = true;
            else if (coolantTemp < max - getTempHysteresis()) tempActive = false;
            if (tempActive) {
                alerts.add(String.format(Locale.US, "TEMPERATURA ALTA: %.0f°C", coolantTemp));
            }
        }

        if (customValues != null) {
            for (CustomAlertRule rule : customRules) {
                Float value = customValues.get(rule.pid);
                if (value == null) continue; // não leu neste ciclo — mantém estado anterior

                float margin = rule.effectiveClearMargin();
                boolean trigger = rule.aboveTriggers ? value > rule.threshold : value < rule.threshold;
                boolean clear = rule.aboveTriggers ? value < rule.threshold - margin : value > rule.threshold + margin;
                if (trigger) rule.active = true;
                else if (clear) rule.active = false;

                if (rule.active) {
                    alerts.add(String.format(Locale.US, "%s: %.1f%s",
                            rule.name.toUpperCase(Locale.US), value, rule.unit.isEmpty() ? "" : " " + rule.unit));
                }
            }
        }

        return alerts;
    }
}
