// Alertas de tensão baixa / temperatura alta — porte do AlertManager.java.
// Histerese: uma vez disparado, só limpa quando o valor volta a cruzar o
// limiar MAIS a margem (evita o alerta piscando quando o valor fica
// oscilando bem em cima do limiar).
//
// Persistido no localStorage do navegador — mesmo papel do SharedPreferences
// do Android (config de aparelho, não estado compartilhado), editável na
// tela de Configurações (botão ☰ na barra inferior, igual ao APK).
//
// Fora de escopo neste porte: as regras de alerta customizadas por PID do
// Android (CustomAlertRule, com tela própria de cadastro) — só os dois
// alertas fixos que realmente aparecem no dashboard do carro hoje.

export const DEFAULT_VOLTAGE_MIN = 13.0;
export const DEFAULT_TEMP_MAX = 105.0;
export const DEFAULT_VOLTAGE_HYSTERESIS = 0.3;
export const DEFAULT_TEMP_HYSTERESIS = 3;

const STORAGE_KEY = 'obd2_alert_settings';

export class AlertManager {
  constructor() {
    const saved = this._load();
    this.voltageMin = saved.voltageMin ?? DEFAULT_VOLTAGE_MIN;
    this.voltageHysteresis = saved.voltageHysteresis ?? DEFAULT_VOLTAGE_HYSTERESIS;
    this.tempMax = saved.tempMax ?? DEFAULT_TEMP_MAX;
    this.tempHysteresis = saved.tempHysteresis ?? DEFAULT_TEMP_HYSTERESIS;

    this._voltageActive = false;
    this._tempActive = false;
  }

  _load() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
    } catch {
      return {};
    }
  }

  saveSettings(voltageMin, voltageHysteresis, tempMax, tempHysteresis) {
    this.voltageMin = voltageMin;
    this.voltageHysteresis = voltageHysteresis;
    this.tempMax = tempMax;
    this.tempHysteresis = tempHysteresis;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ voltageMin, voltageHysteresis, tempMax, tempHysteresis }));
    } catch (e) {
      console.warn('[ALERTS] falha ao salvar configurações:', e);
    }
  }

  /** Avalia os valores atuais e devolve a lista de mensagens de alerta
   * ativas agora — mesma lógica do evaluate() original: dispara ao cruzar o
   * limiar, só desarma ao cruzar limiar+margem (histerese), estado mantido
   * entre chamadas mesmo quando o valor da vez é null (não leu neste
   * ciclo). */
  evaluate(voltage, coolantTemp) {
    const alerts = [];

    if (voltage != null) {
      if (voltage < this.voltageMin) this._voltageActive = true;
      else if (voltage > this.voltageMin + this.voltageHysteresis) this._voltageActive = false;
      if (this._voltageActive) {
        alerts.push(`TENSÃO BAIXA: ${voltage.toFixed(1)}V`);
      }
    }

    if (coolantTemp != null) {
      if (coolantTemp > this.tempMax) this._tempActive = true;
      else if (coolantTemp < this.tempMax - this.tempHysteresis) this._tempActive = false;
      if (this._tempActive) {
        alerts.push(`TEMPERATURA ALTA: ${coolantTemp.toFixed(0)}°C`);
      }
    }

    return alerts;
  }
}
