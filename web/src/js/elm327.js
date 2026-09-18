// Comunicação com o ELM327 via WebSerial. Sequência de inicialização, PIDs e
// conversões portados do Elm327Manager.java (app Android) — mesmos valores,
// pra manter os dois lados calibrados igual se um dia precisar comparar.
import { SerialSession } from './serial.js';

const BAUD_RATE = 38400;
const READ_TIMEOUT_MS = 400; // igual ao Android — respostas rápidas, ATST0A já limita o ELM327 a ~40ms por tentativa

// CH340, FTDI, CP2102, Prolific — mesma lista do device_filter.xml do app
// Android (clones de ELM327 USB mais comuns).
export const ELM327_USB_FILTERS = [
  { usbVendorId: 0x1a86, usbProductId: 0x7523 }, // CH340
  { usbVendorId: 0x0403, usbProductId: 0x6001 }, // FTDI
  { usbVendorId: 0x10c4, usbProductId: 0xea60 }, // CP2102 (Silicon Labs)
  { usbVendorId: 0x067b, usbProductId: 0x2303 }, // Prolific PL2303
];

export class Elm327Manager {
  constructor() {
    this.session = new SerialSession();
    this.connected = false;
  }

  async connect() {
    await this.session.open(BAUD_RATE, ELM327_USB_FILTERS);
    this.connected = true;
    try {
      await this._init();
    } catch (e) {
      this.connected = false;
      await this.session.close();
      throw e;
    }
  }

  async _init() {
    await this.session.write('ATZ\r'); // reset
    await this._sleep(1500);
    this.session.clearBytes();

    await this._cmd('ATE0');   // echo off
    await this._cmd('ATL0');   // linefeeds off
    await this._cmd('ATS0');   // spaces off (respostas mais compactas)
    await this._cmd('ATH0');   // headers off
    await this._cmd('ATAT2');  // adaptive timing agressivo
    await this._cmd('ATST0A'); // timeout curto (10 * 4ms)
    await this._cmd('ATSP0');  // protocolo automático
    await this._cmd('0100');   // warm-up — primeira consulta real costuma demorar mais

    console.log('[ELM327] inicializado (modo rápido)');
  }

  async _cmd(cmd) {
    await this.session.write(cmd + '\r');
    return this.session.readUntil('>', READ_TIMEOUT_MS);
  }

  /** Envia um PID e devolve a resposta limpa (maiúscula, sem \r\n espaço >),
   * ou null se vier erro/sem dado — mesmo contrato do queryPid() original. */
  async queryPid(pid) {
    try {
      const raw = await this._cmd(pid);
      const cleaned = raw.replace(/[\r\n>\s]/g, '').toUpperCase();
      if (!cleaned) return null;
      if (cleaned.includes('NODATA') || cleaned.includes('ERROR') ||
          cleaned.includes('UNABLE') || cleaned.includes('?')) {
        return null;
      }
      return cleaned;
    } catch (e) {
      console.warn(`[ELM327] query ${pid} falhou:`, e);
      return null;
    }
  }

  /** Lê só os dois PIDs de lambda wideband (0134 banco 1, 0138 banco 2) —
   * nenhum outro PID nessa chamada, pra máxima taxa: qualquer PID a mais
   * nessa porta serial única derruba a frequência de todos (mesmo motivo do
   * Android). */
  async readLambdaData() {
    const data = {
      o2s1Current: null, o2s1Lambda: null,
      o2s5Current: null, o2s5Lambda: null,
      timestamp: Date.now(),
    };
    if (!this.connected) return data;

    const r1 = await this.queryPid('0134');
    const p1 = this._parseWidebandO2(r1, '4134');
    if (p1) { data.o2s1Current = p1.current; data.o2s1Lambda = p1.lambda; }

    const r5 = await this.queryPid('0138');
    const p5 = this._parseWidebandO2(r5, '4138');
    if (p5) { data.o2s5Current = p5.current; data.o2s5Lambda = p5.lambda; }

    return data;
  }

  // Formato da resposta (sem espaços, headers já desligados): 4134AABBCCDD.
  // Bytes A/B (razão de equivalência comandada) não são usados aqui, mesmo
  // comportamento do Android original — só C/D (corrente) importam.
  _parseWidebandO2(resp, prefix) {
    if (!resp || resp.length < 12) return null;
    const hex = resp.replace(prefix, '').trim();
    if (hex.length < 8) return null;
    const c = parseInt(hex.substring(4, 6), 16);
    const d = parseInt(hex.substring(6, 8), 16);
    const current = ((256 * c + d) / 256) - 128;
    // Conversão mA→lambda igual ao Android e ao script Python original.
    const lambda = current <= 0 ? 1.0 + current * 0.25 : 1.0 + current * 0.5;
    return { current, lambda };
  }

  /** Lê RPM/água/ar-admissão/velocidade/bateria — chamado só quando a tela
   * de dashboard está ativa (fora dela, o loop de poll nem chama isso, pra
   * não disputar banda com os PIDs de lambda). */
  async readDashboardData() {
    const data = {
      rpm: null, coolantTemp: null, intakeAirTemp: null, speed: null,
      batteryVoltage: null, timestamp: Date.now(),
    };
    if (!this.connected) return data;

    const rpmResp = await this.queryPid('010C');
    if (rpmResp) {
      const hex = rpmResp.replace(/^.*410C/, '');
      if (hex.length >= 4) {
        const a = parseInt(hex.substring(0, 2), 16);
        const b = parseInt(hex.substring(2, 4), 16);
        data.rpm = Math.floor((256 * a + b) / 4);
      }
    }

    data.coolantTemp = await this.readCoolantTemp();

    const iatResp = await this.queryPid('010F');
    if (iatResp) {
      const hex = iatResp.replace(/^.*410F/, '');
      if (hex.length >= 2) data.intakeAirTemp = parseInt(hex.substring(0, 2), 16) - 40;
    }

    const speedResp = await this.queryPid('010D');
    if (speedResp) {
      const hex = speedResp.replace(/^.*410D/, '');
      if (hex.length >= 2) data.speed = parseInt(hex.substring(0, 2), 16);
    }

    data.batteryVoltage = await this.readBatteryVoltage();

    return data;
  }

  /** Só a temperatura do líquido de arrefecimento (PID 0105, 1 byte) —
   * consulta única e rápida, chamada em baixa frequência na tela de lambda
   * (junto da voltagem) pra manter os alertas vivos independente da tela. */
  async readCoolantTemp() {
    if (!this.connected) return null;
    const resp = await this.queryPid('0105');
    if (!resp) return null;
    const hex = resp.replace(/^.*4105/, '');
    if (hex.length < 2) return null;
    return parseInt(hex.substring(0, 2), 16) - 40;
  }

  /** Comando local do adaptador (não consulta a ECU) — leve, pode ser
   * chamado em baixa frequência sem prejudicar a taxa de lambda. */
  async readBatteryVoltage() {
    if (!this.connected) return null;
    try {
      const raw = await this._cmd('ATRV');
      const cleaned = raw.replace(/[\r\n>\s]/g, '').toUpperCase().replace('V', '');
      if (!cleaned) return null;
      const v = parseFloat(cleaned);
      return Number.isNaN(v) ? null : v;
    } catch (e) {
      console.warn('[ELM327] ATRV falhou:', e);
      return null;
    }
  }

  async disconnect() {
    this.connected = false;
    await this.session.close();
  }

  isConnected() {
    return this.connected && this.session.isOpen();
  }

  _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
