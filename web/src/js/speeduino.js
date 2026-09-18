// Comunicação com a Speeduino via WebSerial, usando o mesmo protocolo
// binário com CRC que o TunerStudio usa — porte do SpeeduinoManager.java,
// mesmos offsets/escalas (todos conferidos contra o .ini do firmware
// modificado deste carro, ver CLAUDE.md do repo).
//
// Envelope de cada requisição/resposta:
//   [tamanho do payload, 2 bytes big-endian]
//   [payload]
//   [CRC32 do payload, 4 bytes big-endian — CRC32 padrão, mesmo algoritmo
//    do java.util.zip.CRC32 que o Android usa]
//
// Somente leitura (mais a leitura de página pro stoich) — nenhum comando de
// escrita/tabela é usado aqui, igual ao Android.
import { SerialSession } from './serial.js';

const BAUD_RATE = 115200;
const WRITE_TIMEOUT_MS = 2000; // não usado diretamente (write() do WebSerial não tem timeout próprio), mantido só de referência
const READ_TIMEOUT_MS = 500;

const OCH_TABLE_ID = 0x30; // SEND_OUTPUT_CHANNELS
const OCH_BLOCK_SIZE = 130; // logger.h: LOG_ENTRY_SIZE
const TS_CAN_ID = 0; // sem CAN, conexão serial direta — mesmo caso do Android

const SERIAL_RC_OK = 0x00;
const EXPECTED_SIGNATURE = 'speeduino 202501';

// tamanho[2] + SERIAL_RC_OK(1) + bloco(130) + CRC[4], com folga.
const MAX_RESPONSE_SIZE = 2 + 1 + OCH_BLOCK_SIZE + 4 + 8;

// ---- CRC32 padrão (poly 0xEDB88320, forma refletida de 0x04C11DB7) ----
const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC32_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export class SpeeduinoManager {
  constructor() {
    this.session = new SerialSession();
    this.connected = false;
    // Relação estequiométrica configurada na tune — não é telemetria ao
    // vivo, é config; lida uma vez por conexão (readStoich) e reaproveitada
    // em todo readOutputChannels() daí em diante.
    this.stoich = null;
  }

  async connect() {
    await this.session.open(BAUD_RATE, /* sem filtro USB específico — mesmo VID/PID de qualquer Arduino Mega2560 */ undefined);
    this.connected = true;
    // Mega2560 reseta ao abrir a porta (DTR, mesmo comportamento do
    // TunerStudio) — espera o firmware terminar de subir.
    await this._sleep(2000);
  }

  /** Manda 'Q' e confere a assinatura "speeduino 202501" — chamar antes de
   * confiar em readOutputChannels(), pra garantir que o dispositivo
   * escolhido realmente responde como uma Speeduino. */
  async verifySignature() {
    if (!this.isConnected()) return false;
    const response = await this._sendCommand(new Uint8Array([0x51])); // 'Q'
    if (!response || response.length < 1 + EXPECTED_SIGNATURE.length) return false;
    if (response[0] !== SERIAL_RC_OK) return false;
    const sig = new TextDecoder('ascii').decode(response.slice(1, 1 + EXPECTED_SIGNATURE.length));
    return sig === EXPECTED_SIGNATURE;
  }

  /** Lê um byte de uma página de configuração da tune (comando 'p' — mesmo
   * envelope do 'r', só trocando a tabela de output channels por número de
   * página). Usado só pra valores fixos que não saem no bloco ao vivo. */
  async _readPageByte(page, offset) {
    if (!this.isConnected()) return null;
    const payload = new Uint8Array([
      0x70, // 'p'
      TS_CAN_ID,
      page,
      offset & 0xff, (offset >> 8) & 0xff,
      1, 0, // length = 1 byte
    ]);
    const response = await this._sendCommand(payload);
    if (!response || response.length < 2 || response[0] !== SERIAL_RC_OK) return null;
    return response[1];
  }

  /** Página 1, offset 50 — conferido contra mainController.ini deste carro:
   * "stoich = scalar, U08, 50, ':1', 0.1". Chamar uma vez por conexão
   * (depois de verifySignature() confirmar que é mesmo a Speeduino). */
  async readStoich() {
    const raw = await this._readPageByte(1, 50);
    this.stoich = raw != null ? raw * 0.1 : null;
  }

  /** Lê e decodifica o bloco de dados ao vivo inteiro (130 bytes). */
  async readOutputChannels() {
    const data = { timestamp: Date.now() };
    if (!this.isConnected()) return data;

    const payload = new Uint8Array([
      0x72, // 'r'
      TS_CAN_ID,
      OCH_TABLE_ID,
      0, 0, // offset = 0, little-endian
      OCH_BLOCK_SIZE & 0xff, (OCH_BLOCK_SIZE >> 8) & 0xff, // length, little-endian
    ]);
    const response = await this._sendCommand(payload);
    if (!response || response.length < 1 + OCH_BLOCK_SIZE || response[0] !== SERIAL_RC_OK) {
      return data;
    }
    this._parseBlock(response, data);
    return data;
  }

  // offset é relativo ao bloco de status (0-based); +1 pula o SERIAL_RC_OK.
  _u8(block, offset) {
    return block[offset + 1];
  }

  _s8(block, offset) {
    const v = block[offset + 1];
    return v > 127 ? v - 256 : v;
  }

  _u16le(block, offset) {
    return this._u8(block, offset) | (this._u8(block, offset + 1) << 8);
  }

  // short com sinal — necessário pra rpmDot/TPSdot/MAPdot, que ficam
  // negativos exatamente nos casos que importam (rotação caindo, vácuo
  // subindo, solta o acelerador de repente).
  _s16le(block, offset) {
    const v = this._u16le(block, offset);
    return v > 32767 ? v - 65536 : v;
  }

  /** Decodifica os campos usados a partir do bloco bruto — mesmos
   * offsets/escalas do SpeeduinoManager.java, conferidos contra
   * logger.cpp:getTSLogEntry e o .ini deste carro. */
  _parseBlock(block, data) {
    data.secl = this._u8(block, 0);
    // offsets 1, 2 e 16 — DFCOOn (bits, U08, 1, [4:4]), bloco de flags
    // "engine" (U08 offset 2), accelEnrich (scalar, U08, 16, escala 2.0).
    data.dfco = (this._u8(block, 1) >> 4) & 0x01;
    data.engineStatus = this._u8(block, 2);
    data.accelEnrichPct = this._u8(block, 16) * 2.0;
    data.mapKpa = this._u16le(block, 4);
    data.iatC = this._u8(block, 6) - 40;
    data.coolantC = this._u8(block, 7) - 40;
    data.batteryV = this._u8(block, 9) * 0.1;
    data.afrNative = this._u8(block, 10) * 0.1;
    data.rpm = this._u16le(block, 14);
    // gammaEnrich = scalar, U16, 17.
    data.gammaE = this._u16le(block, 17);
    data.ve1Pct = this._u8(block, 19);
    data.ve2Pct = this._u8(block, 20);
    data.afrTarget = this._u8(block, 21) * 0.1;
    data.lambdaTarget = this.stoich != null ? data.afrTarget / this.stoich : null;
    // TPSdot = scalar, S16, 22.
    data.tpsDot = this._s16le(block, 22);
    data.advanceDeg = this._s8(block, 24);
    data.tpsPct = this._u8(block, 25) * 0.5;
    // rpmDOT = scalar, S16, 33.
    data.rpmDot = this._s16le(block, 33);
    data.ethanolPct = this._u8(block, 35);
    data.baroKpa = this._u8(block, 41);
    data.pw1Ms = this._u16le(block, 76) * 0.001;
    data.pw2Ms = this._u16le(block, 78) * 0.001;
    data.dwellMs = this._u16le(block, 90) * 0.001;
    // MAPdot = scalar, S16, 93.
    data.mapDot = this._s16le(block, 93);
    data.advance1Deg = this._s8(block, 118);
    data.advance2Deg = this._s8(block, 119);
    // baroCorrection = scalar, U08, 101.
    data.baroCorrectionPct = this._u8(block, 101);
    // veCurr = scalar, U08, 102 — VE realmente usada no PW, não a leitura
    // crua da tabela (ve1Pct/ve2Pct).
    data.veCurr = this._u8(block, 102);
  }

  /** Monta o envelope, envia, acumula a resposta até saber (pelos 2
   * primeiros bytes) quanto falta, e confere o CRC32. Retorna null em
   * timeout ou CRC inválido — mesmo contrato do sendCommand() original.
   *
   * Descarta qualquer byte parado no buffer antes de mandar (equivalente ao
   * drainBuffer do Android) — sem marcador de resync no protocolo, sobra de
   * uma troca anterior incompleta seria lida como início da próxima
   * resposta, corrompendo tudo dali em diante. */
  async _sendCommand(payload) {
    this.session.clearBytes();
    await this.session.write(this._wrapRequest(payload));

    const deadline = Date.now() + READ_TIMEOUT_MS;

    await this.session.waitForBytes(2, READ_TIMEOUT_MS);
    if (this.session.byteLength() < 2) return null; // timeout antes de saber o tamanho

    const header = this.session.peekBytes(2);
    const responseLength = (header[0] << 8) | header[1];
    const totalLength = 2 + responseLength + 4;
    if (totalLength > MAX_RESPONSE_SIZE) {
      console.warn(`[SPEEDUINO] resposta maior que o esperado (${totalLength} bytes) — ignorando`);
      this.session.clearBytes();
      return null;
    }

    const remaining = deadline - Date.now();
    if (remaining > 0) await this.session.waitForBytes(totalLength, remaining);
    if (this.session.byteLength() < totalLength) return null; // timeout

    const all = this.session.takeBytes(totalLength);
    const responsePayload = all.slice(2, 2 + responseLength);
    const expectedCrc = (all[2 + responseLength] << 24) | (all[2 + responseLength + 1] << 16) |
      (all[2 + responseLength + 2] << 8) | all[2 + responseLength + 3];
    const actualCrc = crc32(responsePayload);
    if ((actualCrc >>> 0) !== (expectedCrc >>> 0)) {
      console.warn('[SPEEDUINO] CRC da resposta não confere — ignorando pacote');
      return null;
    }
    return responsePayload;
  }

  _wrapRequest(payload) {
    const crcValue = crc32(payload);
    const out = new Uint8Array(2 + payload.length + 4);
    out[0] = (payload.length >> 8) & 0xff;
    out[1] = payload.length & 0xff;
    out.set(payload, 2);
    const base = 2 + payload.length;
    out[base] = (crcValue >> 24) & 0xff;
    out[base + 1] = (crcValue >> 16) & 0xff;
    out[base + 2] = (crcValue >> 8) & 0xff;
    out[base + 3] = crcValue & 0xff;
    return out;
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
