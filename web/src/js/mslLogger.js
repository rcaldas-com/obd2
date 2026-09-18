// Grava um log .msl (formato TunerStudio) combinando duas fontes que
// atualizam em ritmos independentes — porte de MslLogger.java, mesmas
// colunas, mesma ordem, mesmo motivo de cada uma (ver o Javadoc da classe
// Java pro raciocínio completo: por que Lambda Target em vez de AFR, por que
// VE _Current em vez da leitura crua da tabela, por que baro entra, por que
// ponto de ignição NÃO entra aqui).
//
// RPMdot/MAPdot/TPSdot/DFCO/Engine Status/Accel Enrich ficam no fim da
// lista, depois de Ethanol — mesmo motivo do original: preserva a posição
// das colunas já existentes pra qualquer análise por nome de coluna (não por
// posição) continuar funcionando em cima de logs novos.
//
// Sem java.io aqui — o navegador não tem sistema de arquivos direto, então
// isso usa a File System Access API (showSaveFilePicker + stream gravável),
// Chromium-only, mesma exigência da WebSerial que o resto do app já tem.
const TICK_MS = 100; // 10Hz, igual ao Android

const COLUMN_NAMES = [
  'Time', 'RPM', 'MAP', 'TPS', 'CLT', 'IAT', 'Baro Pressure', 'Baro Correction',
  'VE _Current', 'GammaE', 'Lambda Target', 'Lambda', 'Lambda2', 'Ethanol',
  'RPMdot', 'MAPdot', 'TPSdot', 'DFCO', 'Engine Status', 'Accel Enrich',
];
const COLUMN_UNITS = [
  's', 'rpm', 'kpa', '%', '', '', 'kpa', '%',
  '%', '%', 'O2', 'O2', 'O2', '%',
  'rpm/s', 'kpa/s', '%/s', '', 'bits', '%',
];

export class MslLogger {
  constructor() {
    this._writable = null;
    this._recording = false;
    this._paused = false;
    this._startTime = 0;
    this._timer = null;

    // Última leitura conhecida de cada fonte — atualizada pelos loops de
    // poll do ELM327 e da Speeduino, lida só pelo tick. Sem necessidade de
    // lock (JS é single-threaded); mesmo papel do `volatile` no Android.
    this._lambda1 = null;
    this._lambda2 = null;
    this._speeduinoData = null;

    // Encadeia os writes — createWritable() não garante ordem entre
    // chamadas concorrentes de write(), e um tick pode disparar antes do
    // anterior terminar de gravar (I/O do disco pode ser mais lento que
    // 100ms num caso ruim).
    this._rowQueue = Promise.resolve();
  }

  /** Chamado pelo loop de poll do ELM327 sempre que lê lambda — enquanto
   * grava, isso deve acontecer toda volta do loop não importa a tela ativa,
   * já que é a única fonte que existe (ver app.js). */
  updateObd2Lambda(bank1, bank2) {
    this._lambda1 = bank1;
    this._lambda2 = bank2;
  }

  /** Chamado pelo loop de poll da Speeduino a cada leitura. */
  updateSpeeduino(data) {
    this._speeduinoData = data;
  }

  isRecording() {
    return this._recording;
  }

  isPaused() {
    return this._paused;
  }

  /** Pausa sem fechar o arquivo — o Time (relógio desde o start) continua
   * correndo, então uma pausa aparece no .msl como um salto no tempo entre
   * duas linhas, não como se o trecho nunca tivesse acontecido. Sem efeito
   * se não estiver gravando. */
  pause() {
    if (this._recording) this._paused = true;
  }

  resume() {
    this._paused = false;
  }

  /** Abre o diálogo "Salvar como", cria o arquivo e escreve o cabeçalho.
   * @returns nome do arquivo escolhido (pra mostrar na UI) */
  async start() {
    if (!('showSaveFilePicker' in window)) {
      throw new Error('Gravação de log precisa do File System Access API — Chrome/Edge');
    }

    const filename = `${this._formatTimestamp(new Date())}.msl`;
    const handle = await window.showSaveFilePicker({
      suggestedName: filename,
      types: [{ description: 'Log MSL (TunerStudio)', accept: { 'text/plain': ['.msl'] } }],
    });
    this._writable = await handle.createWritable();

    const header =
      'speeduino 202501: Speeduino 2025.01.7\n' +
      `Capture Date: ${new Date().toString()}, File author: lambda_web\n` +
      '#\n' +
      COLUMN_NAMES.join('\t') + '\n' +
      COLUMN_UNITS.join('\t') + '\n';
    await this._writable.write(header);

    this._startTime = Date.now();
    this._recording = true;
    this._paused = false;
    this._timer = setInterval(() => this._tick(), TICK_MS);

    return handle.name || filename;
  }

  async stop() {
    this._recording = false;
    this._paused = false;
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
    // Espera qualquer linha ainda em voo terminar de gravar antes de
    // fechar o stream — fechar com escrita pendente perderia a linha.
    await this._rowQueue;
    if (this._writable) {
      try {
        await this._writable.close();
      } catch (e) {
        console.warn('[MSL] erro ao fechar log:', e);
      }
      this._writable = null;
    }
  }

  _tick() {
    if (!this._recording || this._paused) return;
    this._writeRow();
  }

  _writeRow() {
    // Snapshot local — speeduinoData pode ser trocado pelo loop de poll
    // entre a leitura de um campo e outro; o snapshot evita misturar
    // campos de dois instantes diferentes na mesma linha.
    const sd = this._speeduinoData;
    const t = (Date.now() - this._startTime) / 1000;

    const fields = [
      t.toFixed(3),
      sd?.rpm != null ? String(sd.rpm) : '',
      sd?.mapKpa != null ? String(Math.round(sd.mapKpa)) : '',
      sd?.tpsPct != null ? sd.tpsPct.toFixed(1) : '',
      sd?.coolantC != null ? String(Math.round(sd.coolantC)) : '',
      sd?.iatC != null ? String(Math.round(sd.iatC)) : '',
      sd?.baroKpa != null ? String(Math.round(sd.baroKpa)) : '',
      sd?.baroCorrectionPct != null ? String(sd.baroCorrectionPct) : '',
      sd?.veCurr != null ? String(sd.veCurr) : '',
      sd?.gammaE != null ? String(sd.gammaE) : '',
      sd?.lambdaTarget != null ? sd.lambdaTarget.toFixed(3) : '',
      this._lambda1 != null ? this._lambda1.toFixed(3) : '',
      this._lambda2 != null ? this._lambda2.toFixed(3) : '',
      // O limite de detonação anda junto com o teor de álcool do tanque —
      // mas isso é a tabela de VE, não a de ponto (que não entra aqui);
      // etanol continua útil pra separar tanque a tanque.
      sd?.ethanolPct != null ? String(sd.ethanolPct) : '',
      sd?.rpmDot != null ? String(sd.rpmDot) : '',
      sd?.mapDot != null ? String(sd.mapDot) : '',
      sd?.tpsDot != null ? String(sd.tpsDot) : '',
      sd?.dfco != null ? String(sd.dfco) : '',
      sd?.engineStatus != null ? String(sd.engineStatus) : '',
      sd?.accelEnrichPct != null ? sd.accelEnrichPct.toFixed(0) : '',
    ];
    const line = fields.join('\t') + '\n';

    this._rowQueue = this._rowQueue
      .then(() => this._writable && this._writable.write(line))
      .catch((e) => console.warn('[MSL] erro ao escrever linha:', e));
  }

  _formatTimestamp(date) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}_` +
      `${pad(date.getHours())}.${pad(date.getMinutes())}.${pad(date.getSeconds())}`;
  }
}
