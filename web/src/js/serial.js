// Transporte serial puro (abrir/escrever/ler/fechar), sem conhecimento de
// protocolo — mesma divisão de responsabilidade do UsbSerialSession.java no
// app Android. Cada Elm327Manager/SpeeduinoManager tem sua própria
// instância, portas independentes.
//
// Diferença de propósito em relação ao original: o Android expõe
// port.read(buf, timeoutMs), uma leitura bloqueante com timeout que dá pra
// chamar repetidamente. O WebSerial só dá reader.read() (uma promise que
// resolve quando ALGO chegar, sem timeout embutido). Portar isso ingenuamente
// chamando reader.read() de novo a cada timeout criaria uma corrida: cada
// chamada abandonada continua pendente e pode "roubar" o próximo pedaço de
// dado que chegar, entregando bytes pra quem já desistiu de esperar por eles
// em vez de para a leitura lógica atual. Por isso aqui só existe UM loop de
// leitura (_pump), rodando em segundo plano a vida toda da conexão, jogando
// tudo que chega num buffer de BYTES bruto; quem espera resposta (texto do
// ELM327 ou binário da Speeduino) só espera esse buffer, nunca disputa o
// reader diretamente.
export class SerialSession {
  constructor() {
    this.port = null;
    this.reader = null;
    this.writer = null;
    this._bytes = new Uint8Array(0);
    this._pumping = false;
    this._waiters = [];
  }

  /** Pede ao usuário pra escolher a porta (diálogo do Chrome) e abre com os
   * parâmetros informados. filters é opcional — lista de {usbVendorId,
   * usbProductId} pra restringir o que aparece no diálogo, mesmo papel do
   * device_filter.xml do app Android. */
  async open(baudRate, filters) {
    if (!('serial' in navigator)) {
      throw new Error('WebSerial não disponível — precisa de Chrome/Edge, acessando via http://localhost (não IP de rede)');
    }
    this.port = await navigator.serial.requestPort(filters ? { filters } : {});
    await this.port.open({ baudRate, dataBits: 8, stopBits: 1, parity: 'none' });
    this.writer = this.port.writable.getWriter();
    this.reader = this.port.readable.getReader();
    this._bytes = new Uint8Array(0);
    this._pumping = true;
    this._pump();
  }

  async _pump() {
    try {
      while (this._pumping) {
        const { value, done } = await this.reader.read();
        if (done) break;
        if (value && value.length) {
          const merged = new Uint8Array(this._bytes.length + value.length);
          merged.set(this._bytes, 0);
          merged.set(value, this._bytes.length);
          this._bytes = merged;
          this._wake();
        }
      }
    } catch (e) {
      // porta fechada/desconectada no meio da leitura — normal ao chamar
      // close(); waiters pendentes só estouram por timeout, sem erro aqui.
    }
  }

  _wake() {
    const waiters = this._waiters;
    this._waiters = [];
    for (const wake of waiters) wake();
  }

  /** Aceita string (protocolo texto do ELM327, auto-codificada em ASCII) ou
   * Uint8Array (protocolo binário da Speeduino). */
  async write(data) {
    if (!this.writer) throw new Error('porta não está aberta');
    const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
    await this.writer.write(bytes);
  }

  byteLength() {
    return this._bytes.length;
  }

  /** Olha os primeiros n bytes sem consumir — usado pra ler o cabeçalho de
   * tamanho do protocolo da Speeduino antes de saber quanto ainda falta. */
  peekBytes(n) {
    return this._bytes.slice(0, Math.min(n, this._bytes.length));
  }

  /** Consome e devolve os primeiros n bytes (ou o que tiver, se for menos). */
  takeBytes(n) {
    const take = Math.min(n, this._bytes.length);
    const out = this._bytes.slice(0, take);
    this._bytes = this._bytes.slice(take);
    return out;
  }

  /** Descarta qualquer byte parado no buffer — equivalente ao
   * drainBuffer()/clearBuffer() do Android, chamado antes de mandar um
   * comando novo pra não confundir sobra de uma troca anterior incompleta
   * com o início da resposta atual. */
  clearBytes() {
    this._bytes = new Uint8Array(0);
  }

  /** Espera até o buffer ter pelo menos minBytes ou o tempo esgotar. */
  async waitForBytes(minBytes, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    while (this._bytes.length < minBytes) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await new Promise((resolve) => {
        this._waiters.push(resolve);
        // acorda periodicamente mesmo sem dado novo, só pra reavaliar o
        // deadline.
        setTimeout(resolve, Math.min(remaining, 50));
      });
    }
  }

  /** Espera até o buffer (interpretado como ASCII) conter `marker` ou o
   * tempo esgotar — protocolo de texto do ELM327. Devolve tudo que tiver
   * decodificado até agora (marker incluso) e limpa o buffer. */
  async readUntil(marker, timeoutMs) {
    const decoder = new TextDecoder('ascii');
    const deadline = Date.now() + timeoutMs;
    while (!decoder.decode(this._bytes).includes(marker)) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await new Promise((resolve) => {
        this._waiters.push(resolve);
        setTimeout(resolve, Math.min(remaining, 50));
      });
    }
    const text = decoder.decode(this._bytes);
    this._bytes = new Uint8Array(0);
    return text;
  }

  async close() {
    this._pumping = false;
    try { if (this.reader) { await this.reader.cancel(); this.reader.releaseLock(); } } catch {}
    try { if (this.writer) this.writer.releaseLock(); } catch {}
    try { if (this.port) await this.port.close(); } catch {}
    this.port = null;
    this.reader = null;
    this.writer = null;
  }

  isOpen() {
    return this.port !== null;
  }
}
