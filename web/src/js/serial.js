// Transporte serial puro (abrir/escrever/ler/fechar), sem conhecimento de
// protocolo — mesma divisão de responsabilidade do UsbSerialSession.java no
// app Android. Cada Elm327Manager (e, quando existir, SpeeduinoManager) tem
// sua própria instância, portas independentes.
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
// tudo que chega num buffer de texto; readUntil() só espera esse buffer
// atingir uma condição, nunca disputa o reader diretamente.
export class SerialSession {
  constructor() {
    this.port = null;
    this.reader = null;
    this.writer = null;
    this._buf = '';
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
    this._buf = '';
    this._pumping = true;
    this._pump();
  }

  async _pump() {
    const decoder = new TextDecoder('ascii');
    try {
      while (this._pumping) {
        const { value, done } = await this.reader.read();
        if (done) break;
        if (value && value.length) {
          this._buf += decoder.decode(value, { stream: true });
          const waiters = this._waiters;
          this._waiters = [];
          for (const wake of waiters) wake();
        }
      }
    } catch (e) {
      // porta fechada/desconectada no meio da leitura — normal ao chamar
      // close(); waiters pendentes só estouram por timeout, sem erro aqui.
    }
  }

  async write(text) {
    if (!this.writer) throw new Error('porta não está aberta');
    await this.writer.write(new TextEncoder().encode(text));
  }

  /** Espera até o buffer acumulado conter `marker` ou o tempo esgotar.
   * Devolve tudo que tiver no buffer até agora (marker incluso) e limpa —
   * equivalente ao readResponse() do Elm327Manager.java, que também lê até
   * achar '>' ou estourar o deadline. */
  async readUntil(marker, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    while (!this._buf.includes(marker)) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await new Promise((resolve) => {
        this._waiters.push(resolve);
        // acorda periodicamente mesmo sem dado novo, só pra reavaliar o
        // deadline — sem isso um _wake() que nunca vem prende a espera até
        // o timeout do Promise.race não existir (não tem aqui, é direto).
        setTimeout(resolve, Math.min(remaining, 50));
      });
    }
    const out = this._buf;
    this._buf = '';
    return out;
  }

  clearBuffer() {
    this._buf = '';
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
