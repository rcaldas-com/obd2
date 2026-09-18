// Dashboard em gauges — porte do DashboardView.java (Canvas Android → Canvas
// 2D do navegador). RPM/água/ar/velocidade/bateria vêm do ELM327; ponto/MAP/
// baro/flex/TPS vêm da Speeduino (quem realmente comanda a ignição agora e
// cujo TPS é o que os mapas dela usam) — duas fontes independentes,
// atualizadas por updateData()/updateSpeeduinoData() separados, exatamente
// como no Android.

const COLOR_BG = '#16213E';
const COLOR_GAUGE_BG = '#1E2A4A';
const COLOR_ARC_BG = '#2A3A5C';
const COLOR_GREEN = '#4CAF50';
const COLOR_YELLOW = '#FFC107';
const COLOR_RED = '#F44336';
const COLOR_BLUE = '#2196F3';
const COLOR_CYAN = '#00BCD4';
const COLOR_ORANGE = '#FF9800';
const COLOR_TEXT = '#EEEEEE';
const COLOR_LABEL = 'rgba(255,255,255,0.53)';
const COLOR_UNIT = 'rgba(255,255,255,0.4)';

function rpmColor(rpm) {
  if (rpm == null) return COLOR_GREEN;
  if (rpm > 5500) return COLOR_RED;
  if (rpm > 4000) return COLOR_YELLOW;
  return COLOR_GREEN;
}
function coolantColor(t) {
  if (t == null) return COLOR_GREEN;
  if (t > 105) return COLOR_RED;
  if (t > 95) return COLOR_YELLOW;
  if (t < 60) return COLOR_BLUE;
  return COLOR_GREEN;
}
function timingColor(t) {
  if (t == null) return COLOR_ORANGE;
  if (t < 0) return COLOR_RED;
  if (t < 5) return COLOR_YELLOW;
  return COLOR_ORANGE;
}
function speedColor(s) {
  if (s == null) return COLOR_GREEN;
  if (s > 140) return COLOR_RED;
  if (s > 100) return COLOR_YELLOW;
  return COLOR_GREEN;
}
function tpsColor(t) {
  if (t == null) return COLOR_GREEN;
  if (t > 80) return COLOR_RED;
  if (t > 50) return COLOR_YELLOW;
  return COLOR_GREEN;
}

export class DashboardView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    this.rpm = null;
    this.coolantTemp = null;
    this.intakeAirTemp = null;
    this.speed = null;
    this.batteryVoltage = null;
    this.speeduinoAdvance = null;
    this.speeduinoMap = null;
    this.speeduinoBaro = null;
    this.speeduinoFlexPct = null;
    this.speeduinoTps = null;

    this._resize();
    this._resizeObserver = new ResizeObserver(() => this._resize());
    this._resizeObserver.observe(canvas);
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._draw();
  }

  updateData(data) {
    this.rpm = data.rpm ?? null;
    this.coolantTemp = data.coolantTemp ?? null;
    this.intakeAirTemp = data.intakeAirTemp ?? null;
    this.speed = data.speed ?? null;
    this.batteryVoltage = data.batteryVoltage ?? null;
    this._draw();
  }

  /** Atualizado por um loop de poll independente (Speeduino tem sua
   * própria porta USB) — pode chegar em instantes diferentes de
   * updateData(), por isso é um método separado, não parte do mesmo
   * "pacote" de dados. */
  updateSpeeduinoData(data) {
    this.speeduinoAdvance = data.advanceDeg ?? null;
    this.speeduinoMap = data.mapKpa ?? null;
    this.speeduinoBaro = data.baroKpa ?? null;
    this.speeduinoFlexPct = data.ethanolPct ?? null;
    this.speeduinoTps = data.tpsPct ?? null;
    this._draw();
  }

  clearData() {
    this.rpm = this.coolantTemp = this.intakeAirTemp = this.speed = this.batteryVoltage = null;
    this._draw();
  }

  /** Chamado quando a Speeduino desconecta — os gauges dela voltam a
   * mostrar "--", os outros (OBD2) continuam como estavam. */
  clearSpeeduinoData() {
    this.speeduinoAdvance = this.speeduinoMap = this.speeduinoBaro = null;
    this.speeduinoFlexPct = this.speeduinoTps = null;
    this._draw();
  }

  _draw() {
    const ctx = this.ctx;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    // Canvas escondido (display:none enquanto a tela de lambda está ativa)
    // tem bounding rect zerado — desenhar nesse estado joga raio negativo
    // no ctx.arc() das gauges (IndexSizeError). Nada a desenhar mesmo.
    if (w <= 0 || h <= 0) return;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = COLOR_BG;
    ctx.fillRect(0, 0, w, h);

    // Layout: 3 colunas x 3 linhas — a 3ª linha (MAP/BARO/FLEX) vem da
    // Speeduino, assim como o PONTO na 2ª linha.
    const pad = 8;
    const cellW = (w - pad * 4) / 3;
    const cellH = (h - pad * 4) / 3;
    const row0 = pad, row1 = pad * 2 + cellH, row2 = pad * 3 + cellH * 2;
    const col0 = pad, col1 = pad * 2 + cellW, col2 = pad * 3 + cellW * 2;

    // Linha 1: RPM | Água | Ar admissão
    this._gauge(col0, row0, cellW, cellH, 'RPM',
      this.rpm != null ? String(this.rpm) : '--', '',
      this.rpm != null ? this.rpm / 7000 : 0, rpmColor(this.rpm));

    this._gauge(col1, row0, cellW, cellH, 'ÁGUA',
      this.coolantTemp != null ? this.coolantTemp.toFixed(0) : '--', '°C',
      this.coolantTemp != null ? this.coolantTemp / 130 : 0, coolantColor(this.coolantTemp));

    this._gauge(col2, row0, cellW, cellH, 'AR ADMISSÃO',
      this.intakeAirTemp != null ? this.intakeAirTemp.toFixed(0) : '--', '°C',
      this.intakeAirTemp != null ? (this.intakeAirTemp + 40) / 100 : 0, COLOR_CYAN);

    // Linha 2: Velocidade | Ponto (Speeduino) | TPS
    this._gauge(col0, row1, cellW, cellH, 'VELOCIDADE',
      this.speed != null ? String(this.speed) : '--', 'km/h',
      this.speed != null ? this.speed / 200 : 0, speedColor(this.speed));

    this._gauge(col1, row1, cellW, cellH, 'PONTO',
      this.speeduinoAdvance != null ? this.speeduinoAdvance.toFixed(0) : '--', '°',
      this.speeduinoAdvance != null ? (this.speeduinoAdvance + 20) / 60 : 0, timingColor(this.speeduinoAdvance));

    this._gauge(col2, row1, cellW, cellH, 'TPS',
      this.speeduinoTps != null ? this.speeduinoTps.toFixed(1) : '--', '%',
      this.speeduinoTps != null ? this.speeduinoTps / 100 : 0, tpsColor(this.speeduinoTps));

    // Linha 3: MAP | BARO | FLEX (todos da Speeduino)
    this._gauge(col0, row2, cellW, cellH, 'MAP',
      this.speeduinoMap != null ? this.speeduinoMap.toFixed(0) : '--', 'kPa',
      this.speeduinoMap != null ? this.speeduinoMap / 105 : 0, COLOR_BLUE);

    this._gauge(col1, row2, cellW, cellH, 'BARO',
      this.speeduinoBaro != null ? this.speeduinoBaro.toFixed(0) : '--', 'kPa',
      this.speeduinoBaro != null ? this.speeduinoBaro / 105 : 0, COLOR_CYAN);

    this._gauge(col2, row2, cellW, cellH, 'FLEX',
      this.speeduinoFlexPct != null ? String(this.speeduinoFlexPct) : '--', '% etanol',
      this.speeduinoFlexPct != null ? this.speeduinoFlexPct / 100 : 0, COLOR_ORANGE);
  }

  _gauge(x, y, w, h, label, value, unit, fraction, color) {
    const ctx = this.ctx;
    const cx = x + w / 2;
    const cy = y + h * 0.48;
    const radius = Math.min(w, h) * 0.32;
    const arcWidth = radius * 0.18;
    fraction = Math.max(0, Math.min(1, fraction));

    // Fundo da célula
    ctx.fillStyle = COLOR_GAUGE_BG;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 8);
    ctx.fill();

    // Arco de fundo — 150° a 390° (240° de vão), mesmo ângulo do Android
    // (Canvas Android mede em graus a partir do eixo X positivo, sentido
    // horário; Canvas 2D web usa radianos, mesma convenção de sentido).
    const startAngle = (150 * Math.PI) / 180;
    const sweepBg = (240 * Math.PI) / 180;
    ctx.strokeStyle = COLOR_ARC_BG;
    ctx.lineWidth = arcWidth;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, startAngle + sweepBg);
    ctx.stroke();

    // Arco de valor
    const sweepValue = sweepBg * fraction;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, startAngle + sweepValue);
    ctx.stroke();

    // Valor central
    ctx.fillStyle = color;
    ctx.font = `bold ${Math.round(radius * 0.68)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(value, cx, cy + radius * 0.22);

    // Unidade
    ctx.fillStyle = COLOR_UNIT;
    ctx.font = `${Math.round(radius * 0.3)}px sans-serif`;
    ctx.fillText(unit, cx, cy + radius * 0.6);

    // Label
    ctx.fillStyle = COLOR_LABEL;
    ctx.font = `${Math.min(radius * 0.3, 14)}px sans-serif`;
    ctx.fillText(label, cx, y + h - 8);
  }
}
