// Gráfico de lambda em tempo real — porte pixel a pixel do
// LambdaChartView.java (Canvas do Android → Canvas 2D do navegador, a API é
// deliberadamente parecida). Mesma escala Y fixa (0,7–1,3), mesmas cores por
// faixa, mesmas linhas de referência, mesmo limite de 100 pontos.

const MAX_POINTS = 100;
const Y_MIN = 0.7;
const Y_MAX = 1.3;

const COLOR_STOICH = '#81C784';
const COLOR_RICH = '#64B5F6';
const COLOR_LEAN = '#FFD54F';
const COLOR_EXTREME = '#EF5350';
const COLOR_STOICH_LINE = '#4CAF50';
const COLOR_RICH_LINE = '#2196F3';
const COLOR_LEAN_LINE = '#FFC107';
const COLOR_EXTREME_LINE = '#F44336';
const COLOR_BG = '#16213E';

const MARGIN_LEFT = 60;
const MARGIN_RIGHT = 10;
const MARGIN_TOP = 8;
const MARGIN_BOTTOM = 8;

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function withAlpha(hex, alpha01) {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha01})`;
}

// Mesmos limiares do Android: <0.88 ou >1.15 = extremo; <0.98 = rico;
// >1.02 = pobre; senão estequiométrico.
function lineColorFor(lambda) {
  if (lambda == null) return COLOR_STOICH_LINE;
  if (lambda < 0.88 || lambda > 1.15) return COLOR_EXTREME_LINE;
  if (lambda < 0.98) return COLOR_RICH_LINE;
  if (lambda > 1.02) return COLOR_LEAN_LINE;
  return COLOR_STOICH_LINE;
}

function textColorFor(lambda) {
  if (lambda == null) return COLOR_STOICH;
  if (lambda < 0.88 || lambda > 1.15) return COLOR_EXTREME;
  if (lambda < 0.98) return COLOR_RICH;
  if (lambda > 1.02) return COLOR_LEAN;
  return COLOR_STOICH;
}

function lastNonNull(arr) {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (arr[i] != null) return arr[i];
  }
  return null;
}

export class LambdaChartView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.seriesLambda1 = [];
    this.seriesLambda2 = [];

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

  /** Recebe lambda (pra plotar) — corrente fica disponível pra uso futuro
   * (anotação), mesma assinatura do addData(lambda1, lambda2, current1,
   * current2) do Android, mas hoje só lambda é desenhado. */
  addData(lambda1, lambda2) {
    this.seriesLambda1.push(lambda1);
    this.seriesLambda2.push(lambda2);
    while (this.seriesLambda1.length > MAX_POINTS) this.seriesLambda1.shift();
    while (this.seriesLambda2.length > MAX_POINTS) this.seriesLambda2.shift();
    this._draw();
  }

  clearData() {
    this.seriesLambda1 = [];
    this.seriesLambda2 = [];
    this._draw();
  }

  _draw() {
    const ctx = this.ctx;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const chartLeft = MARGIN_LEFT;
    const chartRight = w - MARGIN_RIGHT;
    const chartTop = MARGIN_TOP;
    const chartBottom = h - MARGIN_BOTTOM;
    const chartW = chartRight - chartLeft;
    const chartH = chartBottom - chartTop;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = COLOR_BG;
    ctx.fillRect(0, 0, w, h);

    // Grid: 0,7 a 1,3, passo 0,1
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 0.5;
    ctx.font = '13px sans-serif';
    ctx.fillStyle = '#999999';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 6; i++) {
      const val = Y_MIN + i * 0.1;
      const y = chartTop + chartH * (1 - (val - Y_MIN) / (Y_MAX - Y_MIN));
      ctx.beginPath();
      ctx.moveTo(chartLeft, y);
      ctx.lineTo(chartRight, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(1), chartLeft - 6, y + 4);
    }

    // Referência λ=0.9 (laranja tracejada)
    const yRef09 = chartTop + chartH * (1 - (0.9 - Y_MIN) / (Y_MAX - Y_MIN));
    ctx.save();
    ctx.setLineDash([10, 6]);
    ctx.strokeStyle = '#FF9800';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(chartLeft, yRef09);
    ctx.lineTo(chartRight, yRef09);
    ctx.stroke();
    ctx.restore();
    ctx.textAlign = 'left';
    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#FF9800';
    ctx.fillText('0.9', chartRight - 30, yRef09 - 4);

    // Referência λ=1.10 (azul tracejada)
    const yRef110 = chartTop + chartH * (1 - (1.1 - Y_MIN) / (Y_MAX - Y_MIN));
    ctx.save();
    ctx.setLineDash([10, 6]);
    ctx.strokeStyle = '#2196F3';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(chartLeft, yRef110);
    ctx.lineTo(chartRight, yRef110);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = '#2196F3';
    ctx.fillText('1.10', chartRight - 34, yRef110 - 4);

    const lastL1 = lastNonNull(this.seriesLambda1);
    const lastL2 = lastNonNull(this.seriesLambda2);
    const color1Line = lineColorFor(lastL1);
    const color1Text = textColorFor(lastL1);
    const color2Line = lineColorFor(lastL2);
    const color2Text = textColorFor(lastL2);

    if (this.seriesLambda1.length) {
      this._drawSeriesWithFill(this.seriesLambda1, color1Line, chartLeft, chartTop, chartW, chartH, chartBottom);
    }
    if (this.seriesLambda2.length) {
      this._drawSeriesWithFill(this.seriesLambda2, color2Line, chartLeft, chartTop, chartW, chartH, chartBottom);
    }

    // Anotações centralizadas
    const centerX = chartLeft + chartW / 2;
    let annotY = chartTop + 36;
    ctx.font = 'bold 28px sans-serif';
    ctx.textAlign = 'center';

    if (lastL1 != null) {
      this._drawAnnotationCentered(`B1: ${lastL1.toFixed(3)}`, centerX, annotY, color1Text);
      annotY += 40;
    }
    if (lastL2 != null) {
      this._drawAnnotationCentered(`B2: ${lastL2.toFixed(3)}`, centerX, annotY, color2Text);
    }
  }

  _drawSeriesWithFill(series, lineColor, chartLeft, chartTop, chartW, chartH, chartBottom) {
    const ctx = this.ctx;
    const count = series.length;
    let started = false;
    let firstX = 0;
    let lastX = 0;
    const path = new Path2D();

    for (let i = 0; i < count; i++) {
      const val = series[i];
      if (val == null) continue;
      const x = chartLeft + (chartW * i) / Math.max(1, MAX_POINTS - 1);
      const clamped = Math.max(Y_MIN, Math.min(Y_MAX, val));
      const y = chartTop + chartH * (1 - (clamped - Y_MIN) / (Y_MAX - Y_MIN));
      if (!started) {
        path.moveTo(x, y);
        firstX = x;
        started = true;
      } else {
        path.lineTo(x, y);
      }
      lastX = x;
    }

    if (!started) return;

    const fillPath = new Path2D(path);
    fillPath.lineTo(lastX, chartBottom);
    fillPath.lineTo(firstX, chartBottom);
    fillPath.closePath();
    ctx.fillStyle = withAlpha(lineColor, 30 / 255);
    ctx.fill(fillPath);

    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke(path);
  }

  _drawAnnotationCentered(text, centerX, y, color) {
    const ctx = this.ctx;
    const metrics = ctx.measureText(text);
    const pad = 10;
    const x = centerX - metrics.width / 2;
    ctx.fillStyle = 'rgba(0,0,0,0.8)';
    ctx.beginPath();
    ctx.roundRect(x - pad, y - 30, metrics.width + pad * 2, 40, 6);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.fillText(text, centerX, y);
  }
}
