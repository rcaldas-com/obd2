// Dashboard estilo painel automotivo — cada sinal desenhado de um jeito que
// combina com a natureza dele, não uma grade uniforme de gauges idênticos:
//
//   - RPM: conta-giros central, grande, com ponteiro e zonas coloridas (é o
//     instrumento "principal" de um painel de carro de verdade), mais o
//     valor em dígitos (não só o analógico). Velocidade em dígitos maiores
//     no centro dele, como um painel híbrido moderno.
//   - Água/Ar admissão: gauges em formato de cúpula, maiores e mais altas
//     nas pontas da faixa de baixo (são as duas que importam de verdade
//     pro motor). O conta-giros é dimensionado pela altura, não a
//     largura — numa tela estreita ele cresce por baixo delas de
//     propósito, sem vão vazio desperdiçado.
//   - Ponto/Baro/Flex: não são "coisa de ponteiro" no dia a dia (consulta
//     ocasional, não algo pra reagir rápido) — viram só um número grande
//     colorido, menores, no meio da mesma faixa de baixo.
//   - MAP/TPS: os dois andam quase juntos (carga do motor), lado a lado,
//     como duas barras verticais que enchem de baixo pra cima, tipo
//     nível de bateria/combustível.
//
// RPM/água/ar/velocidade vêm do ELM327; ponto/MAP/baro/flex/TPS vêm da
// Speeduino — duas fontes independentes, atualizadas por updateData()/
// updateSpeeduinoData() separados, exatamente como no Android.

const COLOR_BG = '#16213E';
const COLOR_GAUGE_BG = '#1E2A4A';
const COLOR_ARC_BG = '#2A3A5C';
const COLOR_GREEN = '#4CAF50';
const COLOR_YELLOW = '#FFC107';
const COLOR_RED = '#F44336';
const COLOR_BLUE = '#2196F3';
const COLOR_TEXT = '#EEEEEE';
const COLOR_LABEL = 'rgba(255,255,255,0.53)';
const COLOR_UNIT = 'rgba(255,255,255,0.4)';

const RPM_MAX = 7000;
const RPM_YELLOW = 4000;
const RPM_RED = 5500;

function rpmColor(rpm) {
  if (rpm == null) return COLOR_GREEN;
  if (rpm > RPM_RED) return COLOR_RED;
  if (rpm > RPM_YELLOW) return COLOR_YELLOW;
  return COLOR_GREEN;
}
function coolantColor(t) {
  if (t == null) return COLOR_GREEN;
  if (t > 105) return COLOR_RED;
  if (t > 95) return COLOR_YELLOW;
  if (t < 60) return COLOR_BLUE;
  return COLOR_GREEN;
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
// Faixa/cores dadas pelo usuário (não existe equivalente no Android — lá é
// ciano fixo): azul até 20°C, verde até 35, amarelo até 45, vermelho daí.
function intakeAirColor(t) {
  if (t == null) return COLOR_BLUE;
  if (t > 45) return COLOR_RED;
  if (t > 35) return COLOR_YELLOW;
  if (t > 20) return COLOR_GREEN;
  return COLOR_BLUE;
}

function polar(cx, cy, r, deg) {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}
function clamp01(v) {
  return Math.max(0, Math.min(1, v));
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

    // Tira à direita reservada pra MAP/TPS (as duas barras verticais, lado
    // a lado); o resto é a área "principal" (conta-giros + faixa de baixo
    // unificada, sem nada tampando a face do mostrador).
    const stripW = Math.max(w * 0.13, 60);
    const mainW = w - stripW;
    const bottomBandH = h * 0.26;

    // ---- Conta-giros central, com velocidade (e RPM em dígitos) no
    // meio — dimensionado pela ALTURA disponível, não pela largura: numa
    // tela estreita isso faz o círculo crescer além de mainW e sobrepor
    // água/ar (que ficam nas pontas, ver faixa de baixo) — de propósito,
    // pra não sobrar vão vazio embaixo do mostrador. O fundo dele quase
    // encosta no topo do FLEX (o vão do conta-giros ali embaixo já fica
    // vazio mesmo, é onde a fileira central pode entrar por baixo). ----
    const marginTop = h * 0.02;
    const tachBottomY = h - bottomBandH * 0.48;
    // O que baliza o tamanho agora são as pontas 0/7 da escala (ficam a
    // r*0.78 de raio, nos ângulos 135°/45° — bem mais perto do centro que
    // a borda do círculo) e a legenda "RPM ×1000", subida pra dentro do
    // vão de baixo do mostrador (r*0.62, não mais r*1.08) — as duas quase
    // encostam na fileira de baixo numa tela larga, de propósito.
    let rTach = (tachBottomY - marginTop) / 1.85;
    // Teto: mainW*0.5 já encosta o círculo nas duas bordas de mainW — um
    // pouco além disso (a tira à direita é repintada por cima, ver
    // abaixo) deixa ele morder as cúpulas de água/ar sem comer a tira.
    rTach = Math.min(rTach, mainW * 0.54);
    const cy = marginTop + rTach;
    const cx = mainW * 0.5;
    this._tach(cx, cy, rTach);

    // ---- Faixa de baixo única: Água | Ponto | Flex | Baro | Ar admissão —
    // água/ar nas pontas ("mais pra fora"), maiores e mais altas que o
    // trio do meio: são as duas que importam pra não fundir o motor,
    // ponto/flex/baro são só consulta ocasional. Nada aqui encosta no
    // conta-giros. ----
    const domeCy = h - bottomBandH * 0.58;
    const rFoot = Math.min(bottomBandH * 0.5, mainW * 0.095);
    const readoutCy = h - bottomBandH * 0.30;
    const readoutBoxH = bottomBandH * 0.62;
    const slotX = (frac) => mainW * frac;

    this._domeGauge(slotX(0.10), domeCy, rFoot, 'ÁGUA',
      this.coolantTemp != null ? this.coolantTemp.toFixed(0) : '—', '°C',
      // Faixa 60-120°C, não 0-130: abaixo de 60 é só "ainda esquentando"
      // (mesmo corte do azul/frio do coolantColor) e não interessa pela
      // posição, só pela cor — dá mais resolução de ponteiro justo na
      // faixa que se observa de verdade dirigindo (60-105, normal até o
      // limiar de alerta).
      this.coolantTemp != null ? (this.coolantTemp - 60) / 60 : 0, coolantColor(this.coolantTemp));

    this._readout(slotX(0.30), readoutCy, 'PONTO',
      this.speeduinoAdvance != null ? this.speeduinoAdvance.toFixed(0) : '—', '°',
      COLOR_GREEN, readoutBoxH);

    this._readout(slotX(0.5), readoutCy, 'FLEX',
      this.speeduinoFlexPct != null ? String(this.speeduinoFlexPct) : '—', '% etanol',
      COLOR_GREEN, readoutBoxH);

    this._readout(slotX(0.70), readoutCy, 'BARO',
      this.speeduinoBaro != null ? this.speeduinoBaro.toFixed(0) : '—', 'kPa',
      COLOR_GREEN, readoutBoxH);

    this._domeGauge(slotX(0.90), domeCy, rFoot, 'AR ADM.',
      this.intakeAirTemp != null ? this.intakeAirTemp.toFixed(0) : '—', '°C',
      // Faixa 10-60°C — limites dados pelo usuário, cobre o que se observa
      // na admissão sem desperdiçar resolução do ponteiro.
      this.intakeAirTemp != null ? (this.intakeAirTemp - 10) / 50 : 0, intakeAirColor(this.intakeAirTemp));

    // Repinta a tira à direita por cima de qualquer coisa — numa tela
    // estreita o conta-giros pode crescer até quase encostar nela (só não
    // pode invadir de vez, ver teto de rTach acima); isso garante que
    // nenhum resto de arco/zona colorida sobre atrás das barras.
    ctx.fillStyle = COLOR_BG;
    ctx.fillRect(mainW, 0, stripW, h);

    // ---- MAP/TPS: duas barras verticais lado a lado, nível subindo de
    // baixo pra cima — os dois andam quase juntos (carga do motor). ----
    const barPad = stripW * 0.14;
    const barW = (stripW - barPad * 3) / 2;
    const barX = mainW + barPad;
    const barY = h * 0.16;
    const barH = h * 0.68;
    this._verticalBar(barX, barY, barW, barH, 'MAP',
      this.speeduinoMap != null ? this.speeduinoMap.toFixed(0) : '—', 'kPa',
      this.speeduinoMap != null ? this.speeduinoMap / 105 : 0, COLOR_BLUE);

    this._verticalBar(barX + barW + barPad, barY, barW, barH, 'TPS',
      this.speeduinoTps != null ? this.speeduinoTps.toFixed(0) : '—', '%',
      this.speeduinoTps != null ? this.speeduinoTps / 100 : 0, tpsColor(this.speeduinoTps));
  }

  /** Conta-giros: ponteiro + zonas coloridas de fundo (verde/amarelo/
   * vermelho, mesmos limiares de sempre) + marcações numeradas, como um
   * tacômetro de painel de verdade. Velocidade em dígitos no centro —
   * é o dado que se quer ler rápido, não o RPM exato (esse já se sente
   * pelo motor; o ponteiro + zona de cor bastam). */
  _tach(cx, cy, r) {
    const ctx = this.ctx;
    const startDeg = 135;
    const sweepDeg = 270;
    const arcWidth = r * 0.09;

    // Zonas de fundo: verde até 4000, amarelo até 5500, vermelho até 7000.
    const zones = [
      { from: 0, to: RPM_YELLOW / RPM_MAX, color: COLOR_GREEN },
      { from: RPM_YELLOW / RPM_MAX, to: RPM_RED / RPM_MAX, color: COLOR_YELLOW },
      { from: RPM_RED / RPM_MAX, to: 1, color: COLOR_RED },
    ];
    ctx.lineWidth = arcWidth;
    ctx.lineCap = 'butt';
    for (const z of zones) {
      ctx.strokeStyle = z.color;
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      ctx.arc(cx, cy, r, ((startDeg + sweepDeg * z.from) * Math.PI) / 180, ((startDeg + sweepDeg * z.to) * Math.PI) / 180);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Marcações numeradas 0-7 (x1000 rpm), como num tacômetro de verdade.
    ctx.fillStyle = COLOR_LABEL;
    ctx.font = `${Math.round(r * 0.13)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 7; i++) {
      const deg = startDeg + sweepDeg * (i / 7);
      const tickOuter = polar(cx, cy, r * 1.02, deg);
      const tickInner = polar(cx, cy, r * 0.9, deg);
      ctx.strokeStyle = i * 1000 > RPM_RED ? COLOR_RED : 'rgba(255,255,255,0.4)';
      ctx.lineWidth = r * 0.02;
      ctx.beginPath();
      ctx.moveTo(tickInner.x, tickInner.y);
      ctx.lineTo(tickOuter.x, tickOuter.y);
      ctx.stroke();
      const labelPos = polar(cx, cy, r * 0.78, deg);
      ctx.fillText(String(i), labelPos.x, labelPos.y);
    }
    ctx.textBaseline = 'alphabetic';

    // Ponteiro — sai de uma "zona de segurança" ao redor do centro (onde
    // ficam os dígitos de RPM e velocidade, até 3 caracteres tipo "168")
    // até perto do aro. Raio interno maior que a metade da largura do
    // texto pra não cruzar por cima dos números.
    const rpmFrac = clamp01((this.rpm ?? 0) / RPM_MAX);
    const needleDeg = startDeg + sweepDeg * rpmFrac;
    const needleColor = rpmColor(this.rpm);
    const inner = polar(cx, cy, r * 0.62, needleDeg);
    const outer = polar(cx, cy, r * 0.85, needleDeg);
    ctx.strokeStyle = needleColor;
    ctx.lineWidth = r * 0.05;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(inner.x, inner.y);
    ctx.lineTo(outer.x, outer.y);
    ctx.stroke();

    // Tampa na ponta interna do ponteiro (não no centro — o centro é o
    // vão reservado pro dígito de velocidade).
    ctx.fillStyle = needleColor;
    ctx.beginPath();
    ctx.arc(inner.x, inner.y, r * 0.045, 0, Math.PI * 2);
    ctx.fill();

    // Legenda "RPM x1000" no vão de baixo do mostrador (contexto da escala
    // do ponteiro/zonas coloridas).
    ctx.fillStyle = COLOR_LABEL;
    ctx.font = `${Math.round(r * 0.1)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText('RPM ×1000', cx, cy + r * 0.62);

    // RPM em dígitos, acima da velocidade — o ponteiro dá a noção rápida
    // de zona, mas o valor exato também fica disponível, não só o
    // analógico.
    ctx.fillStyle = needleColor;
    ctx.font = `bold ${Math.round(r * 0.2)}px sans-serif`;
    ctx.fillText(this.rpm != null ? String(this.rpm) : '—', cx, cy - r * 0.24);
    ctx.fillStyle = COLOR_UNIT;
    ctx.font = `${Math.round(r * 0.09)}px sans-serif`;
    ctx.fillText('rpm', cx, cy - r * 0.1);

    // Velocidade em dígitos, o destaque do centro do mostrador.
    ctx.fillStyle = this.speed != null ? speedColor(this.speed) : COLOR_TEXT;
    ctx.font = `bold ${Math.round(r * 0.38)}px sans-serif`;
    ctx.fillText(this.speed != null ? String(this.speed) : '—', cx, cy + r * 0.28);
    ctx.fillStyle = COLOR_UNIT;
    ctx.font = `${Math.round(r * 0.12)}px sans-serif`;
    ctx.fillText('km/h', cx, cy + r * 0.46);
  }

  /** Gauge pequena em cúpula (meio-círculo, flat pra baixo) — usada pra
   * água/ar admissão na faixa de baixo. */
  _domeGauge(cx, cy, r, label, value, unit, fraction, color) {
    const ctx = this.ctx;
    const arcWidth = r * 0.16;
    fraction = clamp01(fraction);

    // Fundo (dá contraste pra sobrepor o conta-giros sem se perder nele).
    ctx.fillStyle = COLOR_GAUGE_BG;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 1.08, Math.PI, Math.PI * 2);
    ctx.closePath();
    ctx.fill();

    ctx.lineCap = 'round';
    ctx.strokeStyle = COLOR_ARC_BG;
    ctx.lineWidth = arcWidth;
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, Math.PI + Math.PI * fraction);
    ctx.stroke();

    ctx.textAlign = 'center';
    ctx.fillStyle = color;
    ctx.font = `bold ${Math.round(r * 0.44)}px sans-serif`;
    ctx.fillText(value, cx, cy - r * 0.28);
    ctx.fillStyle = COLOR_UNIT;
    ctx.font = `${Math.round(r * 0.22)}px sans-serif`;
    ctx.fillText(unit, cx, cy - r * 0.02);
    ctx.fillStyle = COLOR_LABEL;
    ctx.font = `${Math.round(r * 0.22)}px sans-serif`;
    ctx.fillText(label, cx, cy + r * 0.34);
  }

  /** Leitura periférica — sem forma de gauge, só um número grande
   * colorido; pra sinais de consulta ocasional (ponto/baro/flex), não algo
   * pra reagir em cima na hora. */
  _readout(cx, cy, label, value, unit, color, boxH) {
    const ctx = this.ctx;
    const scale = boxH * 0.42;
    ctx.textAlign = 'center';
    ctx.fillStyle = COLOR_LABEL;
    ctx.font = `${Math.round(scale * 0.34)}px sans-serif`;
    ctx.fillText(label, cx, cy - scale * 0.5);
    ctx.fillStyle = color;
    ctx.font = `bold ${Math.round(scale * 0.72)}px sans-serif`;
    ctx.fillText(value, cx, cy + scale * 0.28);
    ctx.fillStyle = COLOR_UNIT;
    ctx.font = `${Math.round(scale * 0.3)}px sans-serif`;
    ctx.fillText(unit, cx, cy + scale * 0.66);
  }

  /** Barra vertical que enche de baixo pra cima (nível, tipo bateria/
   * combustível). MAP e TPS ficam lado a lado (o mesmo _draw() as posiciona
   * adjacentes) porque andam quase juntos (carga do motor). */
  _verticalBar(x, y, w, h, label, value, unit, fraction, color) {
    const ctx = this.ctx;
    const labelH = h * 0.09;
    const valueH = h * 0.16;
    const trackY = y + labelH;
    const trackH = h - labelH - valueH;
    const trackBottom = trackY + trackH;
    fraction = clamp01(fraction);

    ctx.textAlign = 'center';
    ctx.fillStyle = COLOR_LABEL;
    ctx.font = `${Math.round(Math.min(w * 0.42, labelH * 0.8))}px sans-serif`;
    ctx.fillText(label, x + w / 2, y + labelH * 0.85);

    ctx.fillStyle = COLOR_GAUGE_BG;
    ctx.beginPath();
    ctx.roundRect(x, trackY, w, trackH, Math.min(6, w * 0.2));
    ctx.fill();

    const fillH = trackH * fraction;
    if (fillH > 0) {
      const drawH = Math.max(fillH, Math.min(6, w * 0.2));
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x, trackBottom - drawH, w, drawH, Math.min(6, w * 0.2));
      ctx.fill();
      // Preenchimento é retangular acima da base arredondada — só a borda
      // de baixo precisa de canto, o resto é reto (nível "subindo" mesmo).
      if (fillH > 6) {
        ctx.fillRect(x, trackBottom - fillH, w, fillH - 6);
      }
    }

    ctx.fillStyle = color;
    ctx.font = `bold ${Math.round(Math.min(w * 0.46, valueH * 0.7))}px sans-serif`;
    ctx.fillText(value, x + w / 2, trackY + trackH + valueH * 0.62);
    ctx.fillStyle = COLOR_UNIT;
    ctx.font = `${Math.round(Math.min(w * 0.3, valueH * 0.4))}px sans-serif`;
    ctx.fillText(unit, x + w / 2, trackY + trackH + valueH * 0.94);
  }
}
