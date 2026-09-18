// Orquestração: conexão, loops de poll, troca de tela, alertas e log —
// equivalente ao MainActivity.java (só a parte de lambda/dashboard/alertas/
// log; a tela de ignição/KnockWatch não foi portada, fica só no Android).
//
// Dois loops de poll independentes, cada um dono da própria porta serial,
// mesma divisão do Android: o do ELM327 respeita a tela ativa (cada PID a
// mais nessa porta derruba a taxa de todos); o da Speeduino roda sempre que
// conectada, não importa a tela, porque não disputa banda com o ELM327.
//
// UI: barra inferior mínima (mesmo papel da barra do APK) com só o
// essencial pra dirigir — configurações, voltagem, pause/resume do log,
// alternar tela; conexões/alertas/log ficam atrás do botão ☰.
import { Elm327Manager } from './elm327.js';
import { SpeeduinoManager } from './speeduino.js';
import { LambdaChartView } from './lambdaChart.js';
import { DashboardView } from './dashboard.js';
import { AlertManager } from './alerts.js';
import { MslLogger } from './mslLogger.js';

const POLL_INTERVAL_MS = 50; // igual ao Android — os PIDs/timeouts do ELM327 já controlam o ritmo real
const ALERT_CHECK_INTERVAL_MS = 3000; // cadência lenta de voltagem/água na tela de lambda

const els = {
  btnSettings: document.getElementById('btn-settings'),
  btnCloseSettings: document.getElementById('btn-close-settings'),
  settingsPanel: document.getElementById('settings-panel'),
  barVoltage: document.getElementById('bar-voltage'),
  btnPauseLog: document.getElementById('btn-pause-log'),
  btnToggleScreen: document.getElementById('btn-toggle-screen'),

  btnConnect: document.getElementById('btn-connect'),
  btnDisconnect: document.getElementById('btn-disconnect'),
  status: document.getElementById('status'),
  btnConnectSpeeduino: document.getElementById('btn-connect-speeduino'),
  btnDisconnectSpeeduino: document.getElementById('btn-disconnect-speeduino'),
  statusSpeeduino: document.getElementById('status-speeduino'),

  chartCanvas: document.getElementById('chart'),
  dashboardCanvas: document.getElementById('dashboard'),
  alertBanner: document.getElementById('alert-banner'),

  cfgVoltageMin: document.getElementById('cfg-voltage-min'),
  cfgVoltageHyst: document.getElementById('cfg-voltage-hyst'),
  cfgTempMax: document.getElementById('cfg-temp-max'),
  cfgTempHyst: document.getElementById('cfg-temp-hyst'),
  btnSaveAlerts: document.getElementById('btn-save-alerts'),

  btnLogStart: document.getElementById('btn-log-start'),
  btnLogStop: document.getElementById('btn-log-stop'),
  logStatus: document.getElementById('log-status'),
};

const elm327 = new Elm327Manager();
const speeduino = new SpeeduinoManager();
const chart = new LambdaChartView(els.chartCanvas);
const dashboard = new DashboardView(els.dashboardCanvas);
const alertManager = new AlertManager();
const mslLogger = new MslLogger();

let screen = 'lambda'; // 'lambda' | 'dashboard'
let polling = false;
let pollTimer = null;
let lastAlertCheckTime = 0;

let speeduinoPolling = false;
let speeduinoPollTimer = null;
let lastSpeeduinoData = null;

function setStatus(text) {
  els.status.textContent = text;
}

function setSpeeduinoStatus(text) {
  els.statusSpeeduino.textContent = text;
}

function setBatteryVoltage(v) {
  els.barVoltage.textContent = `${v.toFixed(1)}V`;
}

function renderAlerts(alerts) {
  els.alertBanner.innerHTML = '';
  els.alertBanner.hidden = alerts.length === 0;
  for (const msg of alerts) {
    const div = document.createElement('div');
    div.className = 'alert-item';
    div.textContent = msg;
    els.alertBanner.appendChild(div);
  }
}

// ---- ELM327 ----

async function connect() {
  els.btnConnect.disabled = true;
  setStatus('Conectando…');
  try {
    await elm327.connect();
    setStatus('Conectado');
    els.btnConnect.hidden = true;
    els.btnDisconnect.hidden = false;
    startPolling();
  } catch (e) {
    console.error(e);
    setStatus('Falha ao conectar: ' + (e.message || e));
    els.btnConnect.disabled = false;
  }
}

async function disconnect() {
  stopPolling();
  await elm327.disconnect();
  chart.clearData();
  dashboard.clearData();
  setStatus('Desconectado');
  els.btnConnect.hidden = false;
  els.btnConnect.disabled = false;
  els.btnDisconnect.hidden = true;
}

function startPolling() {
  if (polling) return;
  polling = true;
  lastAlertCheckTime = 0; // lê voltagem/temperatura já no primeiro ciclo
  pollLoop();
}

function stopPolling() {
  polling = false;
  if (pollTimer) clearTimeout(pollTimer);
}

async function pollLoop() {
  if (!polling || !elm327.isConnected()) return;

  // Gravando, o ELM327 fica dedicado a lambda banco 1/2 o tempo todo, não
  // importa a tela — é a única fonte que existe (as sondas estão na
  // injeção original, não na Speeduino) e o log precisa disso ao vivo.
  const recording = mslLogger.isRecording();

  if (screen === 'lambda' || recording) {
    const data = await elm327.readLambdaData();
    mslLogger.updateObd2Lambda(data.o2s1Lambda, data.o2s5Lambda);

    if (screen === 'lambda') {
      chart.addData(data.o2s1Lambda, data.o2s5Lambda);
    } else if (screen === 'dashboard') {
      // Gravando com o dashboard na frente: nada de consultar o ELM327
      // pros PIDs dele (CLT/IAT/velocidade), isso atrasaria o lambda que o
      // log está contando pra ter ao vivo — usa o que a Speeduino já
      // entrega de graça, em porta/thread própria, sem custo nenhum aqui.
      updateDashboardFromSpeeduino();
    }
    await pollSlowAlerts();
  } else if (screen === 'dashboard') {
    const data = await elm327.readDashboardData();
    // Dashboard já lê voltagem+temperatura toda vez (não faz PIDs de
    // lambda), então os alertas usam esses mesmos dados.
    if (data.batteryVoltage != null) setBatteryVoltage(data.batteryVoltage);
    dashboard.updateData(data);
    renderAlerts(alertManager.evaluate(data.batteryVoltage, data.coolantTemp));
  }

  if (polling) {
    pollTimer = setTimeout(pollLoop, POLL_INTERVAL_MS);
  }
}

/** Dashboard durante gravação, sem tocar no ELM327: usa o que a Speeduino
 * (porta própria, atualizando sempre) já tem. */
function updateDashboardFromSpeeduino() {
  const sd = lastSpeeduinoData;
  const data = { timestamp: Date.now() };
  if (sd) {
    data.rpm = sd.rpm;
    data.coolantTemp = sd.coolantC;
    data.intakeAirTemp = sd.iatC;
    data.batteryVoltage = sd.batteryV;
  }
  dashboard.updateData(data);
  renderAlerts(alertManager.evaluate(data.batteryVoltage, data.coolantTemp));
}

/** Consultas leves de fundo (voltagem, água) na cadência lenta — rodam na
 * tela de lambda pra não roubar banda dela, mantendo os alertas vivos
 * durante um teste de pista mesmo sem trocar de tela. */
async function pollSlowAlerts() {
  const now = Date.now();
  if (now - lastAlertCheckTime < ALERT_CHECK_INTERVAL_MS) return;
  lastAlertCheckTime = now;

  const v = await elm327.readBatteryVoltage();
  const temp = await elm327.readCoolantTemp();
  if (v != null) setBatteryVoltage(v);
  renderAlerts(alertManager.evaluate(v, temp));
}

// ---- Speeduino ----

async function connectSpeeduino() {
  els.btnConnectSpeeduino.disabled = true;
  setSpeeduinoStatus('Conectando…');
  try {
    await speeduino.connect();
    const ok = await speeduino.verifySignature();
    if (!ok) throw new Error('assinatura não confere — porta errada?');
    await speeduino.readStoich();
    setSpeeduinoStatus('Speeduino conectada');
    els.btnConnectSpeeduino.hidden = true;
    els.btnDisconnectSpeeduino.hidden = false;
    startSpeeduinoPolling();
  } catch (e) {
    console.error(e);
    setSpeeduinoStatus('Falha ao conectar: ' + (e.message || e));
    await speeduino.disconnect();
    els.btnConnectSpeeduino.disabled = false;
  }
}

async function disconnectSpeeduino() {
  stopSpeeduinoPolling();
  await speeduino.disconnect();
  lastSpeeduinoData = null;
  dashboard.clearSpeeduinoData();
  setSpeeduinoStatus('Speeduino desconectada');
  els.btnConnectSpeeduino.hidden = false;
  els.btnConnectSpeeduino.disabled = false;
  els.btnDisconnectSpeeduino.hidden = true;
}

function startSpeeduinoPolling() {
  if (speeduinoPolling) return;
  speeduinoPolling = true;
  speeduinoPollLoop();
}

function stopSpeeduinoPolling() {
  speeduinoPolling = false;
  if (speeduinoPollTimer) clearTimeout(speeduinoPollTimer);
}

/** Independente do pollLoop do ELM327 — porta USB própria, sem disputa.
 * Roda sempre (não só na tela de dashboard), pra alimentar o MslLogger
 * continuamente mesmo com o gráfico de lambda em primeiro plano. */
async function speeduinoPollLoop() {
  if (!speeduinoPolling || !speeduino.isConnected()) return;

  const data = await speeduino.readOutputChannels();
  mslLogger.updateSpeeduino(data);
  lastSpeeduinoData = data;
  dashboard.updateSpeeduinoData(data);

  if (speeduinoPolling) {
    speeduinoPollTimer = setTimeout(speeduinoPollLoop, POLL_INTERVAL_MS);
  }
}

// ---- Tela ----
// O botão anuncia a PRÓXIMA tela do ciclo (mesma ideia do APK): "⚙" em cima
// da tela de lambda (próxima = dashboard), "λ" em cima do dashboard
// (próxima = lambda).

function applyScreen() {
  els.chartCanvas.hidden = screen !== 'lambda';
  els.dashboardCanvas.hidden = screen !== 'dashboard';
  els.btnToggleScreen.textContent = screen === 'lambda' ? '⚙' : 'λ';
}

function toggleScreen() {
  screen = screen === 'lambda' ? 'dashboard' : 'lambda';
  applyScreen();
}

els.btnToggleScreen.addEventListener('click', toggleScreen);
applyScreen();

// ---- Configurações (painel atrás do ☰) ----

function openSettings() {
  els.cfgVoltageMin.value = alertManager.voltageMin;
  els.cfgVoltageHyst.value = alertManager.voltageHysteresis;
  els.cfgTempMax.value = alertManager.tempMax;
  els.cfgTempHyst.value = alertManager.tempHysteresis;
  els.settingsPanel.hidden = false;
}

function closeSettings() {
  els.settingsPanel.hidden = true;
}

function saveAlertSettings() {
  const voltageMin = parseFloat(els.cfgVoltageMin.value);
  const voltageHyst = parseFloat(els.cfgVoltageHyst.value);
  const tempMax = parseFloat(els.cfgTempMax.value);
  const tempHyst = parseFloat(els.cfgTempHyst.value);
  if ([voltageMin, voltageHyst, tempMax, tempHyst].some(Number.isNaN)) return;
  alertManager.saveSettings(voltageMin, voltageHyst, tempMax, tempHyst);
}

els.btnSettings.addEventListener('click', openSettings);
els.btnCloseSettings.addEventListener('click', closeSettings);
els.btnSaveAlerts.addEventListener('click', saveAlertSettings);

// ---- Log ----

function updateLogUI() {
  const recording = mslLogger.isRecording();
  const paused = mslLogger.isPaused();

  els.btnLogStart.hidden = recording;
  els.btnLogStop.hidden = !recording;
  els.logStatus.textContent = recording ? (paused ? 'Log pausado' : 'Gravando log…') : '';

  // Pausa/retoma fica na barra inferior, acessível em qualquer tela
  // enquanto grava (mesmo botão único do APK, texto/cor trocam de estado).
  els.btnPauseLog.hidden = !recording;
  els.btnPauseLog.textContent = paused ? '►' : 'II';
  els.btnPauseLog.style.color = paused ? '#81c784' : '#ff5252';
}

async function startLog() {
  try {
    await mslLogger.start();
    updateLogUI();
  } catch (e) {
    // AbortError = usuário cancelou o diálogo "Salvar como" — não é erro.
    if (e.name !== 'AbortError') {
      console.error(e);
      els.logStatus.textContent = 'Falha ao iniciar log: ' + (e.message || e);
    }
  }
}

async function stopLog() {
  await mslLogger.stop();
  updateLogUI();
}

els.btnLogStart.addEventListener('click', startLog);
els.btnLogStop.addEventListener('click', stopLog);
els.btnPauseLog.addEventListener('click', () => {
  if (mslLogger.isPaused()) mslLogger.resume();
  else mslLogger.pause();
  updateLogUI();
});

els.btnConnect.addEventListener('click', connect);
els.btnDisconnect.addEventListener('click', disconnect);
els.btnConnectSpeeduino.addEventListener('click', connectSpeeduino);
els.btnDisconnectSpeeduino.addEventListener('click', disconnectSpeeduino);

if (!('serial' in navigator)) {
  setStatus('WebSerial indisponível — abra em Chrome/Edge via http://localhost');
  setSpeeduinoStatus('WebSerial indisponível');
  els.btnConnect.disabled = true;
  els.btnConnectSpeeduino.disabled = true;
}
if (!('showSaveFilePicker' in window)) {
  els.btnLogStart.disabled = true;
  els.logStatus.textContent = 'Log indisponível neste navegador';
}
