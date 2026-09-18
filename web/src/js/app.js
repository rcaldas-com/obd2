// Orquestração: conexão, loop de poll, estado de tela — equivalente ao
// MainActivity.java, só que hoje com uma tela só (lambda). O loop já é
// estruturado por `screen` de propósito: quando outra tela existir, cada
// uma lê só o que precisa (mesma razão do Android — a porta serial do
// ELM327 é uma só a 38400 baud, e cada PID a mais nela derruba a taxa de
// todos). Enquanto não grava, só a tela visível é lida — é o ponto que
// o usuário pediu pra observar ao portar.
import { Elm327Manager } from './elm327.js';
import { LambdaChartView } from './lambdaChart.js';

const POLL_INTERVAL_MS = 50; // igual ao Android — os PIDs/timeouts do ELM327 já controlam o ritmo real

const els = {
  btnConnect: document.getElementById('btn-connect'),
  btnDisconnect: document.getElementById('btn-disconnect'),
  status: document.getElementById('status'),
  canvas: document.getElementById('chart'),
};

const elm327 = new Elm327Manager();
const chart = new LambdaChartView(els.canvas);

// Só existe 'lambda' por enquanto — outras telas (ponto, informações
// gerais) entram aqui como mais um valor, com seu próprio ramo no loop.
let screen = 'lambda';
let polling = false;
let pollTimer = null;

function setStatus(text) {
  els.status.textContent = text;
}

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
  setStatus('Desconectado');
  els.btnConnect.hidden = false;
  els.btnConnect.disabled = false;
  els.btnDisconnect.hidden = true;
}

function startPolling() {
  if (polling) return;
  polling = true;
  pollLoop();
}

function stopPolling() {
  polling = false;
  if (pollTimer) clearTimeout(pollTimer);
}

async function pollLoop() {
  if (!polling || !elm327.isConnected()) return;

  if (screen === 'lambda') {
    const data = await elm327.readLambdaData();
    chart.addData(data.o2s1Lambda, data.o2s5Lambda);
  }
  // outros `screen === '...'` entram aqui quando existirem — sem poll de
  // telas que não estão visíveis, mesma regra do app Android.

  if (polling) {
    pollTimer = setTimeout(pollLoop, POLL_INTERVAL_MS);
  }
}

els.btnConnect.addEventListener('click', connect);
els.btnDisconnect.addEventListener('click', disconnect);

if (!('serial' in navigator)) {
  setStatus('WebSerial indisponível — abra em Chrome/Edge via http://localhost');
  els.btnConnect.disabled = true;
}
