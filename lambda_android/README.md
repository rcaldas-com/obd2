# Lambda Monitor - App Android OBD2 + Speeduino

App nativo Android para acompanhar o motor em tempo real numa multimídia
automotiva, lendo por USB **duas fontes ao mesmo tempo**: a ECU original do
carro (via ELM327) e a Speeduino (protocolo do TunerStudio).

Otimizado para multimídia com tela pequena, processador lento e Android antigo
(5.0+).

## Telas

O botão do canto inferior direito alterna em ciclo: **λ → ponto → informações
gerais**.

### λ — Gráfico de lambda
Duas sondas wideband em tempo real, fullscreen. Lê **apenas 2 PIDs** (0134 +
0138) para taxa máxima (~8-10 Hz). Cores por estado: verde estequiométrica, azul
rico, amarelo pobre, vermelho extremo.

### ° — Gráfico de ponto de ignição
Para achar o maior ponto que o motor aguenta em cada condição enquanto o mapa da
Speeduino é ajustado.

- Curva do **ponto da ECU original** (PID 010E) — é ela que denuncia quando o
  sensor de detonação da original acusa e ela recua o ponto.
- Curva do **ponto da Speeduino** (tracejada), quando disponível, só como
  referência visual do que foi alterado.
- **Escala Y adaptativa**: o que se procura é um recuo de 2-3°, que numa escala
  fixa de 0-50° seria invisível.
- **Detecção automática de recuo**: com rotação/MAP/TPS estáveis por 4 s, o app
  fixa uma referência e mostra o recuo contra ela. Passou do limiar, pinta a
  borda de vermelho e **dá um bipe** — sem depender de alguém olhando o gráfico.
- Botão **REF** trava/destrava a referência manualmente.

> ⚠️ O recuo da ECU original é um indicador **auxiliar**: a janela de escuta do
> sensor de detonação dela é sincronizada com o ponto que *ela* comandaria, e
> com a Speeduino muito mais avançada a detonação pode cair fora dessa janela.
> Use para achar o limite com margem, não como única rede de segurança.

**Funciona só com o ELM327 plugado.** Isso é proposital: para alterar o ponto ao
vivo o TunerStudio precisa da porta USB da Speeduino, e só um programa por
porta. Sem a Speeduino, o app tira rotação/MAP/TPS da própria ECU original.

### ⚙ — Informações gerais
Rotação, temperaturas, velocidade (GPS quando disponível), bateria, e os dados
da Speeduino: ponto aplicado, MAP, TPS, VE, alvo de AFR, largura de pulso,
**% de etanol do sensor flex**.

## Dados lidos

**Da ECU original (ELM327):**

| PID    | Descrição                        | Onde é usado           |
|--------|----------------------------------|------------------------|
| 0x0134 | O2 Sensor 1 — lambda + corrente  | tela λ                 |
| 0x0138 | O2 Sensor 5 — lambda + corrente  | tela λ                 |
| 0x010E | Ponto de ignição                 | tela de ponto, log     |
| 0x010C | Rotação                          | ponto (estabilidade)   |
| 0x010B | Pressão do coletor               | ponto (estabilidade)   |
| 0x0111 | Posição do acelerador            | ponto (estabilidade)   |
| 0x0105 | Temp. arrefecimento              | dashboard, alertas     |
| 0x010F | Temp. ar admissão                | dashboard              |
| 0x010D | Velocidade                       | dashboard              |

Mais os PIDs de `ObdPid.java` disponíveis para alertas personalizados.

**Da Speeduino:** bloco de 130 bytes de output channels (rotação, MAP, IAT,
água, bateria, VE, alvo de AFR, ponto aplicado, TPS, etanol, PW, dwell).

## Alertas

Voltagem e temperatura com histerese configurável, mais alertas personalizados
por PID (com descoberta automática dos PIDs que a ECU confirma suportar).
Aparecem como tarjas vermelhas sobre qualquer tela.

## Logs

Formato **`.msl`** (TunerStudio/MegaLogViewer), com as duas fontes na mesma
linha a 10 Hz.

**Início e parada são manuais**, por um botão em Configurações — nada é gravado
automaticamente. Um indicador `● REC` aparece nas telas enquanto grava.

```
Android/data/com.obd2.lambda/files/logs/
```

Colunas: `Time, RPM, MAP, TPS, CLT, IAT, Advance _Current, Baro Pressure, VE1,
VE2, AFR Target, Battery V, PW, Lambda, Lambda2, Advance_OBD2, Ethanol`

(`Advance _Current` = Speeduino; `Advance_OBD2` = ECU original.)

## Adaptadores compatíveis

Qualquer ELM327 USB com chip **CH340**, **FTDI** (FT232R), **CP2102** ou
**Prolific** (PL2303). Quando há mais de um dispositivo USB conectado, o app
pergunta qual é o ELM327 e qual é a Speeduino (Configurações → dispositivos).

## Como compilar

Requer **JDK 21 com `javac`** (uma instalação só de JRE não serve) e o
**Android SDK** (platform 34 + build-tools 34).

```bash
cd /var/rcaldas/obd2/lambda_android
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 ANDROID_HOME=/var/rcaldas/android-sdk ./gradlew assembleDebug
```

APK em `app/build/outputs/apk/debug/app-debug.apk`.

Se o Gradle reclamar de `does not provide the required capabilities:
[JAVA_COMPILER]` logo após instalar o JDK, é cache do daemon — rode
`./gradlew --stop` e tente de novo.

### Instalar via ADB
```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Android Studio
1. File → Open → pasta `lambda_android`
2. Configure o JDK 21 em File → Project Structure → SDK Location
3. Run → Run 'app'

## Uso

1. Conecte o ELM327 (e, se for o caso, a Speeduino) na USB da multimídia
2. O app abre automaticamente, ou abra manualmente
3. **CONECTAR** e aceite a permissão USB
4. Alterne entre as telas pelo botão do canto

## Estrutura

```
lambda_android/app/src/main/
├── AndroidManifest.xml
├── java/com/obd2/lambda/
│   ├── MainActivity.java        # telas, threads de polling, orquestração
│   ├── Elm327Manager.java       # ECU original (ELM327/OBD2)
│   ├── SpeeduinoManager.java    # Speeduino (protocolo TunerStudio, só leitura)
│   ├── KnockWatch.java          # detecção do recuo de ponto da original
│   ├── IgnitionChartView.java   # gráfico de ponto (escala adaptativa)
│   ├── LambdaChartView.java     # gráfico de lambda (escala fixa)
│   ├── DashboardView.java       # tela de informações gerais
│   ├── MslLogger.java           # log .msl das duas fontes
│   ├── AlertManager.java        # alertas com histerese
│   ├── ObdPid.java              # PIDs decodificáveis
│   ├── DeviceRoleManager.java   # qual USB é ELM327, qual é Speeduino
│   ├── UsbSerialSession.java
│   ├── GpsSpeedProvider.java
│   └── OBD2ForegroundService.java
└── res/
    ├── layout/activity_main.xml
    ├── xml/device_filter.xml    # filtro USB vendor/product IDs
    └── values/strings.xml
```
