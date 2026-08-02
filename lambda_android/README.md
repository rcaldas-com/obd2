# Lambda Monitor - App Android OBD2

App nativo Android para monitorar sensores de lambda (O2 wideband) em tempo real via ELM327 USB.
Otimizado para multimídias automotivas com tela pequena e processador lento.

## Funcionalidades

- **Gráfico fullscreen** com sinal das duas sondas lambda em tempo real
- **Apenas 2 PIDs** (0134 + 0138) → máxima taxa de atualização (~8-10 Hz)
- **Indicador visual**: verde (Banco 1), vermelho (Banco 2)
- **Log CSV automático** salvo no armazenamento do dispositivo
- **Auto-detect** do adaptador USB quando conectado
- **Tela sempre ligada** + modo fullscreen immersive
- Compatível com **Android 5.0+** (API 21)
- **Não precisa de root** - usa USB Host API nativa

## Dados lidos (PIDs OBD2)

| PID    | Descrição                        |
|--------|----------------------------------|
| 0x0134 | O2 Sensor 1 - Lambda + Corrente |
| 0x0138 | O2 Sensor 5 - Lambda + Corrente |

## Adaptadores compatíveis

Qualquer ELM327 USB com chip:
- **CH340** (maioria dos clones chineses)
- **FTDI** (FT232R)
- **CP2102** (Silicon Labs)
- **Prolific** (PL2303)

## Como compilar

**Requer JDK 21** (path: `/usr/lib/jvm/java-21-openjdk-amd64`)

### Linha de comando (recomendado)
```bash
cd ~/obd2/lambda_android
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 ./gradlew assembleDebug
```
O APK será gerado em `app/build/outputs/apk/debug/app-debug.apk`

### Instalar via ADB
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Android Studio
1. File → Open → selecione a pasta `lambda_android`
2. Configure o JDK 21 em File → Project Structure → SDK Location
3. Run → Run 'app'

## Uso

1. Conecte o adaptador ELM327 USB na multimídia/tablet
2. O app abre automaticamente (ou abra manualmente)
3. Toque em **CONECTAR**
4. Aceite a permissão USB se solicitado
5. Os dados começam a ser exibidos em tempo real

## Logs CSV

Os logs são salvos automaticamente em:
```
Android/data/com.obd2.lambda/files/logs/lambda_YYYY-MM-DD_HH-MM.csv
```

Formato: `timestamp,o2s1_current,o2s1_lambda,o2s5_current,o2s5_lambda,stft1,stft2,rpm,timing`

## Estrutura

```
lambda_android/
├── app/
│   ├── build.gradle
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/obd2/lambda/
│       │   ├── MainActivity.java      # Activity principal
│       │   ├── Elm327Manager.java      # Comunicação ELM327/OBD2
│       │   └── LambdaChartView.java    # Gráfico custom (Canvas)
│       └── res/
│           ├── layout/activity_main.xml
│           ├── xml/device_filter.xml   # Filtro USB vendor/product IDs
│           └── values/strings.xml
├── build.gradle
├── settings.gradle
└── gradle.properties
```
