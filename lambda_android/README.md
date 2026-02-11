# Lambda Monitor - App Android OBD2

App nativo Android para monitorar sensores de lambda (O2 wideband) em tempo real via ELM327 USB.

## Funcionalidades

- **Leitura em tempo real** dos sensores O2 wideband (Banco 1 e Banco 2)
- **Gráfico scrolling** com histórico dos últimos 80 pontos
- **Indicador visual**: azul (pobre/lean), verde (estequiométrico), vermelho (rico/rich)
- **Short Fuel Trim** banco 1 e 2 com indicação por cor
- **RPM e Avanço de ignição**
- **Log CSV automático** salvo no armazenamento do dispositivo
- **Auto-detect** do adaptador USB quando conectado
- **Tela sempre ligada** durante monitoramento
- Compatível com **Android 5.0+** (API 21)
- **Não precisa de root** - usa USB Host API nativa

## Dados lidos (PIDs OBD2)

| PID    | Descrição                        |
|--------|----------------------------------|
| 0x0134 | O2 Sensor 1 - Lambda + Corrente |
| 0x0138 | O2 Sensor 5 - Lambda + Corrente |
| 0x010C | RPM                              |
| 0x0106 | Short Fuel Trim Banco 1          |
| 0x0108 | Short Fuel Trim Banco 2          |
| 0x010E | Avanço de Ignição                |

## Adaptadores compatíveis

Qualquer ELM327 USB com chip:
- **CH340** (maioria dos clones chineses)
- **FTDI** (FT232R)
- **CP2102** (Silicon Labs)
- **Prolific** (PL2303)

## Como compilar

### Opção 1: Android Studio
1. Abra o Android Studio
2. File → Open → selecione a pasta `lambda_android`
3. Aguarde o Gradle sync
4. Run → Run 'app' (com o Android conectado via ADB)

### Opção 2: Linha de comando
```bash
cd lambda_android
chmod +x gradlew
./gradlew assembleDebug
```
O APK será gerado em `app/build/outputs/apk/debug/app-debug.apk`

### Instalar via ADB
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

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
