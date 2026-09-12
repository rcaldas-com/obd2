# lambda_android (OBD2 / Speeduino live dashboard)

Projeto Android Studio (`lambda_android/`), repo `rcaldas-com/obd2`. Distinto do
app web `car` (manutenções, Next.js, outro repo) — notas antigas de lá citam
OBD2 como "integração em andamento"; este repo é essa integração, rodando em
hardware real.

**O que é:** painel ao vivo numa multimídia automotiva Android, lendo por USB
duas fontes independentes ao mesmo tempo:
- **ELM327** → ECU original do carro (PIDs OBD2 padrão);
- **Speeduino** → a ECU que realmente comanda ignição/injeção hoje, protocolo
  binário do TunerStudio.

Mais um logger `.msl` (formato TunerStudio) e velocidade por GPS.

## O carro: duas ECUs ao mesmo tempo

A original continua no carro, alimentada e lendo os próprios sensores
(**inclusive o de detonação**), mas **não comanda mais a ignição** — quem
comanda é a Speeduino. Isso é a base do recurso mais importante do app hoje
(tela de ponto, abaixo): o ponto que a original reporta no PID 010E é o que
*ela* comandaria, e ela recua esse valor quando o sensor de detonação dela
acusa. Ou seja, dá pra usar a original como detector de detonação emprestado
enquanto se sobe o ponto na Speeduino.

**Firmware da Speeduino é modificado**, não é o oficial: fica em
`~/docs/speeduino/speeduino-202501.7_6ign` — build 202501.7 alterada para 6
saídas de ignição (o carro usa Wasted COP). A assinatura serial continua
`"speeduino 202501"`, que é o que `SpeeduinoManager.verifySignature()` espera —
não mudar isso sem conferir lá.

## As três telas (e por que cada uma lê o que lê)

O botão do canto alterna em ciclo: **λ → ponto → informações gerais**. Cada tela
manda numa cadência diferente de leitura do ELM327 (ver `pollRunnable` em
`MainActivity`), porque o ELM327 é uma porta serial só a 38400 e cada PID a mais
derruba a taxa de todos:

1. **Lambda (λ)** — só os dois PIDs wideband (0134/0138) pra taxa máxima
   (~8-10 Hz). Voltagem/água/alertas entram a cada 3 s (`ALERT_CHECK_INTERVAL_MS`)
   pra não roubar banda.
2. **Ponto de ignição** — só o 010E no loop rápido. Detalhe importante: antes
   deste trabalho o 010E era lido *junto com os alertas*, a cada 3 s, e um
   evento de recuo cabia inteiro entre duas amostras.
3. **Informações gerais** — pacote lento (rotação, temperaturas, velocidade,
   bateria) + dados da Speeduino, incluindo **% de etanol do sensor flex**
   (`DashboardView`, já funcionava).

## Tela de ponto: o desenho e os porquês

Objetivo: achar o maior ponto que o motor aguenta em cada condição, com a
gasolina de hoje (mais álcool que na época do projeto do motor), usando o recuo
da original como evidência de detonação.

- **`KnockWatch.java`** — lógica pura (sem Android), fácil de raciocinar em
  cima. Só considera a condição "estável" quando rotação/MAP/TPS ficam numa
  faixa estreita por 4 s; aí fixa uma referência (mediana do 010E na janela) e
  passa a medir `recuo = referência − ponto atual`. Passou de 2° por 2 amostras,
  é evento (com histerese pra sair, no mesmo espírito do `AlertManager`).
  - A referência **sobe** junto se a original adiantar sozinha, mas **nunca
    desce** — se descesse, absorveria o próprio recuo que se está caçando.
  - Tolerâncias folgadas de propósito (±150 rpm, ±8 kPa, ±3% TPS): subir ponto
    numa condição estável mexe um pouco na rotação/carga *por causa do próprio
    ajuste sendo testado*; apertado demais soltaria a referência exatamente na
    hora do teste.
  - Botão **REF** trava/destrava a referência à mão (travada, sobrevive à
    condição sair da faixa).
- **`IgnitionChartView.java`** — escala Y **adaptativa**, ao contrário do
  gráfico de lambda que é fixa. O que se procura é um recuo de 2-3°; numa escala
  fixa de 0-50° isso seria 4% da altura da tela, invisível justo no que importa.
  Vão mínimo de 8° e limites presos a passos de 2° pra não tremer a cada amostra.
- Evento dá **bipe** (`ToneGenerator`) além do aviso visual — o ajuste é feito
  com o carro andando, não dá pra depender de alguém olhando o gráfico.

**Limitação conhecida, e é séria:** a janela em que a original escuta o sensor
de detonação é sincronizada com o ângulo onde *ela* espera a combustão. Com a
Speeduino bem mais avançada, a detonação acontece mais cedo e pode cair fora
dessa janela — detonação real sem recuo no 010E. Ou seja, **quando falha, falha
em silêncio e para o lado perigoso**. Serve pra achar o limite com margem, não
como única rede de segurança. O certo, mais pra frente, é entrada de detonação
na própria Speeduino (TPIC8101/HIP9011), janelada contra o ponto que ela mesma
comanda; aí o 010E vira segunda opinião.

## A porta serial é uma só — e por isso a tela de ponto não depende da Speeduino

Pra **alterar** o ponto ao vivo hoje é preciso TunerStudio no notebook, ligado
na USB da Speeduino. Só um mestre por porta: com o TS conectado, o app **não**
consegue falar com a Speeduino.

Por isso a tela de ponto foi feita pra funcionar **só com o ELM327**: o 010E é
da original, e rotação/MAP/TPS (que o `KnockWatch` usa pra medir estabilidade)
também saem dela (PIDs 010C/010B/0111, via `readGenericPid`). Quando a Speeduino
*está* disponível, o app prefere os dados dela (é a carga que os mapas usam de
verdade) e ainda libera o ELM327 pra ficar 100% no 010E —
ver `readIgnitionContextStep()`.

Alternativas descartadas por ora: **Serial3** do Mega existe no firmware mas
exigiria fiação nova do cofre até o painel; **FTDI + extensor USB** não resolve
dois mestres numa porta, só permite levar o notebook pra dentro do carro (o que
é o método atual de escrever).

## Alterar o ponto pelo app (ainda não implementado) — tudo que já foi levantado

`SpeeduinoManager` é **somente leitura** hoje, de propósito. Se for implementar
escrita, o levantamento já está feito:

**Protocolo** (conferido em `comms.cpp` do firmware modificado; mesmo envelope
tamanho+payload+CRC32 que o `readOutputChannels()` já usa):
- **Ler página** — `'p'`: `['p', tsCanId, página, offsetLo, offsetHi, lenLo, lenHi]`
- **Escrever** — `'M'`: `['M', tsCanId, página, offsetLo, offsetHi, lenLo, lenHi, dados...]`
- **Gravar na EEPROM** — `'b'`: `['b', tsCanId, página]`
  (sem o `'b'`, a alteração fica só na RAM e um reset volta ao mapa salvo — o
  que é uma rede de segurança natural na fase de testes)

**Tabela de ignição = página 3** (de `projectCfg/mainController.ini` do projeto
TS em `~/docs/tuner/Omega_3.6_0.3/`, que é o `.ini` deste carro; `pageSize` da
página 3 = 288 bytes):
| campo | tipo | offset | tamanho | conversão |
|-------|------|--------|---------|-----------|
| `advTable1` (mapa de ponto) | U08 | 0 | 16×16 = 256 | **graus = byte − 40** |
| `rpmBins2` (eixo X) | U08 | 256 | 16 | RPM = byte × 100 |
| `mapBins1` (eixo Y) | U08 | 272 | 16 | carga, resolução por `ignLoadRes` |

**Como alterar sem criar armadilha (importante):** a Speeduino interpola
bilinearmente entre as 4 células em volta, `entregue = Σ wᵢ·Cᵢ` com `Σ wᵢ = 1`.
Somar Δ **nas 4 células de uma vez** faz o valor entregue subir exatamente Δ em
*qualquer* posição dentro daquele quadrado — o que se testou é o que se tem,
mesmo se o motor derivar. Mexer só na célula mais próxima (o "follow mode" do
TS) dá só `w·Δ` naquele ponto, então se acaba enfiando muito mais grau na célula
pra sentir o efeito — e aí, se o motor andar pra cima dela, pega tudo, um valor
que nunca foi testado. É o mesmo motivo pelo qual o usuário, no TS, sai do
follow e uniformiza os vizinhos na mão.

Ordem segura pra construir isso: ler a página 3 e conferir contra o TS →
escrever com **motor desligado** e ler de volta comparando → só então ao vivo.

## Ambiente de build (esta máquina é o local de build)

Precisa de JDK **com** `javac` (a instalação padrão aqui era só JRE) e do
Android SDK — ambos já instalados:

```bash
cd /var/rcaldas/obd2/lambda_android
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 ANDROID_HOME=/var/rcaldas/android-sdk ./gradlew assembleDebug
```
APK em `app/build/outputs/apk/debug/app-debug.apk` (versionado no git, ao
contrário de `local.properties`, que é ignorado). Se o Gradle reclamar de
`does not provide the required capabilities: [JAVA_COMPILER]` depois de instalar
o JDK, é cache do daemon: `./gradlew --stop` e roda de novo.

## Restrições da multimídia

`minSdk 21` (Android 5.0) e o rádio original do carro é antigo mesmo. Na
prática: **nada de emoji fora do BMP** em texto de UI (sai quadradinho com a
fonte velha) — os rótulos usam só glifos tipo `λ`, `⚙`, `☰`, `°`. Tracejado
(`DashPathEffect`) com camada de hardware pode sair sólido em Android antigo;
não quebra nada, as curvas se distinguem por cor e espessura.

## Logs

`.msl` (formato TunerStudio), **start/stop manual** por um botão em
Configurações — **nada grava sozinho**. O logger CSV automático antigo foi
removido (rodava em toda conexão do ELM327, escrevia 4 colunas sempre vazias e
fazia I/O síncrono na thread de UI). O `.msl` tem `fsync` periódico e um
`BroadcastReceiver` de `ACTION_SHUTDOWN` pra sobreviver a corte de energia, e um
indicador "REC" nas telas.

Colunas incluem `Advance _Current` (Speeduino), `Advance_OBD2` (original) e
`Ethanol` — esta última porque **o limite de detonação anda junto com o teor de
álcool do tanque**, então um ponto medido só significa alguma coisa etiquetado
com o etanol daquele momento.

## Arquivos

`MainActivity.java` (orquestra telas/threads/polling), `KnockWatch.java`,
`IgnitionChartView.java`, `LambdaChartView.java`, `DashboardView.java`,
`Elm327Manager.java`, `SpeeduinoManager.java`, `MslLogger.java`,
`ObdPid.java` (tabela de PIDs decodificáveis), `DeviceRoleManager.java` (qual
USB é o quê), `UsbSerialSession.java`, `GpsSpeedProvider.java`,
`AlertManager.java`.

## Fora deste repo

`~/obd2` é este projeto. O app de manutenções (`car`) é outro repo com
`CLAUDE.md` próprio; se e como os dados de OBD2 entram lá nunca foi
reconfirmado — não assuma um plano de integração sem checar.
