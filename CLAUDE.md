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

**O log é hoje pra corrigir VE, não ponto** (ponto é ao vivo — ver a tela
acima; o log já teve `Advance _Current`/`Advance_OBD2`, foram removidas).
Colunas atuais, todas da Speeduino exceto Lambda/Lambda2 (que só o ELM327 tem,
sondas na injeção original):

`Time, RPM, MAP, TPS, CLT, IAT, Baro Pressure, Baro Correction, VE _Current,
GammaE, Lambda Target, Lambda, Lambda2, Ethanol, RPMdot, MAPdot, TPSdot, DFCO,
Engine Status, Accel Enrich`

- **VE _Current**, não VE1/VE2: é a VE realmente usada no cálculo do PW, as
  outras são só a leitura crua da tabela.
- **Lambda Target**, não AFR: AFR não é comparável entre etanol/gasolina;
  calculado como `afrTarget / stoich`, e `stoich` é config da tune (não sai no
  bloco ao vivo) — lido uma vez por conexão via comando de leitura de página
  `'p'` (`SpeeduinoManager.readStoich()`, página 1 offset 50).
- **DFCO** (byte 1 bit 4), **Engine Status** (byte 2 cru — running/crank/ASE/
  warmup/AE por TPS/enleanment de desaceleração/AE por MAP/enleanment por MAP)
  e **Accel Enrich** (byte 16, %): decodificadas pra filtrar corte/transitório
  com precisão, sem depender de olhar o GammaE cru (que carrega correções
  legítimas de flex/IAT/CLT junto, então nem sempre fica perto de 100 — ver
  seção de VE abaixo).
- **MAPdot vem sempre zero** — o firmware só calcula quando `aeMode = MAP` na
  tune; este carro usa `aeMode = TPS`. Não é bug, é peso morto nesta config.

## Acerto de tabela VE a partir do log — ferramenta e achados

Objetivo: usar os logs pra corrigir a tabela VE (motor tunado com 35% de
etanol, tabela original não reflete isso direito). Fica em `filter_log/`
(ignorado no git — são dados de teste, não código; ver `.gitignore`).

**`filter_log/ve_filter.py`** — filtra um `.msl` deixando só os momentos
utilizáveis, com **peso contínuo (0-1) por amostra**, não corte binário: o que
estraga uma leitura não é estar variando agora, é ter variado há pouco e a
sonda ainda não ter alcançado.

- Descarte duro: DFCO, AE/enleanment ativo (via `Engine Status`/`DFCO`; em log
  antigo sem essas colunas, cai pro fallback `GammaE == 0` — mais fraco, só
  pega corte, não AE/enleanment isolado), ASE/warmup, motor frio (CLT < 70°C),
  lambda fora de 0,6-1,6.
- Janela de assentamento depois de qualquer um desses (1,5s + rampa de 1,5s) —
  sem isso sobra "eco" do transitório mesmo com a flag já desarmada.
- Atraso da sonda **variável**, não fixo: escala com o inverso do fluxo
  (rotação × MAP), ancorado em 0,8s a 1800rpm/45kPa (valor que
  `ve_map_optimizer.py` já estimava fixo pelo DFCO).
- Peso cai com |RPMdot| e com variação do MAP numa janela de 1s.
- Roda com `--tune CurrentTune.msq` pra cobertura por célula nos bins reais e
  aviso de bins finos demais (lê a resolução do sensor MAP da própria tune —
  `mapMin`/`mapMax` da calibração, não hardcoded, funciona pra qualquer
  sensor/carro); sem `--tune`, ainda sai distribuição e sugestão de bins.
- Gera um `.msl` novo só com as amostras aprovadas + coluna `FilterWeight`,
  pra abrir no MegaLogViewer.
- **Limiar padrão: peso ≥ 0,3.** Achado medindo dispersão do erro de lambda
  dentro de cada célula nos 3 logs — cai pela metade de 0 pra 0,3 e depois
  não melhora mais, só perde célula. `--min-peso` ajusta.

**Achados, todos verificados no dado real (4 logs, ~124 mil amostras
injetando):**

1. **MAP vs. baro discordam ~8-10 kPa em repouso** (MAP lê ~90 e poucos, baro
   100). É o sensor MPX5700A (7 bar) dentro do próprio spec (±2,5% do fundo de
   escala = ±17,5 kPa) — **não é erro de calibração, `mapMin=-31`/`mapMax=746`
   é a calibração correta pro sensor físico**. Recalibrar mentiria a leitura.
   Sem ação por enquanto, mas guardar: **qualquer correção baseada no baro
   neste carro está parcialmente corrigindo discordância entre sensores, não
   só altitude** — diferente do outro carro do usuário, onde MAP e baro batem
   exatos parados e por isso a multiplicação por MAP já corrige sozinha sem
   precisar de `Baro Correction`. Critério do usuário pra ativar essa tabela
   aqui: só quando a diferença for inconstestável (mesma carga/condição,
   baro diferente, nada mais explica).
2. **Resolução do MAP**: uniforme, ~0,76 kPa/contagem de ADC (10 bits em 777
   kPa de faixa). Sem lacuna na leitura nem piso de resolução diferente —
   degrada em vácuo alto só por ser fração maior da leitura (5% a 15kPa vs.
   1% a 80kPa), não por limitação física adicional.
3. **Bins de carga**: tinha 3 pares a 2 kPa de distância (72/74, 78/80) — menos
   que a resolução do sensor (~2,6 contagens), não resolvíveis. E o primeiro
   bin (22) ficava acima de onde o carro roda de verdade: com o filtro
   excluindo corte corretamente, ~14% de tudo que injeta fica ≤21 kPa (descida
   com TPS de 3-5% segurando o carro, não marcha lenta — marcha lenta deste
   carro fica em 26-29 kPa). Primeiro bin recalibrado pelo usuário pro menor
   MAP médio que ainda injeta na maioria das vezes.
4. **Viés subida × descida é real, mas é transitório, não estrutural** —
   mistura fica mais pobre subindo que descendo na MESMA célula RPM×MAP,
   consistente nos 3 logs (+0,4 a +1,4pp). Não é IAT (ΔIAT subida-descida
   medido em +0,07°C, irrelevante). Encolhe quando o filtro aperta → é filme
   de combustível na parede do coletor/porta (pior com etanol, calor de
   vaporização maior), o mesmo motivo de existir AE/enleanment. **Implicação:
   não deve virar valor de VE** — entra na tabela só o valor das amostras mais
   estáveis; a diferença residual é papel do AE/enleanment, não da VE.
   Pendente: separar "ladeira real" de "MAP subindo" (são a mesma coisa nesse
   teste) exigiria altitude real (GPS) com MAP constante — não fechado, só
   citado se um dia quiser confirmar isso à parte.
5. **`decelAmount = 78%`** na tune explica os GammaE≈77 vistos no log
   (enleanment de desaceleração, não enriquecimento). Critério do usuário:
   GammaE fora de ~95-105 não é descarte automático por si só — ele carrega
   correções legítimas (flex/IAT/CLT) que costumam estar bem calibradas; o
   descarte certo é pelas flags de transitório (item acima), não por faixa de
   GammaE.

**Descartado por ora**: rede neural pra classificar "trecho bom" (não há
rótulo independente pra aprender — o filtro de regras/peso é o método;
ML já é usado no estágio de *correção* em `ve_map_optimizer.py`, que é onde
faz sentido). GPS pra qualificar carga (MAP com mesma rotação já é a carga —
redundante); GPS só teria uso se um dia quiser desconfundir ladeira de MAP
subindo (item 4 acima), não como substituto de carga.

**Pendente, próximo passo natural**: pausar/retomar gravação da tela
principal (não só start/stop em Configurações) — deixa o usuário pausar em
trânsito/semáforo/oscilação e eliminar lixo na origem, sem precisar do
filtro pra isso depois.

## Arquivos

`MainActivity.java` (orquestra telas/threads/polling), `KnockWatch.java`,
`IgnitionChartView.java`, `LambdaChartView.java`, `DashboardView.java`,
`Elm327Manager.java`, `SpeeduinoManager.java`, `MslLogger.java`,
`ObdPid.java` (tabela de PIDs decodificáveis), `DeviceRoleManager.java` (qual
USB é o quê), `UsbSerialSession.java`, `GpsSpeedProvider.java`,
`AlertManager.java`.

## Porte web (web/) — WebSerial + Docker

Mesmo app, rodando no navegador (Chrome/Edge, precisa de contexto seguro —
`http://localhost` conta, IP de rede não) em vez de instalado no celular.
Container só serve estático (`nginx:alpine`, sem build step — ES modules
direto, `<script type="module">`); a comunicação serial acontece no próprio
navegador via `navigator.serial` (WebSerial), não no container — Docker aqui
é só o servidor de arquivos, USB nunca entra nele.

Portado até agora: telas de lambda e dashboard, alertas de tensão/água,
log .msl. Arquitetura por arquivo, cada um porte direto do equivalente
Android:

- `serial.js` — porte de `UsbSerialSession`, mas com uma diferença
  deliberada: um único loop de leitura (`_pump`) persistente por trás de um
  buffer de bytes compartilhado, em vez de repetir `reader.read()` a cada
  timeout (que correria risco de uma leitura abandonada roubar o próximo
  pedaço de dado — a WebSerial não tem `read(buf, timeoutMs)` bloqueante
  como o Android). Serve tanto o protocolo texto do ELM327 (`readUntil`)
  quanto o binário da Speeduino (`byteLength/peekBytes/takeBytes/waitForBytes`).
- `elm327.js` / `speeduino.js` — porte de `Elm327Manager`/`SpeeduinoManager`,
  mesmos PIDs, offsets, escalas e CRC32 (conferido contra `java.util.zip.CRC32`
  e `zlib.crc32` do Python).
- `dashboard.js` — porte de `DashboardView` pra Canvas 2D (mapeamento quase
  1:1: `Paint`→`ctx.fillStyle`, `drawArc`→`ctx.arc`, `drawRoundRect`→
  `ctx.roundRect`, nativo desde Chrome 99 — seguro dado que o app já exige
  Chromium pela WebSerial).
- `alerts.js` — porte de `AlertManager`, só os dois alertas fixos (tensão
  baixa, água quente); os alertas personalizados por PID (`CustomAlertRule`,
  com tela própria de cadastro no Android) ficaram de fora — o web app ainda
  não tem tela de configurações.
- `mslLogger.js` — porte de `MslLogger`, mesmas colunas/ordem/motivos (ver
  `## Logs` acima). Sem `java.io` aqui: usa a File System Access API
  (`showSaveFilePicker` + stream gravável), Chromium-only como o resto.

`app.js` orquestra os dois loops de poll independentes (mesma razão do
Android: a porta do ELM327 é uma só, cada PID a mais nela derruba a taxa de
todos; a Speeduino tem porta própria, então seu loop roda sempre, não só na
tela de dashboard) e a troca de tela — que só existe como estado de UI, sem
efeito na exclusividade de porta serial (isso já é garantido pelo SO, do
mesmo jeito nas duas plataformas).

Fora de escopo por enquanto: tela de ponto/`KnockWatch` (é ao vivo por
natureza — analisar depois sem lembrar da condição de pista é o problema que
ela existe pra evitar; não portada), GPS (velocidade vem só do PID 010D),
tela de configurações/alertas personalizados.

## Fora deste repo

`~/obd2` é este projeto. O app de manutenções (`car`) é outro repo com
`CLAUDE.md` próprio; se e como os dados de OBD2 entram lá nunca foi
reconfirmado — não assuma um plano de integração sem checar.
