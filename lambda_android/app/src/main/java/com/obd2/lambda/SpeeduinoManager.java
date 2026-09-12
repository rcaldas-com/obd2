package com.obd2.lambda;

import android.hardware.usb.UsbManager;
import android.util.Log;

import com.hoho.android.usbserial.driver.UsbSerialDriver;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.zip.CRC32;

/**
 * Gerencia comunicação com uma ECU Speeduino via USB Serial, usando o mesmo
 * protocolo binário com CRC que o TunerStudio usa (confirmado lendo o
 * firmware-fonte real, tag 202501.7, que é exatamente a versão deste carro:
 * speeduino/speeduino/speeduino/comms.cpp e logger.cpp).
 *
 * Envelope de cada requisição/resposta (comms.cpp: serialWrite(uint16_t),
 * sendBufferAndCrcNonBlocking, reverse_bytes):
 *   [tamanho do payload, 2 bytes big-endian]
 *   [payload]
 *   [CRC32 do payload, 4 bytes big-endian — CRC32 padrão (poly 0x04C11DB7,
 *    refin/refout, mesmo algoritmo do java.util.zip.CRC32)]
 *
 * Comando de dados ao vivo ("output channels"), payload (comms.cpp case 'r',
 * offset/length lidos via word(hi,lo) = little-endian):
 *   ['r', tsCanId, 0x30, offsetLo, offsetHi, lengthLo, lengthHi]
 * Resposta: [SERIAL_RC_OK(0x00)] + os `length` bytes pedidos do bloco de
 * status (130 bytes no total, logger.cpp:getTSLogEntry — offsets abaixo
 * conferidos um a um contra esse arquivo e contra os scalars do próprio
 * .ini deste carro).
 *
 * Somente leitura — nenhum comando de escrita/página/tabela é usado aqui.
 */
public class SpeeduinoManager {

    private static final String TAG = "SPEEDUINO";
    private static final int BAUD_RATE = 115200;
    private static final int WRITE_TIMEOUT_MS = 2000;
    private static final int READ_TIMEOUT_MS = 500;

    private static final int OCH_TABLE_ID = 0x30; // SEND_OUTPUT_CHANNELS
    private static final int OCH_BLOCK_SIZE = 130; // logger.h: LOG_ENTRY_SIZE (confere com o .ini: ochBlockSize)
    // tsCanId: 0 = sem CAN, conexão serial direta (é o caso deste carro —
    // "speeduino_tsCanId" na tune não usa CAN passthrough).
    private static final int TS_CAN_ID = 0;

    private static final byte SERIAL_RC_OK = 0x00;
    private static final String EXPECTED_SIGNATURE = "speeduino 202501";

    private final UsbSerialSession session = new UsbSerialSession();
    private boolean connected = false;
    // Relação estequiométrica configurada na tune (página 1, offset 50) —
    // não é telemetria ao vivo, é config; lida uma vez por conexão em
    // readStoich() e reaproveitada em todo readOutputChannels() daí em
    // diante pra calcular lambdaTarget = afrTarget / stoich.
    private volatile Float stoich;

    /** Um "snapshot" decodificado do bloco de status da Speeduino. Campos
     * cobrem os bytes 0-41 do bloco (mais alguns pontuais como PW/advance1-2/
     * dwell) — o suficiente pro dashboard e pro log .msl; o restante do
     * bloco (CAN inputs, VVT, etc.) não é decodificado por não ser usado. */
    public static class SpeeduinoData {
        public Integer secl;
        public Integer rpm;
        public Float mapKpa;
        public Float iatC;
        public Float coolantC;
        public Float batteryV;
        public Float afrNative;       // O2 nativo da Speeduino (sem sonda ligada nesta instalação — só referência)
        public Integer ve1Pct;         // tabela banco 1 (referência apenas — ver veCurr)
        public Integer ve2Pct;         // tabela banco 2 (referência apenas — ver veCurr)
        public Integer veCurr;        // VE realmente usada no cálculo do PW nesse instante (o que TunerStudio chama "VE (Current)")
        public Integer gammaE;         // % de correção de combustível total aplicada (warmup/AE/etc) — precisa pra separar erro de VE de enriquecimento temporário ao analisar o log
        public Float afrTarget;
        public Float lambdaTarget;    // afrTarget / stoich (stoich é config da tune, lido uma vez — ver readStoich)
        public Float advanceDeg;      // ponto de ignição REAL aplicado pela Speeduino (signed)
        public Float tpsPct;
        public Float baroKpa;
        public Integer baroCorrectionPct; // correção de mistura por pressão barométrica que a própria Speeduino já aplica
        public Integer ethanolPct;    // sensor flex
        public Float pw1Ms;
        public Float pw2Ms;
        public Float dwellMs;
        public Float advance1Deg;
        public Float advance2Deg;
        public Integer tpsDot;   // %/s — solta o acelerador de repente dá negativo
        public Integer rpmDot;   // rpm/s — rotação caindo dá negativo
        public Integer mapDot;   // kPa/s — vácuo subindo (mais negativo = MAP caindo) dá negativo
        public long timestamp;
    }

    public String connect(UsbManager usbManager, UsbSerialDriver driver) throws IOException {
        String deviceName = session.open(usbManager, driver, BAUD_RATE);
        connected = true;

        // A Speeduino roda em Mega2560, que tem o circuito clássico de
        // auto-reset via DTR (mesmo comportamento do TunerStudio ao
        // conectar — esperado). Espera o firmware terminar de subir antes
        // do primeiro handshake.
        sleep(2000);

        return deviceName;
    }

    public void disconnect() {
        connected = false;
        session.close();
    }

    public boolean isConnected() {
        return connected && session.isOpen();
    }

    private void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) {}
    }

    /** Descarta qualquer byte parado no buffer de recepção — ver
     * comentário em sendCommand() sobre por que isso importa aqui
     * (nenhum marcador de resync no protocolo, ao contrário do ELM327
     * que sempre termina com '>'). */
    private void drainBuffer() {
        byte[] buf = new byte[MAX_RESPONSE_SIZE];
        try {
            while (session.read(buf, 15) > 0) { /* drain */ }
        } catch (IOException ignored) {}
    }

    /** Handshake de conectividade: manda 'Q' e confere a assinatura
     * "speeduino 202501" na resposta. Chamar antes de confiar em
     * readOutputChannels() — como o filtro USB não distingue ELM327 de
     * Speeduino por VID/PID, isso confirma que o dispositivo atribuído
     * como "Speeduino" realmente responde como uma. */
    public boolean verifySignature() {
        if (!isConnected()) return false;
        try {
            byte[] response = sendCommand(new byte[]{'Q'});
            if (response == null || response.length < 1 + EXPECTED_SIGNATURE.length()) return false;
            if (response[0] != SERIAL_RC_OK) return false;
            String signature = new String(response, 1, EXPECTED_SIGNATURE.length(), StandardCharsets.US_ASCII);
            return EXPECTED_SIGNATURE.equals(signature);
        } catch (IOException e) {
            Log.w(TAG, "Falha no handshake 'Q': " + e.getMessage());
            return false;
        }
    }

    /**
     * Lê um byte de uma página de configuração da tune (comando 'p' —
     * mesmo envelope tamanho+payload+CRC32 do 'r', só trocando a tabela de
     * output channels por número de página; comms.cpp confirma o payload
     * idêntico: tsCanId, página, offset little-endian, tamanho
     * little-endian). Usado só pra valores fixos da tune que não saem no
     * bloco de status ao vivo, como o stoich configurado.
     */
    private Integer readPageByte(int page, int offset) {
        if (!isConnected()) return null;
        try {
            byte[] payload = new byte[]{
                    'p',
                    (byte) TS_CAN_ID,
                    (byte) page,
                    (byte) (offset & 0xFF), (byte) ((offset >> 8) & 0xFF),
                    (byte) 1, (byte) 0, // length = 1 byte
            };
            byte[] response = sendCommand(payload);
            if (response == null || response.length < 2 || response[0] != SERIAL_RC_OK) {
                return null;
            }
            return response[1] & 0xFF;
        } catch (IOException e) {
            Log.w(TAG, "Falha ao ler página " + page + " offset " + offset + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Lê o stoich configurado na tune (página 1, offset 50 — conferido
     * contra mainController.ini deste carro: "stoich = scalar, U08, 50,
     * ':1', 0.1"). Não é telemetria, é config — chamar uma vez por conexão
     * (depois de verifySignature() confirmar que é mesmo a Speeduino), não
     * a cada leitura. Se falhar, lambdaTarget fica null em vez de usar um
     * valor errado — melhor faltar a coluna que mentir nela.
     */
    public void readStoich() {
        Integer raw = readPageByte(1, 50);
        stoich = raw != null ? raw * 0.1f : null;
    }

    /** Lê e decodifica o bloco de dados ao vivo inteiro (130 bytes). */
    public SpeeduinoData readOutputChannels() {
        SpeeduinoData data = new SpeeduinoData();
        data.timestamp = System.currentTimeMillis();
        if (!isConnected()) return data;

        try {
            byte[] payload = new byte[]{
                    'r',
                    (byte) TS_CAN_ID,
                    (byte) OCH_TABLE_ID,
                    (byte) (0), (byte) (0),                                   // offset = 0, little-endian
                    (byte) (OCH_BLOCK_SIZE & 0xFF), (byte) ((OCH_BLOCK_SIZE >> 8) & 0xFF), // length, little-endian
            };
            byte[] response = sendCommand(payload);
            if (response == null || response.length < 1 + OCH_BLOCK_SIZE || response[0] != SERIAL_RC_OK) {
                return data;
            }
            parseBlock(response, data);
        } catch (IOException e) {
            Log.w(TAG, "Falha ao ler output channels: " + e.getMessage());
        }
        return data;
    }

    /** Decodifica os campos usados a partir do bloco bruto (offsets/escalas
     * conferidos contra logger.cpp:getTSLogEntry e o .ini deste carro).
     * `block[0]` é o SERIAL_RC_OK, então os offsets do bloco de status
     * (documentados a partir de 0) ficam em block[offset + 1]. */
    private void parseBlock(byte[] block, SpeeduinoData data) {
        data.secl = u8(block, 0);
        data.mapKpa = (float) u16le(block, 4);
        data.iatC = u8(block, 6) - 40f;
        data.coolantC = u8(block, 7) - 40f;
        data.batteryV = u8(block, 9) * 0.1f;
        data.afrNative = u8(block, 10) * 0.1f;
        data.rpm = u16le(block, 14);
        // offset 17 (U16, escala 1.0) — conferido contra
        // mainController.ini deste carro: "gammaEnrich = scalar, U16, 17".
        data.gammaE = u16le(block, 17);
        data.ve1Pct = u8(block, 19);
        data.ve2Pct = u8(block, 20);
        data.afrTarget = u8(block, 21) * 0.1f;
        // stoich vem de config (readStoich), não do bloco de status — sem
        // ele lido ainda (ou se a leitura falhou), fica null em vez de usar
        // um valor chutado.
        Float stoichNow = stoich;
        data.lambdaTarget = stoichNow != null ? data.afrTarget / stoichNow : null;
        // offset 22 (S16, escala 1.0) — conferido contra speeduino.ini:
        // "TPSdot = scalar, S16, 22". Solta o acelerador de repente dá
        // negativo — é o caso que importa enxergar, por isso o s16le.
        data.tpsDot = s16le(block, 22);
        data.advanceDeg = (float) s8(block, 24);
        data.tpsPct = u8(block, 25) * 0.5f;
        // offset 33 (S16, escala 1.0) — conferido contra speeduino.ini:
        // "rpmDOT = scalar, S16, 33". Rotação caindo dá negativo.
        data.rpmDot = s16le(block, 33);
        data.ethanolPct = u8(block, 35);
        data.baroKpa = (float) u8(block, 41);
        data.pw1Ms = u16le(block, 76) * 0.001f;
        data.pw2Ms = u16le(block, 78) * 0.001f;
        data.dwellMs = u16le(block, 90) * 0.001f;
        // offset 93 (S16, escala 1.0) — conferido contra speeduino.ini:
        // "MAPdot = scalar, S16, 93". Vácuo subindo (MAP caindo) dá negativo.
        data.mapDot = s16le(block, 93);
        data.advance1Deg = (float) s8(block, 118);
        data.advance2Deg = (float) s8(block, 119);
        // offset 101 (U08, escala 1.0) — conferido contra mainController.ini:
        // "baroCorrection = scalar, U08, 101". Correção de mistura por
        // pressão barométrica que a própria Speeduino já aplica — junto com
        // baroKpa (offset 41, acima) pra corrigir a VE de verdade, já que a
        // pressão muda de dia pra dia e de altitude pra altitude.
        data.baroCorrectionPct = u8(block, 101);
        // offset 102 (U08, escala 1.0) — conferido contra mainController.ini:
        // "veCurr = scalar, U08, 102". É a VE realmente usada no cálculo do
        // PW nesse instante — diferente de ve1Pct/ve2Pct, que são só a
        // leitura crua da célula da tabela.
        data.veCurr = u8(block, 102);
    }

    // offset é relativo ao bloco de status (0-based); +1 pula o SERIAL_RC_OK.
    private static int u8(byte[] block, int offset) {
        return block[offset + 1] & 0xFF;
    }

    private static int s8(byte[] block, int offset) {
        return block[offset + 1]; // já signed em Java
    }

    private static int u16le(byte[] block, int offset) {
        return u8(block, offset) | (u8(block, offset + 1) << 8);
    }

    // short com sinal — necessário pra rpmDOT/TPSdot/MAPdot, que ficam
    // negativos exatamente nos casos que importam (rotação caindo, vácuo
    // subindo, solta o acelerador de repente).
    private static int s16le(byte[] block, int offset) {
        return (short) (u8(block, offset) | (u8(block, offset + 1) << 8));
    }

    // Folga generosa pro maior pacote possível (tamanho[2] + SERIAL_RC_OK(1)
    // + bloco(130) + CRC[4]) — usado como teto de segurança pro acumulador.
    private static final int MAX_RESPONSE_SIZE = 2 + 1 + OCH_BLOCK_SIZE + 4 + 8;

    /** Monta o envelope (tamanho + payload + CRC32), envia, lê o envelope de
     * resposta e devolve só o payload da resposta (sem tamanho nem CRC).
     * Retorna null se o CRC da resposta não bater.
     *
     * Lê tudo num único buffer acumulador, sem descartar bytes — CDC-ACM
     * (USB nativo, como esta Speeduino) costuma entregar a resposta inteira
     * de uma vez num pacote só, então uma leitura pode trazer bem mais do
     * que o pedaço "atual" (ex.: os 2 bytes de tamanho + o payload inteiro +
     * o CRC, tudo junto). Ler cada pedaço num buffer descartável separado
     * (implementação anterior) jogava fora esse excedente e travava
     * esperando dados que o dispositivo já tinha mandado.
     *
     * Antes de mandar, drena qualquer byte perdido de uma troca anterior
     * incompleta (ex.: um timeout no meio de uma resposta, plausível no
     * ambiente eletricamente ruidoso de um carro com bobinas de ignição
     * disparando perto da placa) — sem isso, a sobra fica no buffer de
     * recepção e é lida como se fosse o início da PRÓXIMA resposta,
     * corrompendo tudo dali em diante até religar a porta (e às vezes nem
     * isso resolve, se o firmware do outro lado também ficou confuso). */
    private byte[] sendCommand(byte[] payload) throws IOException {
        drainBuffer();
        session.write(wrapRequest(payload), WRITE_TIMEOUT_MS);

        long deadline = System.currentTimeMillis() + READ_TIMEOUT_MS;
        byte[] acc = new byte[MAX_RESPONSE_SIZE];
        byte[] chunk = new byte[MAX_RESPONSE_SIZE];
        int have = 0;
        int totalLength = -1;

        while (totalLength < 0 || have < totalLength) {
            if (System.currentTimeMillis() > deadline) return null;
            int len = session.read(chunk, 100);
            if (len <= 0) continue;
            if (have + len > acc.length) len = acc.length - have; // não deve acontecer, só por segurança
            System.arraycopy(chunk, 0, acc, have, len);
            have += len;
            if (totalLength < 0 && have >= 2) {
                int responseLength = ((acc[0] & 0xFF) << 8) | (acc[1] & 0xFF);
                totalLength = 2 + responseLength + 4;
                if (totalLength > acc.length) {
                    Log.w(TAG, "Resposta maior que o esperado (" + totalLength + " bytes) — ignorando");
                    return null;
                }
            }
        }

        int responseLength = totalLength - 6;
        byte[] responsePayload = new byte[responseLength];
        System.arraycopy(acc, 2, responsePayload, 0, responseLength);

        long expectedCrc = ((long) (acc[2 + responseLength] & 0xFF) << 24)
                | ((acc[2 + responseLength + 1] & 0xFF) << 16)
                | ((acc[2 + responseLength + 2] & 0xFF) << 8)
                | (acc[2 + responseLength + 3] & 0xFF);
        CRC32 crc = new CRC32();
        crc.update(responsePayload);
        if (crc.getValue() != expectedCrc) {
            Log.w(TAG, "CRC da resposta não confere — ignorando pacote");
            return null;
        }
        return responsePayload;
    }

    private byte[] wrapRequest(byte[] payload) {
        CRC32 crc = new CRC32();
        crc.update(payload);
        long crcValue = crc.getValue();

        byte[] out = new byte[2 + payload.length + 4];
        out[0] = (byte) ((payload.length >> 8) & 0xFF);
        out[1] = (byte) (payload.length & 0xFF);
        System.arraycopy(payload, 0, out, 2, payload.length);
        int base = 2 + payload.length;
        out[base] = (byte) ((crcValue >> 24) & 0xFF);
        out[base + 1] = (byte) ((crcValue >> 16) & 0xFF);
        out[base + 2] = (byte) ((crcValue >> 8) & 0xFF);
        out[base + 3] = (byte) (crcValue & 0xFF);
        return out;
    }
}
