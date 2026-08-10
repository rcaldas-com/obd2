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
        public Integer ve1Pct;
        public Integer ve2Pct;
        public Float afrTarget;
        public Float advanceDeg;      // ponto de ignição REAL aplicado pela Speeduino (signed)
        public Float tpsPct;
        public Float baroKpa;
        public Integer ethanolPct;    // sensor flex
        public Float pw1Ms;
        public Float pw2Ms;
        public Float dwellMs;
        public Float advance1Deg;
        public Float advance2Deg;
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
        data.ve1Pct = u8(block, 19);
        data.ve2Pct = u8(block, 20);
        data.afrTarget = u8(block, 21) * 0.1f;
        data.advanceDeg = (float) s8(block, 24);
        data.tpsPct = u8(block, 25) * 0.5f;
        data.ethanolPct = u8(block, 35);
        data.baroKpa = (float) u8(block, 41);
        data.pw1Ms = u16le(block, 76) * 0.001f;
        data.pw2Ms = u16le(block, 78) * 0.001f;
        data.dwellMs = u16le(block, 90) * 0.001f;
        data.advance1Deg = (float) s8(block, 118);
        data.advance2Deg = (float) s8(block, 119);
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
