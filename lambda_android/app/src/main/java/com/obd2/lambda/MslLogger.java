package com.obd2.lambda;

import android.content.Context;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Grava um log combinado, no formato .msl do TunerStudio, misturando duas
 * fontes que atualizam em ritmos independentes: lambda banco 1/2 (só o OBD2
 * tem, já que as sondas estão ligadas na injeção original, não na Speeduino)
 * e RPM/MAP/TPS/CLT/IAT/baro (pressão + correção)/VE em uso/GammaE/lambda
 * alvo/etanol "de verdade" (da Speeduino, que são os eixos e os valores que
 * os mapas dela usam) — pensado pra corrigir tabela de VE offline. Baro
 * importa porque pressão barométrica muda a mistura tanto quanto altitude/
 * clima mudam de um dia pro outro — sem isso no log, uma diferença de
 * mistura por causa do tempo vira "erro de VE" por engano. Lambda alvo, não
 * AFR: AFR é relativo ao
 * combustível (14,7:1 gasolina, ~9:1 etanol — não comparável direto entre
 * tanques diferentes), lambda já normaliza isso. GammaE separa erro de VE de
 * enriquecimento temporário tipo warmup/aceleração; "VE _Current" é a VE
 * realmente usada no cálculo do PW, não a leitura crua da tabela (ve1Pct/
 * ve2Pct) — é o que interessa pra corrigir. Sem ponto de ignição aqui — ver
 * MainActivity/KnockWatch/IgnitionChartView pra isso, que é ao vivo, não por
 * log (o recuo da original só faz sentido observado na hora, na condição
 * real; analisar depois sem lembrar exatamente das condições de pista é o
 * próprio problema que aquela tela existe pra evitar).
 *
 * Um HandlerThread próprio grava uma linha a cada 100ms (10Hz), sempre com
 * o último valor conhecido de cada fonte — os campos abaixo são atualizados
 * pelos loops de poll do ELM327 e da Speeduino (threads diferentes), por
 * isso `volatile`; não precisa de lock porque cada um só escreve o próprio
 * grupo de campos e o gravador só lê.
 *
 * Enquanto grava, o ELM327 fica dedicado a lambda banco 1/2 em qualquer tela
 * (ver MainActivity#pollRunnable) — é a única fonte deles, e esse log
 * precisa ao vivo, sem depender de qual tela ficou aberta.
 *
 * Formato do arquivo (conferido byte a byte contra um .msl real do
 * TunerStudio deste carro): 5 linhas de cabeçalho (assinatura, data+autor,
 * "#", nomes, unidades), colunas separadas por tab, fim de linha LF.
 */
public class MslLogger {

    private static final String TAG = "MSL_LOGGER";
    private static final long TICK_MS = 100; // 10Hz
    // A cada quantas linhas força os dados até o armazenamento físico
    // (fsync) — não a cada linha, tem custo real; a cada ~1s já limita bem
    // a perda num desligamento abrupto sem gastar demais.
    private static final int SYNC_EVERY_N_ROWS = 10;

    // RPMdot/MAPdot/TPSdot ficam no FIM da lista, depois de Ethanol —
    // proposital: mantém a posição das colunas já existentes intacta pra
    // qualquer análise que já leia log antigo por nome de coluna (não por
    // posição) continuar funcionando igual em cima de logs novos.
    private static final String[] COLUMN_NAMES = {
            "Time", "RPM", "MAP", "TPS", "CLT", "IAT", "Baro Pressure", "Baro Correction",
            "VE _Current", "GammaE", "Lambda Target", "Lambda", "Lambda2", "Ethanol",
            "RPMdot", "MAPdot", "TPSdot",
    };
    private static final String[] COLUMN_UNITS = {
            "s", "rpm", "kpa", "%", "", "", "kpa", "%",
            "%", "%", "O2", "O2", "O2", "%",
            "rpm/s", "kpa/s", "%/s",
    };

    private final Context context;
    private HandlerThread thread;
    private Handler handler;
    private FileOutputStream fileOutputStream;
    private BufferedWriter writer;
    private long startTimeMillis;
    private int rowsSinceSync = 0;
    private volatile boolean recording = false;

    // Última leitura conhecida de cada fonte — ver comentário da classe.
    private volatile Float lambda1;
    private volatile Float lambda2;
    private volatile SpeeduinoManager.SpeeduinoData speeduinoData;

    private final Runnable tickRunnable = new Runnable() {
        @Override
        public void run() {
            if (!recording) return;
            writeRow();
            handler.postDelayed(this, TICK_MS);
        }
    };

    public MslLogger(Context context) {
        this.context = context.getApplicationContext();
    }

    /** Chamado pelo loop de poll do ELM327 sempre que lê lambda — enquanto
     * grava, isso acontece toda volta do loop não importa a tela ativa (ver
     * MainActivity#pollRunnable), já que é a única fonte que existe. */
    public void updateObd2Lambda(Float bank1, Float bank2) {
        lambda1 = bank1;
        lambda2 = bank2;
    }

    /** Chamado pelo loop de poll da Speeduino a cada leitura. */
    public void updateSpeeduino(SpeeduinoManager.SpeeduinoData data) {
        speeduinoData = data;
    }

    public boolean isRecording() {
        return recording;
    }

    /** Inicia uma gravação nova, cria o arquivo e escreve o cabeçalho.
     * @return nome do arquivo criado (pra mostrar na UI) */
    public String start() throws IOException {
        File dir = new File(context.getExternalFilesDir(null), "logs");
        if (!dir.exists()) dir.mkdirs();

        String filename = new SimpleDateFormat("yyyy-MM-dd_HH.mm.ss", Locale.US).format(new Date()) + ".msl";
        File file = new File(dir, filename);
        fileOutputStream = new FileOutputStream(file);
        writer = new BufferedWriter(new OutputStreamWriter(fileOutputStream));

        writer.write("speeduino 202501: Speeduino 2025.01.7\n");
        writer.write("Capture Date: " + new Date() + ", File author: lambda_android\n");
        writer.write("#\n");
        writer.write(joinTab(COLUMN_NAMES) + "\n");
        writer.write(joinTab(COLUMN_UNITS) + "\n");
        writer.flush();
        syncQuietly();

        startTimeMillis = System.currentTimeMillis();
        rowsSinceSync = 0;
        recording = true;

        thread = new HandlerThread("MslLogger");
        thread.start();
        handler = new Handler(thread.getLooper());
        handler.post(tickRunnable);

        Log.i(TAG, "Gravando log .msl em: " + file.getAbsolutePath());
        return filename;
    }

    public void stop() {
        recording = false;
        if (thread != null) {
            thread.quitSafely();
            thread = null;
        }
        if (writer != null) {
            try {
                writer.flush();
                syncQuietly();
                writer.close();
            } catch (IOException e) {
                Log.w(TAG, "Erro ao fechar log .msl: " + e.getMessage());
            }
            writer = null;
            fileOutputStream = null;
        }
    }

    /** Força os dados até o armazenamento físico — writer.flush() só tira o
     * dado do buffer do Java, não garante que o SO já gravou no flash.
     * Best-effort: alguns sistemas de arquivo/dispositivos não suportam
     * sync (SyncFailedException) — nesse caso não há mais nada a fazer por
     * software, só seguir. */
    private void syncQuietly() {
        if (fileOutputStream == null) return;
        try {
            fileOutputStream.getFD().sync();
        } catch (IOException e) {
            Log.w(TAG, "Sync do log .msl falhou (ignorado): " + e.getMessage());
        }
    }

    // String.join só existe a partir da API 26 — este app roda em Android 5.0
    // (API 21), então junta na mão.
    private static String joinTab(String[] parts) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < parts.length; i++) {
            if (i > 0) sb.append('\t');
            sb.append(parts[i]);
        }
        return sb.toString();
    }

    private void writeRow() {
        if (writer == null) return;

        // Snapshot local — speeduinoData pode ser trocado por outra thread
        // entre uma leitura de campo e outra; um snapshot evita misturar
        // campos de dois instantes diferentes dentro da mesma linha.
        SpeeduinoManager.SpeeduinoData sd = speeduinoData;
        double t = (System.currentTimeMillis() - startTimeMillis) / 1000.0;

        String line = String.format(Locale.US, "%.3f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s",
                t,
                sd != null && sd.rpm != null ? sd.rpm.toString() : "",
                sd != null && sd.mapKpa != null ? String.valueOf(Math.round(sd.mapKpa)) : "",
                sd != null && sd.tpsPct != null ? String.format(Locale.US, "%.1f", sd.tpsPct) : "",
                sd != null && sd.coolantC != null ? String.valueOf(Math.round(sd.coolantC)) : "",
                sd != null && sd.iatC != null ? String.valueOf(Math.round(sd.iatC)) : "",
                sd != null && sd.baroKpa != null ? String.valueOf(Math.round(sd.baroKpa)) : "",
                sd != null && sd.baroCorrectionPct != null ? sd.baroCorrectionPct.toString() : "",
                sd != null && sd.veCurr != null ? sd.veCurr.toString() : "",
                sd != null && sd.gammaE != null ? sd.gammaE.toString() : "",
                sd != null && sd.lambdaTarget != null ? String.format(Locale.US, "%.3f", sd.lambdaTarget) : "",
                lambda1 != null ? String.format(Locale.US, "%.3f", lambda1) : "",
                lambda2 != null ? String.format(Locale.US, "%.3f", lambda2) : "",
                // O limite de detonação anda junto com o teor de álcool do
                // tanque — mas isso é a tabela de VE, não a de ponto (que não
                // entra aqui); etanol continua útil pra separar tank a tanque.
                sd != null && sd.ethanolPct != null ? sd.ethanolPct.toString() : "",
                sd != null && sd.rpmDot != null ? sd.rpmDot.toString() : "",
                sd != null && sd.mapDot != null ? sd.mapDot.toString() : "",
                sd != null && sd.tpsDot != null ? sd.tpsDot.toString() : ""
        );

        try {
            writer.write(line);
            writer.write("\n");
            writer.flush();
            if (++rowsSinceSync >= SYNC_EVERY_N_ROWS) {
                rowsSinceSync = 0;
                syncQuietly();
            }
        } catch (IOException e) {
            Log.w(TAG, "Erro ao escrever linha do log .msl: " + e.getMessage());
        }
    }
}
