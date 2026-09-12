package com.obd2.lambda;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * Vigia o ponto da ECU original (PID 010E) procurando o recuo que ela aplica
 * quando o sensor de detonação dela acusa — enquanto o ponto que o motor
 * realmente usa (Speeduino) é subido à mão numa condição de rodagem estável.
 *
 * O valor bruto do 010E sozinho não diz nada: ele muda com rotação, carga,
 * temperatura e as correções internas da original. O que tem significado é
 * quanto ela recuou em relação ao que ela mesma estava comandando naquela
 * MESMA condição segundos atrás. Daí o desenho:
 *
 *  1. Só considera a condição "estável" quando rotação/MAP/TPS ficam dentro de
 *     uma faixa estreita por {@link #STABLE_WINDOW_MS} seguidos (rotação, MAP e
 *     TPS vêm da Speeduino, que é lida em porta/thread própria a ~20Hz, então
 *     medir estabilidade não custa banda do ELM327).
 *  2. Estabilizou: fixa uma referência com a mediana do ponto da original na
 *     janela. Mediana, não média, pra um único valor esquisito de leitura não
 *     envenenar a referência.
 *  3. Estável e com referência: recuo = referência − ponto atual. Passou de
 *     {@link #EVENT_THRESHOLD_DEG} por algumas amostras seguidas, é evento.
 *  4. Saiu da faixa de estabilidade: solta a referência. Uma queda de ponto com
 *     a condição mudando é explicável pela própria mudança — não serve de
 *     evidência de detonação.
 *
 * As tolerâncias são propositalmente folgadas: subir ponto numa condição
 * estável aumenta um pouco o torque, então rotação/MAP mexem um pouco *por
 * causa do próprio ajuste sendo testado*. Tolerância apertada demais soltaria a
 * referência exatamente no momento do teste.
 */
public class KnockWatch {

    // Faixa que ainda conta como "mesma condição" (ver comentário da classe).
    private static final int RPM_TOLERANCE = 150;
    private static final float MAP_TOLERANCE_KPA = 8f;
    private static final float TPS_TOLERANCE_PCT = 3f;

    private static final long STABLE_WINDOW_MS = 4000;

    /** Recuo (em graus abaixo da referência) que caracteriza evento. */
    private static final float EVENT_THRESHOLD_DEG = 2.0f;
    /** Amostras seguidas acima do limiar pra confirmar — filtra leitura solta. */
    private static final int EVENT_CONFIRM_SAMPLES = 2;
    /** Histerese pra sair do evento, no mesmo espírito do AlertManager. */
    private static final float EVENT_CLEAR_DEG = 1.0f;

    public enum State {
        /** Sem dados suficientes (sem ponto da original, ou sem Speeduino). */
        SEM_DADOS,
        /** Condição mudando — referência solta de propósito. */
        INSTAVEL,
        /** Condição estável, referência fixada, sem recuo relevante. */
        ESTAVEL,
        /** Recuo confirmado abaixo da referência com a condição estável. */
        RECUO
    }

    public static class Status {
        public State state = State.SEM_DADOS;
        /** Referência de ponto da original, quando fixada. */
        public Float reference;
        /** referência − ponto atual (positivo = recuou). Null sem referência. */
        public Float dropDeg;
        /** true só na transição para RECUO — para disparar o aviso uma vez. */
        public boolean eventJustFired;
    }

    private static class Sample {
        final long t;
        final float advance;
        final int rpm;
        final float map;
        final float tps;

        Sample(long t, float advance, int rpm, float map, float tps) {
            this.t = t;
            this.advance = advance;
            this.rpm = rpm;
            this.map = map;
            this.tps = tps;
        }
    }

    private final Deque<Sample> window = new ArrayDeque<>();
    private Float reference;
    private int consecutiveOverThreshold = 0;
    private boolean eventActive = false;
    private boolean anchorLocked = false;

    private final Status status = new Status();

    /**
     * Alimenta uma leitura. `advance` é o ponto da ECU original; rotação/MAP/TPS
     * vêm da Speeduino. Devolve o estado já recalculado (mesma instância
     * reaproveitada — copie o que precisar guardar).
     */
    public Status update(long now, Float advance, Integer rpm, Float map, Float tps) {
        status.eventJustFired = false;

        if (advance == null || rpm == null || map == null || tps == null) {
            status.state = State.SEM_DADOS;
            status.reference = reference;
            status.dropDeg = null;
            return status;
        }

        window.addLast(new Sample(now, advance, rpm, map, tps));
        while (!window.isEmpty() && now - window.peekFirst().t > STABLE_WINDOW_MS) {
            window.removeFirst();
        }

        boolean stable = isWindowStable(now);

        if (!stable) {
            // Referência fixada à mão sobrevive à instabilidade — quem fixou
            // sabe o que está fazendo; a automática é solta.
            if (!anchorLocked) {
                reference = null;
                eventActive = false;
                consecutiveOverThreshold = 0;
            }
            status.state = anchorLocked && reference != null ? evaluateDrop(advance) : State.INSTAVEL;
            status.reference = reference;
            status.dropDeg = reference != null ? reference - advance : null;
            return status;
        }

        if (reference == null) {
            reference = medianAdvance();
        } else if (!eventActive) {
            // Deixa a referência subir junto se a original resolveu adiantar
            // sozinha (recuperação de adaptativa, IAT caindo), mas nunca descer:
            // se descesse, o próprio recuo que estamos caçando seria absorvido
            // pela referência e o evento sumiria.
            float recent = medianAdvance();
            if (recent > reference) reference = recent;
        }

        status.state = evaluateDrop(advance);
        status.reference = reference;
        status.dropDeg = reference - advance;
        return status;
    }

    /** Avalia recuo contra a referência já fixada, com confirmação e histerese. */
    private State evaluateDrop(float advance) {
        float drop = reference - advance;

        if (eventActive) {
            if (drop < EVENT_CLEAR_DEG) {
                eventActive = false;
                consecutiveOverThreshold = 0;
                return State.ESTAVEL;
            }
            return State.RECUO;
        }

        if (drop >= EVENT_THRESHOLD_DEG) {
            if (++consecutiveOverThreshold >= EVENT_CONFIRM_SAMPLES) {
                eventActive = true;
                status.eventJustFired = true;
                return State.RECUO;
            }
        } else {
            consecutiveOverThreshold = 0;
        }
        return State.ESTAVEL;
    }

    private boolean isWindowStable(long now) {
        if (window.size() < 3) return false;
        if (now - window.peekFirst().t < STABLE_WINDOW_MS) return false;

        int rpmMin = Integer.MAX_VALUE, rpmMax = Integer.MIN_VALUE;
        float mapMin = Float.MAX_VALUE, mapMax = -Float.MAX_VALUE;
        float tpsMin = Float.MAX_VALUE, tpsMax = -Float.MAX_VALUE;

        for (Sample s : window) {
            rpmMin = Math.min(rpmMin, s.rpm);
            rpmMax = Math.max(rpmMax, s.rpm);
            mapMin = Math.min(mapMin, s.map);
            mapMax = Math.max(mapMax, s.map);
            tpsMin = Math.min(tpsMin, s.tps);
            tpsMax = Math.max(tpsMax, s.tps);
        }

        return (rpmMax - rpmMin) <= RPM_TOLERANCE
                && (mapMax - mapMin) <= MAP_TOLERANCE_KPA
                && (tpsMax - tpsMin) <= TPS_TOLERANCE_PCT;
    }

    private float medianAdvance() {
        List<Float> values = new ArrayList<>(window.size());
        for (Sample s : window) values.add(s.advance);
        java.util.Collections.sort(values);
        int n = values.size();
        return n % 2 == 1
                ? values.get(n / 2)
                : (values.get(n / 2 - 1) + values.get(n / 2)) / 2f;
    }

    /**
     * Botão REF: trava a referência no valor atual (mesmo sem os
     * {@link #STABLE_WINDOW_MS} completos — quem apertou está dizendo "a
     * condição é esta"), e travada, destrava e devolve ao automático. Travada,
     * ela sobrevive à condição sair da faixa de estabilidade, que é o que
     * permite subir ponto observando o mesmo ponto de partida mesmo se a
     * estrada oscilar um pouco.
     */
    public void toggleAnchor() {
        if (anchorLocked) {
            anchorLocked = false;
            reference = null;
            eventActive = false;
            consecutiveOverThreshold = 0;
        } else if (!window.isEmpty()) {
            reference = medianAdvance();
            anchorLocked = true;
            eventActive = false;
            consecutiveOverThreshold = 0;
        }
    }

    public boolean isAnchorLocked() {
        return anchorLocked;
    }

    public void reset() {
        window.clear();
        reference = null;
        anchorLocked = false;
        eventActive = false;
        consecutiveOverThreshold = 0;
        status.state = State.SEM_DADOS;
        status.reference = null;
        status.dropDeg = null;
        status.eventJustFired = false;
    }
}
