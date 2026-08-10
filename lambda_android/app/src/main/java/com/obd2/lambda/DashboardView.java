package com.obd2.lambda;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.Locale;

/**
 * View customizada para dashboard do carro.
 * Mostra RPM, temperatura da água, temperatura do ar, pressão barométrica,
 * ponto de ignição e TPS em formato de gauges.
 */
public class DashboardView extends View {

    // Dados atuais — RPM/água/ar/velocidade vêm do OBD2 (velocidade pode vir
    // do GPS do dispositivo, sobrescrita em MainActivity); ponto/MAP/baro/
    // flex/TPS vêm da Speeduino (quem realmente comanda a ignição agora e
    // cujo TPS é o que os mapas dela usam), lidos por um loop de poll
    // independente — ver updateSpeeduinoData().
    private Integer rpm;
    private Float coolantTemp;
    private Float intakeAirTemp;
    private Integer speed;
    private Float batteryVoltage;
    private Float speeduinoAdvance;
    private Float speeduinoMap;
    private Float speeduinoBaro;
    private Integer speeduinoFlexPct;
    private Float speeduinoTps;

    // Paints
    private final Paint paintBg = new Paint();
    private final Paint paintGaugeArc = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintGaugeValue = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintGaugeBg = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintValue = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintLabel = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintUnit = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintDivider = new Paint();

    // Cores
    private static final int COLOR_BG = Color.parseColor("#16213E");
    private static final int COLOR_GAUGE_BG = Color.parseColor("#1E2A4A");
    private static final int COLOR_ARC_BG = Color.parseColor("#2A3A5C");
    private static final int COLOR_GREEN = Color.parseColor("#4CAF50");
    private static final int COLOR_YELLOW = Color.parseColor("#FFC107");
    private static final int COLOR_RED = Color.parseColor("#F44336");
    private static final int COLOR_BLUE = Color.parseColor("#2196F3");
    private static final int COLOR_CYAN = Color.parseColor("#00BCD4");
    private static final int COLOR_ORANGE = Color.parseColor("#FF9800");
    private static final int COLOR_TEXT = Color.parseColor("#EEEEEE");
    private static final int COLOR_LABEL = Color.parseColor("#88FFFFFF");
    private static final int COLOR_DIVIDER = Color.parseColor("#2A3A5C");

    public DashboardView(Context context) {
        super(context);
        init();
    }

    public DashboardView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public DashboardView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        setLayerType(LAYER_TYPE_HARDWARE, null);

        paintBg.setColor(COLOR_BG);
        paintBg.setStyle(Paint.Style.FILL);

        paintGaugeArc.setStyle(Paint.Style.STROKE);
        paintGaugeArc.setStrokeCap(Paint.Cap.ROUND);

        paintGaugeValue.setStyle(Paint.Style.STROKE);
        paintGaugeValue.setStrokeCap(Paint.Cap.ROUND);

        paintGaugeBg.setColor(COLOR_GAUGE_BG);
        paintGaugeBg.setStyle(Paint.Style.FILL);

        paintValue.setColor(COLOR_TEXT);
        paintValue.setTextAlign(Paint.Align.CENTER);
        paintValue.setFakeBoldText(true);

        paintLabel.setColor(COLOR_LABEL);
        paintLabel.setTextAlign(Paint.Align.CENTER);

        paintUnit.setColor(Color.parseColor("#66FFFFFF"));
        paintUnit.setTextAlign(Paint.Align.CENTER);

        paintDivider.setColor(COLOR_DIVIDER);
        paintDivider.setStrokeWidth(1f);
    }

    public void updateData(Elm327Manager.DashboardData data) {
        this.rpm = data.rpm;
        this.coolantTemp = data.coolantTemp;
        this.intakeAirTemp = data.intakeAirTemp;
        this.speed = data.speed;
        this.batteryVoltage = data.batteryVoltage;
        postInvalidate();
    }

    /** Atualizado por um loop de poll independente (Speeduino tem sua
     * própria porta USB) — pode chegar em instantes diferentes de
     * updateData(), por isso é um método separado, não parte do mesmo
     * "pacote" de dados. */
    public void updateSpeeduinoData(SpeeduinoManager.SpeeduinoData data) {
        this.speeduinoAdvance = data.advanceDeg;
        this.speeduinoMap = data.mapKpa;
        this.speeduinoBaro = data.baroKpa;
        this.speeduinoFlexPct = data.ethanolPct;
        this.speeduinoTps = data.tpsPct;
        postInvalidate();
    }

    public void clearData() {
        rpm = null;
        coolantTemp = null;
        intakeAirTemp = null;
        speed = null;
        batteryVoltage = null;
        postInvalidate();
    }

    /** Chamado quando a Speeduino desconecta — os gauges dela voltam a
     * mostrar "--", os outros (OBD2) continuam como estavam. */
    public void clearSpeeduinoData() {
        speeduinoAdvance = null;
        speeduinoMap = null;
        speeduinoBaro = null;
        speeduinoFlexPct = null;
        speeduinoTps = null;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth();
        int h = getHeight();

        canvas.drawRect(0, 0, w, h, paintBg);

        // Layout: 3 colunas x 3 linhas — a 3ª linha (MAP/BARO/FLEX) vem da
        // Speeduino, assim como o PONTO na 2ª linha (trocado do OBD2 pra
        // Speeduino: quem comanda a ignição de verdade agora é ela).
        float pad = 8f;
        float cellW = (w - pad * 4) / 3f;
        float cellH = (h - pad * 4) / 3f;
        float row0 = pad, row1 = pad * 2 + cellH, row2 = pad * 3 + cellH * 2;
        float col0 = pad, col1 = pad * 2 + cellW, col2 = pad * 3 + cellW * 2;

        // Linha 1: RPM | Temp Água | Temp Ar
        drawGauge(canvas, col0, row0,
                cellW, cellH, "RPM",
                rpm != null ? String.valueOf(rpm) : "--", "",
                rpm != null ? rpm / 7000f : 0f,
                getRpmColor(rpm));

        drawGauge(canvas, col1, row0,
                cellW, cellH, "ÁGUA",
                coolantTemp != null ? String.format(Locale.US, "%.0f", coolantTemp) : "--", "°C",
                coolantTemp != null ? coolantTemp / 130f : 0f,
                getCoolantColor(coolantTemp));

        drawGauge(canvas, col2, row0,
                cellW, cellH, "AR ADMISSÃO",
                intakeAirTemp != null ? String.format(Locale.US, "%.0f", intakeAirTemp) : "--", "°C",
                intakeAirTemp != null ? (intakeAirTemp + 40f) / 100f : 0f,
                COLOR_CYAN);

        // Linha 2: Velocidade | Ponto (Speeduino) | TPS
        drawGauge(canvas, col0, row1,
                cellW, cellH, "VELOCIDADE",
                speed != null ? String.valueOf(speed) : "--", "km/h",
                speed != null ? speed / 200f : 0f,
                getSpeedColor(speed));

        drawGauge(canvas, col1, row1,
                cellW, cellH, "PONTO",
                speeduinoAdvance != null ? String.format(Locale.US, "%.0f", speeduinoAdvance) : "--", "°",
                speeduinoAdvance != null ? (speeduinoAdvance + 20f) / 60f : 0f,
                getTimingColor(speeduinoAdvance));

        drawGauge(canvas, col2, row1,
                cellW, cellH, "TPS",
                speeduinoTps != null ? String.format(Locale.US, "%.1f", speeduinoTps) : "--", "%",
                speeduinoTps != null ? speeduinoTps / 100f : 0f,
                getTpsColor(speeduinoTps));

        // Linha 3: MAP | BARO | FLEX (todos da Speeduino)
        drawGauge(canvas, col0, row2,
                cellW, cellH, "MAP",
                speeduinoMap != null ? String.format(Locale.US, "%.0f", speeduinoMap) : "--", "kPa",
                speeduinoMap != null ? speeduinoMap / 105f : 0f,
                COLOR_BLUE);

        drawGauge(canvas, col1, row2,
                cellW, cellH, "BARO",
                speeduinoBaro != null ? String.format(Locale.US, "%.0f", speeduinoBaro) : "--", "kPa",
                speeduinoBaro != null ? speeduinoBaro / 105f : 0f,
                COLOR_CYAN);

        drawGauge(canvas, col2, row2,
                cellW, cellH, "FLEX",
                speeduinoFlexPct != null ? String.valueOf(speeduinoFlexPct) : "--", "% etanol",
                speeduinoFlexPct != null ? speeduinoFlexPct / 100f : 0f,
                COLOR_ORANGE);
    }

    private void drawGauge(Canvas canvas, float x, float y, float w, float h,
                           String label, String value, String unit,
                           float fraction, int color) {
        float cx = x + w / 2f;
        float cy = y + h * 0.48f;
        float radius = Math.min(w, h) * 0.32f;
        float arcWidth = radius * 0.18f;
        fraction = Math.max(0f, Math.min(1f, fraction));

        // Background da célula
        RectF cellRect = new RectF(x, y, x + w, y + h);
        canvas.drawRoundRect(cellRect, 8f, 8f, paintGaugeBg);

        // Arco de fundo
        RectF arcRect = new RectF(cx - radius, cy - radius, cx + radius, cy + radius);
        paintGaugeArc.setColor(COLOR_ARC_BG);
        paintGaugeArc.setStrokeWidth(arcWidth);
        canvas.drawArc(arcRect, 150f, 240f, false, paintGaugeArc);

        // Arco de valor
        paintGaugeValue.setColor(color);
        paintGaugeValue.setStrokeWidth(arcWidth);
        canvas.drawArc(arcRect, 150f, 240f * fraction, false, paintGaugeValue);

        // Valor central
        paintValue.setTextSize(radius * 0.7f);
        paintValue.setColor(color);
        canvas.drawText(value, cx, cy + radius * 0.2f, paintValue);

        // Unidade
        paintUnit.setTextSize(radius * 0.3f);
        canvas.drawText(unit, cx, cy + radius * 0.55f, paintUnit);

        // Label
        paintLabel.setTextSize(Math.min(radius * 0.3f, 22f));
        canvas.drawText(label, cx, y + h - 8f, paintLabel);
    }

    private int getRpmColor(Integer rpm) {
        if (rpm == null) return COLOR_GREEN;
        if (rpm > 5500) return COLOR_RED;
        if (rpm > 4000) return COLOR_YELLOW;
        return COLOR_GREEN;
    }

    private int getCoolantColor(Float temp) {
        if (temp == null) return COLOR_GREEN;
        if (temp > 105) return COLOR_RED;
        if (temp > 95) return COLOR_YELLOW;
        if (temp < 60) return COLOR_BLUE;
        return COLOR_GREEN;
    }

    private int getTimingColor(Float timing) {
        if (timing == null) return COLOR_ORANGE;
        if (timing < 0) return COLOR_RED;
        if (timing < 5) return COLOR_YELLOW;
        return COLOR_ORANGE;
    }

    private int getSpeedColor(Integer speed) {
        if (speed == null) return COLOR_GREEN;
        if (speed > 140) return COLOR_RED;
        if (speed > 100) return COLOR_YELLOW;
        return COLOR_GREEN;
    }

    private int getTpsColor(Float tps) {
        if (tps == null) return COLOR_GREEN;
        if (tps > 80) return COLOR_RED;
        if (tps > 50) return COLOR_YELLOW;
        return COLOR_GREEN;
    }
}
