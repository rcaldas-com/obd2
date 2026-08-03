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

    // Dados atuais
    private Integer rpm;
    private Float coolantTemp;
    private Float intakeAirTemp;
    private Integer speed;
    private Float timingAdvance;
    private Float tps;
    private Float batteryVoltage;

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
        this.timingAdvance = data.timingAdvance;
        this.tps = data.tps;
        this.batteryVoltage = data.batteryVoltage;
        postInvalidate();
    }

    public void clearData() {
        rpm = null;
        coolantTemp = null;
        intakeAirTemp = null;
        speed = null;
        timingAdvance = null;
        tps = null;
        batteryVoltage = null;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth();
        int h = getHeight();

        canvas.drawRect(0, 0, w, h, paintBg);

        // Layout: 3 colunas x 2 linhas
        float pad = 8f;
        float cellW = (w - pad * 4) / 3f;
        float cellH = (h - pad * 3) / 2f;

        // Linha 1: RPM | Temp Água | Temp Ar
        drawGauge(canvas, pad, pad,
                cellW, cellH, "RPM",
                rpm != null ? String.valueOf(rpm) : "--", "",
                rpm != null ? rpm / 7000f : 0f,
                getRpmColor(rpm));

        drawGauge(canvas, pad * 2 + cellW, pad,
                cellW, cellH, "ÁGUA",
                coolantTemp != null ? String.format(Locale.US, "%.0f", coolantTemp) : "--", "°C",
                coolantTemp != null ? coolantTemp / 130f : 0f,
                getCoolantColor(coolantTemp));

        drawGauge(canvas, pad * 3 + cellW * 2, pad,
                cellW, cellH, "AR ADMISSÃO",
                intakeAirTemp != null ? String.format(Locale.US, "%.0f", intakeAirTemp) : "--", "°C",
                intakeAirTemp != null ? (intakeAirTemp + 40f) / 100f : 0f,
                COLOR_CYAN);

        // Linha 2: Velocidade | Ponto | TPS
        drawGauge(canvas, pad, pad * 2 + cellH,
                cellW, cellH, "VELOCIDADE",
                speed != null ? String.valueOf(speed) : "--", "km/h",
                speed != null ? speed / 200f : 0f,
                getSpeedColor(speed));

        drawGauge(canvas, pad * 2 + cellW, pad * 2 + cellH,
                cellW, cellH, "PONTO",
                timingAdvance != null ? String.format(Locale.US, "%.1f", timingAdvance) : "--", "°",
                timingAdvance != null ? (timingAdvance + 20f) / 60f : 0f,
                getTimingColor(timingAdvance));

        drawGauge(canvas, pad * 3 + cellW * 2, pad * 2 + cellH,
                cellW, cellH, "TPS",
                tps != null ? String.format(Locale.US, "%.1f", tps) : "--", "%",
                tps != null ? tps / 100f : 0f,
                getTpsColor(tps));
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
