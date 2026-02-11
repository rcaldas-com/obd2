package com.obd2.lambda;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

/**
 * View customizada para gráfico de lambda em tempo real.
 * Desenha diretamente no Canvas, sem dependências externas.
 * Compatível com Android 5+.
 */
public class LambdaChartView extends View {

    private static final int MAX_POINTS = 80;
    private static final float Y_MIN = -1.0f;
    private static final float Y_MAX = 1.0f;

    private final List<Float> seriesO2S1 = new ArrayList<>();
    private final List<Float> seriesO2S5 = new ArrayList<>();

    private final Paint paintO2S1 = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintO2S5 = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintGrid = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintRefLine = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintBg = new Paint();
    private final Paint paintText = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintLabel = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintFillO2S1 = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintFillO2S5 = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final Path pathO2S1 = new Path();
    private final Path pathO2S5 = new Path();
    private final Path fillPathO2S1 = new Path();
    private final Path fillPathO2S5 = new Path();

    // Margens
    private static final float MARGIN_LEFT = 60f;
    private static final float MARGIN_RIGHT = 16f;
    private static final float MARGIN_TOP = 12f;
    private static final float MARGIN_BOTTOM = 8f;

    public LambdaChartView(Context context) {
        super(context);
        init();
    }

    public LambdaChartView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public LambdaChartView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        // O2 S1 - verde
        paintO2S1.setColor(Color.parseColor("#4CAF50"));
        paintO2S1.setStrokeWidth(3f);
        paintO2S1.setStyle(Paint.Style.STROKE);

        paintFillO2S1.setColor(Color.parseColor("#1A4CAF50"));
        paintFillO2S1.setStyle(Paint.Style.FILL);

        // O2 S5 - vermelho
        paintO2S5.setColor(Color.parseColor("#F44336"));
        paintO2S5.setStrokeWidth(3f);
        paintO2S5.setStyle(Paint.Style.STROKE);

        paintFillO2S5.setColor(Color.parseColor("#1AF44336"));
        paintFillO2S5.setStyle(Paint.Style.FILL);

        // Grid
        paintGrid.setColor(Color.parseColor("#333355"));
        paintGrid.setStrokeWidth(1f);
        paintGrid.setStyle(Paint.Style.STROKE);

        // Linhas de referência (-0.3, 0, +0.3)
        paintRefLine.setColor(Color.parseColor("#55FFFFFF"));
        paintRefLine.setStrokeWidth(1f);
        paintRefLine.setStyle(Paint.Style.STROKE);
        paintRefLine.setPathEffect(new DashPathEffect(new float[]{8, 6}, 0));

        // Background
        paintBg.setColor(Color.parseColor("#16213E"));
        paintBg.setStyle(Paint.Style.FILL);

        // Texto das escalas
        paintText.setColor(Color.parseColor("#AAAAAA"));
        paintText.setTextSize(28f);
        paintText.setTextAlign(Paint.Align.RIGHT);

        // Labels
        paintLabel.setColor(Color.parseColor("#CCCCCC"));
        paintLabel.setTextSize(24f);
    }

    /**
     * Adiciona novos pontos ao gráfico.
     */
    public void addData(Float o2s1, Float o2s5) {
        seriesO2S1.add(o2s1);
        seriesO2S5.add(o2s5);

        while (seriesO2S1.size() > MAX_POINTS) seriesO2S1.remove(0);
        while (seriesO2S5.size() > MAX_POINTS) seriesO2S5.remove(0);

        postInvalidate(); // Thread-safe invalidate
    }

    public void clearData() {
        seriesO2S1.clear();
        seriesO2S5.clear();
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int w = getWidth();
        int h = getHeight();
        float chartLeft = MARGIN_LEFT;
        float chartRight = w - MARGIN_RIGHT;
        float chartTop = MARGIN_TOP;
        float chartBottom = h - MARGIN_BOTTOM;
        float chartW = chartRight - chartLeft;
        float chartH = chartBottom - chartTop;

        // Background
        canvas.drawRect(0, 0, w, h, paintBg);

        // Linhas de referência com labels
        float[] refValues = {-1.0f, -0.3f, 0f, 0.3f, 1.0f};
        for (float val : refValues) {
            float y = chartTop + chartH * (1f - (val - Y_MIN) / (Y_MAX - Y_MIN));
            if (val == -0.3f || val == 0f || val == 0.3f) {
                canvas.drawLine(chartLeft, y, chartRight, y, paintRefLine);
            } else {
                canvas.drawLine(chartLeft, y, chartRight, y, paintGrid);
            }
            String label;
            if (val == 0f) label = "0";
            else if (val == -0.3f) label = "-0.3";
            else if (val == 0.3f) label = "+0.3";
            else if (val == -1f) label = "-1";
            else label = "+1";
            canvas.drawText(label, chartLeft - 6f, y + 10f, paintText);
        }

        // Borda do gráfico
        canvas.drawRect(chartLeft, chartTop, chartRight, chartBottom, paintGrid);

        // Desenhar séries
        if (!seriesO2S1.isEmpty()) {
            drawSeries(canvas, seriesO2S1, pathO2S1, fillPathO2S1, paintO2S1, paintFillO2S1,
                    chartLeft, chartRight, chartTop, chartBottom, chartW, chartH);
        }
        if (!seriesO2S5.isEmpty()) {
            drawSeries(canvas, seriesO2S5, pathO2S5, fillPathO2S5, paintO2S5, paintFillO2S5,
                    chartLeft, chartRight, chartTop, chartBottom, chartW, chartH);
        }

        // Legenda
        float legendX = chartLeft + 10f;
        float legendY = chartTop + 30f;

        paintLabel.setColor(Color.parseColor("#4CAF50"));
        canvas.drawText("● O2 B1", legendX, legendY, paintLabel);

        paintLabel.setColor(Color.parseColor("#F44336"));
        canvas.drawText("● O2 B2", legendX + 150f, legendY, paintLabel);
    }

    private void drawSeries(Canvas canvas, List<Float> series, Path linePath, Path fillPath,
                            Paint linePaint, Paint fillPaint,
                            float chartLeft, float chartRight, float chartTop, float chartBottom,
                            float chartW, float chartH) {
        linePath.reset();
        fillPath.reset();
        boolean started = false;
        float lastX = chartLeft;

        int count = series.size();
        for (int i = 0; i < count; i++) {
            Float val = series.get(i);
            if (val == null) continue;

            float x = chartLeft + (chartW * i / Math.max(1, MAX_POINTS - 1));
            float clamped = Math.max(Y_MIN, Math.min(Y_MAX, val));
            float y = chartTop + chartH * (1f - (clamped - Y_MIN) / (Y_MAX - Y_MIN));

            if (!started) {
                linePath.moveTo(x, y);
                fillPath.moveTo(x, chartBottom);
                fillPath.lineTo(x, y);
                started = true;
            } else {
                linePath.lineTo(x, y);
                fillPath.lineTo(x, y);
            }
            lastX = x;
        }

        if (started) {
            // Fechar o path de preenchimento
            fillPath.lineTo(lastX, chartBottom);
            fillPath.close();
            canvas.drawPath(fillPath, fillPaint);
            canvas.drawPath(linePath, linePaint);
        }
    }
}
