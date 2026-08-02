package com.obd2.lambda;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.Shader;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * View customizada para gráfico de lambda em tempo real.
 * Réplica exata do gráfico matplotlib do script Python lambda_monitor.py:
 * - Escala Y fixa: 0.5 a 1.5 (lambda)
 * - Linhas de referência: λ=0.9 (laranja) e λ=1.10 (azul)
 * - Banco 1 verde, Banco 2 vermelho
 * - Anotações com valor lambda + corrente em mA
 * - 100 pontos máximo
 */
public class LambdaChartView extends View {

    private static final int MAX_POINTS = 100;
    private static final float Y_MIN = 0.7f;
    private static final float Y_MAX = 1.3f;

    // Séries de dados: lambda (para plotar) e corrente (para anotação)
    private final List<Float> seriesLambda1 = new ArrayList<>();
    private final List<Float> seriesLambda2 = new ArrayList<>();
    private Float lastCurrent1 = null;
    private Float lastCurrent2 = null;

    // Paints - otimizado: sem título, sem legenda, sem dots individuais
    private final Paint paintLine1 = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintLine2 = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintFill1 = new Paint();  // Fill sutil sob linha 1
    private final Paint paintFill2 = new Paint();  // Fill sutil sob linha 2
    private final Paint paintGrid = new Paint();
    private final Paint paintRefOrange = new Paint();
    private final Paint paintRefBlue = new Paint();
    private final Paint paintBg = new Paint();
    private final Paint paintAxisText = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintAnnotBg = new Paint();
    private final Paint paintAnnot = new Paint(Paint.ANTI_ALIAS_FLAG);  // Compartilhado, cor muda dinamicamente
    private final Paint paintRefLabel = new Paint(Paint.ANTI_ALIAS_FLAG);

    // Cores por estado
    private static final int COLOR_STOICH = Color.parseColor("#81C784");      // Verde - estequiométrica
    private static final int COLOR_RICH = Color.parseColor("#64B5F6");        // Azul - rico
    private static final int COLOR_LEAN = Color.parseColor("#FFD54F");        // Amarelo - pobre
    private static final int COLOR_EXTREME = Color.parseColor("#EF5350");     // Vermelho - extremo (<0.88 ou >1.15)
    private static final int COLOR_STOICH_LINE = Color.parseColor("#4CAF50");
    private static final int COLOR_RICH_LINE = Color.parseColor("#2196F3");
    private static final int COLOR_LEAN_LINE = Color.parseColor("#FFC107");
    private static final int COLOR_EXTREME_LINE = Color.parseColor("#F44336");

    private final Path pathLine1 = new Path();
    private final Path pathLine2 = new Path();

    // Margens reduzidas para maximizar área do gráfico
    private static final float MARGIN_LEFT = 60f;
    private static final float MARGIN_RIGHT = 10f;
    private static final float MARGIN_TOP = 8f;
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
        // Habilitar hardware acceleration
        setLayerType(LAYER_TYPE_HARDWARE, null);

        // Banco 1 - cor dinâmica (começa verde)
        paintLine1.setColor(COLOR_STOICH_LINE);
        paintLine1.setStrokeWidth(3f);
        paintLine1.setStyle(Paint.Style.STROKE);
        paintLine1.setStrokeJoin(Paint.Join.ROUND);
        paintLine1.setStrokeCap(Paint.Cap.ROUND);

        // Banco 2 - cor dinâmica (começa verde)
        paintLine2.setColor(COLOR_STOICH_LINE);
        paintLine2.setStrokeWidth(3f);
        paintLine2.setStyle(Paint.Style.STROKE);
        paintLine2.setStrokeJoin(Paint.Join.ROUND);
        paintLine2.setStrokeCap(Paint.Cap.ROUND);

        // Fill semi-transparente sob as linhas
        paintFill1.setStyle(Paint.Style.FILL);
        paintFill2.setStyle(Paint.Style.FILL);

        // Grid leve
        paintGrid.setColor(Color.parseColor("#33FFFFFF"));
        paintGrid.setStrokeWidth(0.5f);
        paintGrid.setStyle(Paint.Style.STROKE);

        // Linha de referência λ=0.9 - laranja tracejada
        paintRefOrange.setColor(Color.parseColor("#FF9800"));
        paintRefOrange.setStrokeWidth(1.5f);
        paintRefOrange.setStyle(Paint.Style.STROKE);
        paintRefOrange.setPathEffect(new DashPathEffect(new float[]{10, 6}, 0));

        // Linha de referência λ=1.10 - azul tracejada
        paintRefBlue.setColor(Color.parseColor("#2196F3"));
        paintRefBlue.setStrokeWidth(1.5f);
        paintRefBlue.setStyle(Paint.Style.STROKE);
        paintRefBlue.setPathEffect(new DashPathEffect(new float[]{10, 6}, 0));

        // Background
        paintBg.setColor(Color.parseColor("#16213E"));
        paintBg.setStyle(Paint.Style.FILL);

        // Texto eixo Y - compacto
        paintAxisText.setColor(Color.parseColor("#999999"));
        paintAxisText.setTextSize(20f);
        paintAxisText.setTextAlign(Paint.Align.RIGHT);

        // Fundo das anotações
        paintAnnotBg.setColor(Color.parseColor("#CC000000"));
        paintAnnotBg.setStyle(Paint.Style.FILL);

        // Anotação - cor dinâmica, compartilhada
        paintAnnot.setColor(COLOR_STOICH);
        paintAnnot.setTextSize(48f);
        paintAnnot.setFakeBoldText(true);

        // Labels das linhas de referência
        paintRefLabel.setTextSize(16f);
        paintRefLabel.setTextAlign(Paint.Align.LEFT);
    }

    /**
     * Adiciona novos pontos ao gráfico.
     * Recebe valores de lambda (para plotar) e corrente em mA (para anotação).
     */
    public void addData(Float lambda1, Float lambda2, Float current1, Float current2) {
        seriesLambda1.add(lambda1);
        seriesLambda2.add(lambda2);
        lastCurrent1 = current1;
        lastCurrent2 = current2;

        while (seriesLambda1.size() > MAX_POINTS) seriesLambda1.remove(0);
        while (seriesLambda2.size() > MAX_POINTS) seriesLambda2.remove(0);

        postInvalidate();
    }

    /**
     * Versão simplificada para compatibilidade - aceita apenas lambda.
     */
    public void addData(Float lambda1, Float lambda2) {
        addData(lambda1, lambda2, null, null);
    }

    public void clearData() {
        seriesLambda1.clear();
        seriesLambda2.clear();
        lastCurrent1 = null;
        lastCurrent2 = null;
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

        // Grid Y reduzido: só linhas a cada 0.2 + referências
        // Grid: de 0.7 a 1.3, passo 0.1
        for (int i = 0; i <= 6; i++) {
            float val = Y_MIN + (i * 0.1f);
            float y = chartTop + chartH * (1f - (val - Y_MIN) / (Y_MAX - Y_MIN));
            canvas.drawLine(chartLeft, y, chartRight, y, paintGrid);
            String label = String.format(Locale.US, "%.1f", val);
            canvas.drawText(label, chartLeft - 6f, y + 7f, paintAxisText);
        }

        // Linha de referência λ=0.9 (laranja tracejada)
        float yRef09 = chartTop + chartH * (1f - (0.9f - Y_MIN) / (Y_MAX - Y_MIN));
        canvas.drawLine(chartLeft, yRef09, chartRight, yRef09, paintRefOrange);
        paintRefLabel.setColor(Color.parseColor("#FF9800"));
        canvas.drawText("0.9", chartRight - 50f, yRef09 - 4f, paintRefLabel);

        // Linha de referência λ=1.10 (azul tracejada)
        float yRef110 = chartTop + chartH * (1f - (1.10f - Y_MIN) / (Y_MAX - Y_MIN));
        canvas.drawLine(chartLeft, yRef110, chartRight, yRef110, paintRefBlue);
        paintRefLabel.setColor(Color.parseColor("#2196F3"));
        canvas.drawText("1.10", chartRight - 60f, yRef110 - 4f, paintRefLabel);

        // Determinar cores dinâmicas baseadas no lambda atual
        Float lastL1 = getLastNonNull(seriesLambda1);
        Float lastL2 = getLastNonNull(seriesLambda2);

        int color1Line = getLineColor(lastL1);
        int color1Text = getTextColor(lastL1);
        int color2Line = getLineColor(lastL2);
        int color2Text = getTextColor(lastL2);

        paintLine1.setColor(color1Line);
        paintLine2.setColor(color2Line);

        // Séries com fill sutil sob a linha
        if (!seriesLambda1.isEmpty()) {
            paintFill1.setColor(withAlpha(color1Line, 30));
            drawSeriesWithFill(canvas, seriesLambda1, pathLine1, paintLine1, paintFill1,
                    chartLeft, chartTop, chartW, chartH, chartBottom);
        }
        if (!seriesLambda2.isEmpty()) {
            paintFill2.setColor(withAlpha(color2Line, 30));
            drawSeriesWithFill(canvas, seriesLambda2, pathLine2, paintLine2, paintFill2,
                    chartLeft, chartTop, chartW, chartH, chartBottom);
        }

        // Anotações centralizadas - cor muda com estado
        float centerX = chartLeft + chartW / 2f;
        float annotY = chartTop + 44f;

        if (lastL1 != null) {
            String text = String.format(Locale.US, "B1: %.3f", lastL1);
            paintAnnot.setColor(color1Text);
            drawAnnotationCentered(canvas, text, centerX, annotY, paintAnnot);
            annotY += 56f;
        }

        if (lastL2 != null) {
            String text = String.format(Locale.US, "B2: %.3f", lastL2);
            paintAnnot.setColor(color2Text);
            drawAnnotationCentered(canvas, text, centerX, annotY, paintAnnot);
        }
    }

    private void drawSeriesWithFill(Canvas canvas, List<Float> series, Path linePath,
                            Paint linePaint, Paint fillPaint,
                            float chartLeft, float chartTop,
                            float chartW, float chartH, float chartBottom) {
        linePath.reset();
        boolean started = false;
        float firstX = 0, lastX = 0;

        int count = series.size();
        for (int i = 0; i < count; i++) {
            Float val = series.get(i);
            if (val == null) continue;

            float x = chartLeft + (chartW * i / Math.max(1, MAX_POINTS - 1));
            float clamped = Math.max(Y_MIN, Math.min(Y_MAX, val));
            float y = chartTop + chartH * (1f - (clamped - Y_MIN) / (Y_MAX - Y_MIN));

            if (!started) {
                linePath.moveTo(x, y);
                firstX = x;
                started = true;
            } else {
                linePath.lineTo(x, y);
            }
            lastX = x;
        }

        if (started) {
            // Fill: fechar path até a base do gráfico
            Path fillPath = new Path(linePath);
            fillPath.lineTo(lastX, chartBottom);
            fillPath.lineTo(firstX, chartBottom);
            fillPath.close();
            canvas.drawPath(fillPath, fillPaint);

            // Linha sobre o fill
            canvas.drawPath(linePath, linePaint);
        }
    }

    private void drawAnnotationCentered(Canvas canvas, String text, float centerX, float y, Paint textPaint) {
        float textWidth = textPaint.measureText(text);
        float pad = 12f;
        float x = centerX - textWidth / 2f;
        RectF bg = new RectF(x - pad, y - 44f, x + textWidth + pad, y + 10f);
        canvas.drawRoundRect(bg, 6f, 6f, paintAnnotBg);
        canvas.drawText(text, x, y, textPaint);
    }

    private Float getLastNonNull(List<Float> list) {
        for (int i = list.size() - 1; i >= 0; i--) {
            if (list.get(i) != null) return list.get(i);
        }
        return null;
    }

    /** Cor da linha: verde=estequio, azul=rico, amarelo=pobre, vermelho=extremo */
    private int getLineColor(Float lambda) {
        if (lambda == null) return COLOR_STOICH_LINE;
        if (lambda < 0.88f || lambda > 1.15f) return COLOR_EXTREME_LINE;
        if (lambda < 0.98f) return COLOR_RICH_LINE;
        if (lambda > 1.02f) return COLOR_LEAN_LINE;
        return COLOR_STOICH_LINE;
    }

    /** Cor do texto: verde=estequio, azul=rico, amarelo=pobre, vermelho=extremo */
    private int getTextColor(Float lambda) {
        if (lambda == null) return COLOR_STOICH;
        if (lambda < 0.88f || lambda > 1.15f) return COLOR_EXTREME;
        if (lambda < 0.98f) return COLOR_RICH;
        if (lambda > 1.02f) return COLOR_LEAN;
        return COLOR_STOICH;
    }

    /** Cria cor com alpha especificado (0-255) */
    private int withAlpha(int color, int alpha) {
        return Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color));
    }
}
