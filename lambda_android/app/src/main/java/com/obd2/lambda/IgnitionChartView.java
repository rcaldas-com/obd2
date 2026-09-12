package com.obd2.lambda;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Gráfico de ponto de ignição em tempo real: a curva da ECU original (PID 010E,
 * a que denuncia o recuo do sensor de detonação) e, como referência visual, o
 * ponto que a Speeduino está realmente aplicando.
 *
 * Diferença importante pro LambdaChartView: a escala Y aqui é adaptativa. O que
 * se procura nessa tela é um recuo de 2-3° — numa escala fixa de 0 a 50° isso
 * seria um degrau de 5% da altura, praticamente invisível justo no momento que
 * importa. A escala segue os dados, com um vão mínimo pra não "explodir" o
 * ruído quando tudo está parado, e presa a passos de 2° pra não ficar tremendo
 * a cada amostra.
 */
public class IgnitionChartView extends View {

    private static final int MAX_POINTS = 120;

    /** Vão mínimo do eixo Y, pra variação de ruído não virar montanha. */
    private static final float MIN_SPAN_DEG = 8f;
    /** Passo de arredondamento dos limites do eixo (evita tremedeira). */
    private static final float AXIS_STEP_DEG = 2f;

    private static final int COLOR_OEM = Color.parseColor("#FFB74D");        // original
    private static final int COLOR_SPEEDUINO = Color.parseColor("#4FC3F7");  // Speeduino
    private static final int COLOR_REF = Color.parseColor("#B0BEC5");
    private static final int COLOR_OK = Color.parseColor("#81C784");
    private static final int COLOR_EVENT = Color.parseColor("#EF5350");

    private final List<Float> seriesOem = new ArrayList<>();
    private final List<Float> seriesSpeeduino = new ArrayList<>();

    private Float reference;
    private Float dropDeg;
    private KnockWatch.State state = KnockWatch.State.SEM_DADOS;
    private boolean anchorLocked = false;

    private final Paint paintOem = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintSpeeduino = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintRef = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintGrid = new Paint();
    private final Paint paintBg = new Paint();
    private final Paint paintAxisText = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintAnnotBg = new Paint();
    private final Paint paintAnnot = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintStatus = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintEventBorder = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintLegend = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final Path pathOem = new Path();
    private final Path pathSpeeduino = new Path();

    private static final float MARGIN_LEFT = 60f;
    private static final float MARGIN_RIGHT = 10f;
    private static final float MARGIN_TOP = 8f;
    private static final float MARGIN_BOTTOM = 8f;

    public IgnitionChartView(Context context) {
        super(context);
        init();
    }

    public IgnitionChartView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public IgnitionChartView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        setLayerType(LAYER_TYPE_HARDWARE, null);

        paintOem.setColor(COLOR_OEM);
        paintOem.setStrokeWidth(4f);
        paintOem.setStyle(Paint.Style.STROKE);
        paintOem.setStrokeJoin(Paint.Join.ROUND);
        paintOem.setStrokeCap(Paint.Cap.ROUND);

        // Mais fina e tracejada: é referência do que EU mandei, não a medida
        // que está sendo caçada.
        paintSpeeduino.setColor(COLOR_SPEEDUINO);
        paintSpeeduino.setStrokeWidth(2.5f);
        paintSpeeduino.setStyle(Paint.Style.STROKE);
        paintSpeeduino.setStrokeJoin(Paint.Join.ROUND);
        paintSpeeduino.setPathEffect(new DashPathEffect(new float[]{12, 8}, 0));

        paintRef.setColor(COLOR_REF);
        paintRef.setStrokeWidth(1.5f);
        paintRef.setStyle(Paint.Style.STROKE);
        paintRef.setPathEffect(new DashPathEffect(new float[]{8, 6}, 0));

        paintGrid.setColor(Color.parseColor("#33FFFFFF"));
        paintGrid.setStrokeWidth(0.5f);
        paintGrid.setStyle(Paint.Style.STROKE);

        paintBg.setColor(Color.parseColor("#16213E"));
        paintBg.setStyle(Paint.Style.FILL);

        paintAxisText.setColor(Color.parseColor("#999999"));
        paintAxisText.setTextSize(20f);
        paintAxisText.setTextAlign(Paint.Align.RIGHT);

        paintAnnotBg.setColor(Color.parseColor("#CC000000"));
        paintAnnotBg.setStyle(Paint.Style.FILL);

        paintAnnot.setTextSize(46f);
        paintAnnot.setFakeBoldText(true);

        paintStatus.setTextSize(22f);
        paintStatus.setTextAlign(Paint.Align.CENTER);

        paintLegend.setTextSize(20f);

        paintEventBorder.setStyle(Paint.Style.STROKE);
        paintEventBorder.setStrokeWidth(6f);
        paintEventBorder.setColor(COLOR_EVENT);
    }

    /** Uma amostra: ponto da original e ponto aplicado pela Speeduino. */
    public void addData(Float oemAdvance, Float speeduinoAdvance) {
        seriesOem.add(oemAdvance);
        seriesSpeeduino.add(speeduinoAdvance);
        while (seriesOem.size() > MAX_POINTS) seriesOem.remove(0);
        while (seriesSpeeduino.size() > MAX_POINTS) seriesSpeeduino.remove(0);
        postInvalidate();
    }

    public void updateStatus(KnockWatch.State state, Float reference, Float dropDeg, boolean anchorLocked) {
        this.state = state;
        this.reference = reference;
        this.dropDeg = dropDeg;
        this.anchorLocked = anchorLocked;
        postInvalidate();
    }

    public void clearData() {
        seriesOem.clear();
        seriesSpeeduino.clear();
        reference = null;
        dropDeg = null;
        state = KnockWatch.State.SEM_DADOS;
        anchorLocked = false;
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

        canvas.drawRect(0, 0, w, h, paintBg);

        float[] bounds = computeBounds();
        float yMin = bounds[0];
        float yMax = bounds[1];

        // Grid + rótulos do eixo em graus.
        int lines = Math.max(2, Math.round((yMax - yMin) / AXIS_STEP_DEG));
        for (int i = 0; i <= lines; i++) {
            float val = yMin + (yMax - yMin) * i / lines;
            float y = yFor(val, yMin, yMax, chartTop, chartH);
            canvas.drawLine(chartLeft, y, chartRight, y, paintGrid);
            canvas.drawText(String.format(Locale.US, "%.0f", val), chartLeft - 6f, y + 7f, paintAxisText);
        }

        // Referência fixada: é dela que o recuo é medido, então precisa estar
        // visível na tela o tempo todo.
        if (reference != null) {
            float yRef = yFor(reference, yMin, yMax, chartTop, chartH);
            canvas.drawLine(chartLeft, yRef, chartRight, yRef, paintRef);
            paintLegend.setColor(COLOR_REF);
            // Sem emoji no rótulo: a multimídia roda Android 5 e fonte velha
            // não tem os blocos fora do BMP — sairia quadradinho.
            canvas.drawText(String.format(Locale.US, "REF %.1f°%s", reference, anchorLocked ? " (fixa)" : ""),
                    chartLeft + 6f, yRef - 6f, paintLegend);
        }

        drawSeries(canvas, seriesSpeeduino, pathSpeeduino, paintSpeeduino, chartLeft, chartTop, chartW, chartH, yMin, yMax);
        drawSeries(canvas, seriesOem, pathOem, paintOem, chartLeft, chartTop, chartW, chartH, yMin, yMax);

        Float lastOem = lastNonNull(seriesOem);
        Float lastSpd = lastNonNull(seriesSpeeduino);

        // Números grandes ao centro, mesma linguagem visual da tela de lambda.
        float centerX = chartLeft + chartW / 2f;
        float annotY = chartTop + 44f;

        if (lastOem != null) {
            paintAnnot.setColor(COLOR_OEM);
            drawAnnotationCentered(canvas, String.format(Locale.US, "ORIG %.1f°", lastOem), centerX, annotY);
            annotY += 54f;
        }
        if (lastSpd != null) {
            paintAnnot.setColor(COLOR_SPEEDUINO);
            drawAnnotationCentered(canvas, String.format(Locale.US, "SPD %.0f°", lastSpd), centerX, annotY);
            annotY += 54f;
        }
        if (dropDeg != null && state != KnockWatch.State.SEM_DADOS && state != KnockWatch.State.INSTAVEL) {
            paintAnnot.setColor(state == KnockWatch.State.RECUO ? COLOR_EVENT : COLOR_OK);
            drawAnnotationCentered(canvas, String.format(Locale.US, "RECUO %.1f°", dropDeg), centerX, annotY);
        }

        // Linha de estado logo acima da barra inferior.
        paintStatus.setColor(statusColor());
        canvas.drawText(statusText(), centerX, chartBottom - 10f, paintStatus);

        if (state == KnockWatch.State.RECUO) {
            canvas.drawRect(3, 3, w - 3, h - 3, paintEventBorder);
        }
    }

    /**
     * Limites do eixo pelos dados visíveis (mais a referência, que precisa
     * caber), com vão mínimo e arredondamento em passos de 2° — sem isso a
     * escala mudaria a cada amostra e o olho perderia a noção de degrau.
     */
    private float[] computeBounds() {
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;

        for (Float v : seriesOem) if (v != null) { min = Math.min(min, v); max = Math.max(max, v); }
        for (Float v : seriesSpeeduino) if (v != null) { min = Math.min(min, v); max = Math.max(max, v); }
        if (reference != null) { min = Math.min(min, reference); max = Math.max(max, reference); }

        if (min == Float.MAX_VALUE) return new float[]{0f, MIN_SPAN_DEG};

        float span = max - min;
        if (span < MIN_SPAN_DEG) {
            float center = (max + min) / 2f;
            min = center - MIN_SPAN_DEG / 2f;
            max = center + MIN_SPAN_DEG / 2f;
        } else {
            float pad = span * 0.12f;
            min -= pad;
            max += pad;
        }

        min = (float) Math.floor(min / AXIS_STEP_DEG) * AXIS_STEP_DEG;
        max = (float) Math.ceil(max / AXIS_STEP_DEG) * AXIS_STEP_DEG;
        return new float[]{min, max};
    }

    private float yFor(float value, float yMin, float yMax, float chartTop, float chartH) {
        float clamped = Math.max(yMin, Math.min(yMax, value));
        return chartTop + chartH * (1f - (clamped - yMin) / (yMax - yMin));
    }

    private void drawSeries(Canvas canvas, List<Float> series, Path path, Paint paint,
                            float chartLeft, float chartTop, float chartW, float chartH,
                            float yMin, float yMax) {
        path.reset();
        boolean started = false;
        int count = series.size();
        for (int i = 0; i < count; i++) {
            Float val = series.get(i);
            if (val == null) continue;
            float x = chartLeft + (chartW * i / Math.max(1, MAX_POINTS - 1));
            float y = yFor(val, yMin, yMax, chartTop, chartH);
            if (!started) {
                path.moveTo(x, y);
                started = true;
            } else {
                path.lineTo(x, y);
            }
        }
        if (started) canvas.drawPath(path, paint);
    }

    private void drawAnnotationCentered(Canvas canvas, String text, float centerX, float y) {
        float textWidth = paintAnnot.measureText(text);
        float pad = 12f;
        float x = centerX - textWidth / 2f;
        RectF bg = new RectF(x - pad, y - 42f, x + textWidth + pad, y + 10f);
        canvas.drawRoundRect(bg, 6f, 6f, paintAnnotBg);
        canvas.drawText(text, x, y, paintAnnot);
    }

    private String statusText() {
        switch (state) {
            case RECUO:
                return "RECUO DA ORIGINAL — volte o ponto";
            case ESTAVEL:
                return anchorLocked ? "referência fixa — pode subir o ponto" : "condição estável — pode subir o ponto";
            case INSTAVEL:
                return "condição instável — segure rotação/carga";
            default:
                return "sem dados (precisa de original + Speeduino)";
        }
    }

    private int statusColor() {
        switch (state) {
            case RECUO: return COLOR_EVENT;
            case ESTAVEL: return COLOR_OK;
            case INSTAVEL: return Color.parseColor("#FFD54F");
            default: return Color.parseColor("#999999");
        }
    }

    private Float lastNonNull(List<Float> list) {
        for (int i = list.size() - 1; i >= 0; i--) {
            if (list.get(i) != null) return list.get(i);
        }
        return null;
    }
}
