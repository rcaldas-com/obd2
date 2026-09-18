#!/usr/bin/env python3
"""
Filtra um log .msl deixando só os momentos utilizáveis pra acerto de tabela VE,
e gera um .msl novo com esses momentos (mais uma coluna de peso).

A ideia não é cortar binário e sim dar um PESO de 0 a 1 por amostra: o que
estraga uma leitura não é estar variando agora, é ter variado há pouco e a
sonda ainda não ter alcançado. Amostra em regime perfeito pesa 1, amostra
logo depois de um transitório pesa 0 e vai subindo conforme assenta.

Também imprime três coisas úteis:
  - quanto sobra por limiar (pra achar o ponto de não deixar lixo nem perder
    célula pouco visitada);
  - cobertura por célula usando os bins REAIS da tune;
  - conselho de bins pelo que o log mostra + resolução real do sensor MAP.

Uso:
    python3 ve_filter.py log.msl [log2.msl ...] [--tune CurrentTune.msq]
                        [--min-peso 0.3] [--saida pasta/]
"""

import argparse
import math
import os
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------- parâmetros
# Todos ajustáveis por linha de comando; estes são os padrões de partida.
CLT_MIN = 70.0              # motor frio tem correção de aquecimento ativa
LAMBDA_VALIDO = (0.6, 1.6)  # fora disso é sonda saturada/corte, não mistura real
SETTLE_APOS_TRANSITORIO = 1.5   # s de descarte total depois de corte/AE/etc
SETTLE_RAMPA = 1.5              # s adicionais subindo o peso de 0 a 1
TPS_MOVIMENTO = 1.0         # %/s acima disso conta como "mexeu no acelerador"
RPMDOT_ESCALA = 150.0       # rpm/s onde o peso cai a ~37%
MAP_JANELA_S = 1.0          # janela pra medir estabilidade do MAP
MAP_ESTAVEL = 2.0           # kPa de variação na janela onde o peso cai a ~37%

# Atraso da sonda: não é constante, escala com o inverso do fluxo de escape.
# Ancorado em 0,8 s a 1800 rpm / 45 kPa (valor que o ve_map_optimizer estimou
# pelo DFCO nestes mesmos logs).
DELAY_REF_S = 0.8
DELAY_REF_FLUXO = 1800 * 45
DELAY_MIN_S, DELAY_MAX_S = 0.25, 2.0

# Bits do byte "Engine Status" (offset 2 do bloco de status — ver
# SpeeduinoManager.java). Só existem nos logs gerados depois desse decode.
BIT_RUNNING, BIT_CRANK, BIT_ASE, BIT_WARMUP = 0, 1, 2, 3
BIT_AE_TPS, BIT_DECEL_TPS, BIT_AE_MAP, BIT_DECEL_MAP = 4, 5, 6, 7


# ------------------------------------------------------------------- leitura
def ler_msl(caminho):
    """Devolve (cabecalho_3_linhas, nomes, unidades, linhas_dict)."""
    with open(caminho, encoding='utf-8', errors='replace') as f:
        linhas = f.read().splitlines()
    cab = linhas[:3]
    nomes = [n.strip().strip('"') for n in linhas[3].split('\t')]
    unid = linhas[4].split('\t')
    dados = []
    for ln in linhas[5:]:
        p = ln.split('\t')
        if len(p) < len(nomes):
            continue
        dados.append(dict(zip(nomes, p)))
    return cab, nomes, unid, dados


def num(v, padrao=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return padrao


def ler_bins_tune(caminho_msq):
    """Extrai fuelLoadBins e rpmBins do .msq do TunerStudio."""
    if not caminho_msq or not os.path.exists(caminho_msq):
        return None, None
    txt = open(caminho_msq, encoding='utf-8', errors='replace').read()

    def pega(nome):
        m = re.search(r'name="%s"[^>]*>(.*?)</constant>' % nome, txt, re.S)
        if not m:
            return None
        return [float(x) for x in m.group(1).split()]

    return pega('fuelLoadBins'), pega('rpmBins')


def ler_resolucao_map(caminho_msq):
    """kPa por contagem de ADC, a partir da calibração do sensor na tune.
    É o que define quão perto dois bins de carga podem ficar sem virarem a
    mesma leitura — muda de carro pra carro conforme o sensor."""
    if not caminho_msq or not os.path.exists(caminho_msq):
        return None
    txt = open(caminho_msq, encoding='utf-8', errors='replace').read()

    def esc(nome):
        m = re.search(r'name="%s"[^>]*>\s*([-\d.]+)' % nome, txt)
        return float(m.group(1)) if m else None

    mn, mx = esc('mapMin'), esc('mapMax')
    if mn is None or mx is None:
        return None
    return (mx - mn) / 1023.0  # ADC de 10 bits


# ---------------------------------------------------------------- derivados
def preparar(dados):
    """Anota cada amostra com o que o filtro precisa: flags, estabilidade,
    tempo desde o último transitório e lambda já casado com o atraso."""
    tem_flags = 'Engine Status' in dados[0] if dados else False
    tem_alvo = 'Lambda Target' in dados[0] if dados else False

    for i, r in enumerate(dados):
        r['_t'] = num(r.get('Time'), 0.0)
        r['_rpm'] = num(r.get('RPM'))
        r['_map'] = num(r.get('MAP'))
        r['_tps'] = num(r.get('TPS'))
        r['_clt'] = num(r.get('CLT'))
        r['_lam'] = num(r.get('Lambda'))
        r['_alvo'] = num(r.get('Lambda Target')) if tem_alvo else None
        r['_ve'] = num(r.get('VE _Current'))
        r['_gammae'] = num(r.get('GammaE'))
        r['_rpmdot'] = num(r.get('RPMdot'), 0.0)
        r['_tpsdot'] = num(r.get('TPSdot'), 0.0)

        # --- transitório ativo? preferir as flags da ECU; sem elas, heurística
        est = int(num(r.get('Engine Status'), 0) or 0) if tem_flags else None
        dfco_flag = num(r.get('DFCO'))
        ae_pct = num(r.get('Accel Enrich'))

        motivos = []
        if tem_flags and est is not None:
            if not (est >> BIT_RUNNING) & 1:
                motivos.append('parado')
            if (est >> BIT_CRANK) & 1:
                motivos.append('partida')
            if (est >> BIT_ASE) & 1:
                motivos.append('ASE')
            if (est >> BIT_WARMUP) & 1:
                motivos.append('warmup')
            if (est >> BIT_AE_TPS) & 1 or (est >> BIT_AE_MAP) & 1:
                motivos.append('AE')
            if (est >> BIT_DECEL_TPS) & 1 or (est >> BIT_DECEL_MAP) & 1:
                motivos.append('enleanment')
        if dfco_flag is not None and dfco_flag >= 1:
            motivos.append('DFCO')
        elif dfco_flag is None and r['_gammae'] == 0:
            # log antigo, sem a flag: GammaE==0 é o marcador de corte
            motivos.append('DFCO(gammae)')
        if ae_pct is not None and abs(ae_pct - 100) > 1:
            motivos.append('AE(pct)')

        r['_transitorio'] = motivos

    # --- tempo desde o último transitório e desde o último movimento de TPS
    ult_trans = -1e9
    ult_tps = -1e9
    for r in dados:
        if r['_transitorio']:
            ult_trans = r['_t']
        if abs(r['_tpsdot']) > TPS_MOVIMENTO:
            ult_tps = r['_t']
        r['_desde_trans'] = r['_t'] - ult_trans
        r['_desde_tps'] = r['_t'] - ult_tps

    # --- estabilidade do MAP numa janela, e tendência (subindo/descendo)
    j = 0
    for i, r in enumerate(dados):
        while dados[j]['_t'] < r['_t'] - MAP_JANELA_S:
            j += 1
        jan = [x['_map'] for x in dados[j:i + 1] if x['_map'] is not None]
        if len(jan) >= 2:
            r['_map_var'] = max(jan) - min(jan)
            r['_map_tend'] = jan[-1] - jan[0]
        else:
            r['_map_var'], r['_map_tend'] = 0.0, 0.0

    # --- lambda casado com o atraso da sonda (lambda de agora reflete o que
    #     queimou antes; então a condição de AGORA casa com lambda do FUTURO)
    tempos = [r['_t'] for r in dados]
    k = 0
    for i, r in enumerate(dados):
        fluxo = (r['_rpm'] or 0) * (r['_map'] or 0)
        atraso = DELAY_REF_S * (DELAY_REF_FLUXO / fluxo) if fluxo > 0 else DELAY_MAX_S
        atraso = min(max(atraso, DELAY_MIN_S), DELAY_MAX_S)
        r['_atraso'] = atraso
        alvo_t = r['_t'] + atraso
        while k < len(tempos) - 1 and tempos[k] < alvo_t:
            k += 1
        kk = max(0, min(k, len(dados) - 1))
        r['_lam_casado'] = dados[kk]['_lam']
        r['_idx_lam'] = kk

    return dados


# ------------------------------------------------------------------- filtro
def avaliar(r):
    """Devolve (peso, motivo_do_descarte). peso 0 = descartada."""
    if r['_rpm'] is None or r['_map'] is None or r['_lam_casado'] is None:
        return 0.0, 'sem dados'
    if r['_clt'] is not None and r['_clt'] < CLT_MIN:
        return 0.0, 'motor frio'
    lam = r['_lam_casado']
    if not (LAMBDA_VALIDO[0] <= lam <= LAMBDA_VALIDO[1]):
        return 0.0, 'lambda fora de faixa'
    if r['_transitorio']:
        return 0.0, 'transitório: ' + '+'.join(r['_transitorio'])
    if r['_desde_trans'] < SETTLE_APOS_TRANSITORIO:
        return 0.0, 'assentando pós-transitório'
    if r['_desde_tps'] < SETTLE_APOS_TRANSITORIO:
        return 0.0, 'assentando pós-TPS'

    # peso contínuo: quanto mais parado e mais tempo assentado, maior
    w_rpm = math.exp(-(abs(r['_rpmdot']) / RPMDOT_ESCALA) ** 2)
    w_map = math.exp(-(r['_map_var'] / MAP_ESTAVEL) ** 2)
    folga = min(r['_desde_trans'], r['_desde_tps']) - SETTLE_APOS_TRANSITORIO
    w_set = min(1.0, folga / SETTLE_RAMPA) if SETTLE_RAMPA > 0 else 1.0
    return w_rpm * w_map * w_set, None


# ------------------------------------------------------------------ relatórios
def bin_de(v, bins):
    """Índice do bin (com clamp nas pontas), estilo tabela do TunerStudio."""
    if v is None or not bins:
        return None
    if v <= bins[0]:
        return 0
    for i in range(len(bins) - 1):
        if bins[i] <= v < bins[i + 1]:
            return i
    return len(bins) - 1


def relatorio_limiar(dados):
    print('\n--- quanto sobra por limiar de peso ---')
    pesos = [r['_peso'] for r in dados if r['_peso'] > 0]
    total = len(dados)
    for lim in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        n = sum(1 for p in pesos if p >= lim)
        print(f'   peso >= {lim:.1f}: {n:6d} amostras ({100*n/total:5.1f}% do log)')


def relatorio_celulas(dados, load_bins, rpm_bins, limiar):
    if not load_bins or not rpm_bins:
        print('\n(sem .msq: pulando cobertura por célula)')
        return
    print(f'\n--- cobertura por célula da tabela real (peso >= {limiar}) ---')
    cel = defaultdict(lambda: [0, 0.0])
    for r in dados:
        if r['_peso'] < limiar:
            continue
        ir = bin_de(r['_rpm'], rpm_bins)
        il = bin_de(r['_map'], load_bins)
        if ir is None or il is None:
            continue
        cel[(ir, il)][0] += 1
        cel[(ir, il)][1] += r['_peso']
    print(f'   células com ao menos 1 amostra: {len(cel)} de {len(rpm_bins)*len(load_bins)}')
    fortes = sorted(cel.items(), key=lambda kv: -kv[1][1])[:15]
    print('   15 células mais amostradas (rpm / kPa : n, peso somado):')
    for (ir, il), (n, w) in fortes:
        print(f'      {rpm_bins[ir]:5.0f} / {load_bins[il]:5.0f} : {n:5d}, {w:7.1f}')
    fracas = [(k, v) for k, v in cel.items() if v[1] < 5]
    print(f'   células com peso somado < 5 (pouco confiáveis): {len(fracas)}')


def relatorio_bins(dados, load_bins, res_map=None):
    """O 'conselheiro de bins' — serve pra qualquer carro, é só o log."""
    print('\n--- conselho de bins de carga (a partir deste log) ---')
    inj = [r['_map'] for r in dados if not r['_transitorio'] and r['_map'] is not None]
    if not inj:
        print('   sem amostras injetando')
        return
    s = sorted(inj)
    q = lambda p: s[int(p * (len(s) - 1))]
    print(f'   MAP injetando: min={s[0]:.0f}  p1={q(.01):.0f}  p5={q(.05):.0f}  '
          f'mediana={q(.5):.0f}  p95={q(.95):.0f}  max={s[-1]:.0f}')
    print(f'   >> primeiro bin sugerido: {q(.01):.0f} kPa '
          f'(p1 das amostras injetando — abaixo disso é essencialmente corte)')
    if load_bins:
        abaixo = sum(1 for v in inj if v < load_bins[0])
        if abaixo:
            print(f'   !! {abaixo} amostras ({100*abaixo/len(inj):.1f}%) estão ABAIXO do '
                  f'primeiro bin atual ({load_bins[0]:.0f}) — todas grampeiam na 1a linha')
        # bins separados por menos do que o sensor consegue distinguir
        if res_map:
            print(f'   resolução do sensor MAP desta tune: {res_map:.2f} kPa por contagem de ADC')
            for i in range(len(load_bins) - 1):
                d = load_bins[i+1] - load_bins[i]
                if d < 4 * res_map:
                    print(f'   !! bins {load_bins[i]:.0f} e {load_bins[i+1]:.0f} a {d:.0f} kPa '
                          f'(~{d/res_map:.1f} contagens de ADC) — fino demais pra resolver')
    # sugestão por quantis: bins que dividem o tempo de uso em partes iguais
    n_bins = len(load_bins) if load_bins else 16
    sug = sorted({round(q(i / n_bins)) for i in range(n_bins)})
    print(f'   bins por uso igual ({n_bins} faixas): {sug}')


def teste_subida_descida(dados, load_bins, rpm_bins, limiar):
    """Mesma célula, carga subindo x descendo: a mistura difere?"""
    print(f'\n--- subida x descida na MESMA célula (peso >= {limiar}) ---')
    grupos = defaultdict(lambda: {'sobe': [], 'desce': [], 'plano': []})
    usaveis = 0
    for r in dados:
        if r['_peso'] < limiar or r['_alvo'] in (None, 0) or r['_lam_casado'] is None:
            continue
        ir = bin_de(r['_rpm'], rpm_bins) if rpm_bins else int((r['_rpm'] or 0) // 500)
        il = bin_de(r['_map'], load_bins) if load_bins else int((r['_map'] or 0) // 10)
        erro = (r['_lam_casado'] - r['_alvo']) / r['_alvo'] * 100
        t = r['_map_tend']
        cls = 'sobe' if t > 0.5 else ('desce' if t < -0.5 else 'plano')
        grupos[(ir, il)][cls].append(erro)
        usaveis += 1
    print(f'   amostras com alvo de lambda disponível: {usaveis}')
    linhas = []
    for (ir, il), g in grupos.items():
        if len(g['sobe']) >= 20 and len(g['desce']) >= 20:
            ms = sum(g['sobe']) / len(g['sobe'])
            md = sum(g['desce']) / len(g['desce'])
            linhas.append((abs(ms - md), ir, il, ms, md, len(g['sobe']), len(g['desce'])))
    if not linhas:
        print('   nenhuma célula tem >=20 amostras dos dois lados — sem conclusão')
        return
    linhas.sort(reverse=True)
    print('   célula (rpm/kPa) | erro% subindo | erro% descendo | diferença | n')
    difs = []
    for d, ir, il, ms, md, ns, nd in linhas[:12]:
        rl = rpm_bins[ir] if rpm_bins else ir * 500
        ll = load_bins[il] if load_bins else il * 10
        print(f'      {rl:5.0f}/{ll:4.0f} | {ms:+8.2f}% | {md:+9.2f}% | {ms-md:+8.2f}pp | {ns}/{nd}')
        difs.append(ms - md)
    med = sum(difs) / len(difs)
    print(f'   >> diferença média (subindo menos descendo): {med:+.2f} pontos percentuais')
    print('      positivo = fica mais POBRE subindo; negativo = mais RICO subindo')


# -------------------------------------------------------------------- saída
def escrever_msl(caminho, cab, nomes, unid, dados, limiar):
    aceitos = [r for r in dados if r['_peso'] >= limiar]
    with open(caminho, 'w', encoding='utf-8') as f:
        for l in cab:
            f.write(l + '\n')
        f.write('\t'.join(nomes + ['FilterWeight']) + '\n')
        f.write('\t'.join(list(unid) + ['']) + '\n')
        for r in aceitos:
            f.write('\t'.join([r.get(n, '') for n in nomes] + [f"{r['_peso']:.3f}"]) + '\n')
    return len(aceitos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--tune', default='/var/rcaldas/docs/tuner/Omega_3.6_0.3/CurrentTune.msq')
    ap.add_argument('--min-peso', type=float, default=0.3)
    ap.add_argument('--saida', default='.')
    args = ap.parse_args()

    load_bins, rpm_bins = ler_bins_tune(args.tune)
    res_map = ler_resolucao_map(args.tune)
    if load_bins:
        print(f'bins de carga da tune: {[int(b) for b in load_bins]}')

    for caminho in args.logs:
        print('\n' + '=' * 72)
        print(f'LOG: {caminho}')
        cab, nomes, unid, dados = ler_msl(caminho)
        if not dados:
            print('  vazio'); continue
        tem_flags = 'Engine Status' in dados[0]
        print(f'  {len(dados)} amostras | flags da ECU: {"sim" if tem_flags else "não (log antigo, usando GammaE)"}')

        dados = preparar(dados)
        motivos = Counter()
        for r in dados:
            r['_peso'], mot = avaliar(r)
            if mot:
                motivos[mot] += 1

        print('\n--- por que amostras foram descartadas ---')
        for m, n in motivos.most_common(10):
            print(f'   {n:6d}  {m}')

        relatorio_limiar(dados)
        relatorio_celulas(dados, load_bins, rpm_bins, args.min_peso)
        relatorio_bins(dados, load_bins, res_map)
        teste_subida_descida(dados, load_bins, rpm_bins, args.min_peso)

        base = os.path.basename(caminho).replace('.msl', '')
        saida = os.path.join(args.saida, f'{base}_filtrado.msl')
        n = escrever_msl(saida, cab, nomes, unid, dados, args.min_peso)
        print(f'\n>> gerado: {saida}  ({n} amostras com peso >= {args.min_peso})')


if __name__ == '__main__':
    main()
