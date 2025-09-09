#!/usr/bin/env python3
"""
Script para sincronizar logs da injeção original (CSV) com logs da injeção programável (MSL)
e calcular valores de lambda baseados no sinal da sonda O2_S5_WR_CURRENT.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import argparse
import os

def convert_current_to_lambda(current_amperes):
    """
    Converte corrente da sonda lambda (A) para valor lambda
    Sinal positivo = mistura pobre (lambda > 1)
    Sinal negativo = mistura rica (lambda < 1)
    
    Baseado na calibração específica da sonda:
    - Corrente mais negativa (-1.285A) → Lambda 0.70
    - 0 A → Lambda 1.0 (mistura estequiométrica)
    - Corrente mais positiva (0.895A) → Lambda 1.95
    
    Conversão: A → mA → Lambda
    """
    # Converte Amperes para miliamperes
    current_ma = current_amperes * 1000.0
    
    if current_ma < 0:
        # Mistura rica: -1285mA → 0.70 lambda (diferença de -0.30)
        lambda_val = 1.0 + (current_ma * 0.30 / 1285.0)
    else:
        # Mistura pobre: +895mA → 1.95 lambda (diferença de +0.95)
        lambda_val = 1.0 + (current_ma * 0.95 / 895.0)
    
    # Limitar valores dentro da faixa física possível
    return max(0.60, min(2.0, lambda_val))

def load_csv_log(csv_file):
    """Carrega e processa o log CSV da injeção original."""
    print(f"Carregando arquivo CSV: {csv_file}")
    
    # Carrega o CSV
    df = pd.read_csv(csv_file)
    
    # Converte timestamp para datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Remove linhas com RPM vazio ou zero
    df = df.dropna(subset=['RPM'])
    df = df[df['RPM'] > 0]
    
    # Calcula lambda a partir do sinal O2_S5_WR_CURRENT
    df['lambda_calculated'] = df['O2_S5_WR_CURRENT'].apply(convert_current_to_lambda)
    
    print(f"CSV carregado: {len(df)} registros")
    print(f"Faixa de RPM: {df['RPM'].min():.0f} - {df['RPM'].max():.0f}")
    print(f"Período: {df['datetime'].min()} - {df['datetime'].max()}")
    
    return df

def load_msl_log(msl_file):
    """Carrega e processa o log MSL da injeção programável."""
    print(f"\nCarregando arquivo MSL: {msl_file}")
    
    # Lê o arquivo MSL pulando as primeiras linhas de cabeçalho
    with open(msl_file, 'r') as f:
        lines = f.readlines()
    
    # Encontra a linha de cabeçalho com as colunas
    header_line = None
    data_start = None
    
    for i, line in enumerate(lines):
        if line.startswith('Time\t'):
            header_line = i
            data_start = i + 2  # Pula a linha de unidades
            break
    
    if header_line is None:
        raise ValueError("Não foi possível encontrar o cabeçalho no arquivo MSL")
    
    # Extrai colunas do cabeçalho
    headers = lines[header_line].strip().split('\t')
    
    # Carrega dados
    df = pd.read_csv(msl_file, sep='\t', skiprows=data_start, names=headers, low_memory=False)
    
    # Converte colunas numéricas importantes
    numeric_cols = ['Time', 'RPM', 'Lambda', 'AFR', 'MAP', 'TPS']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove linhas com RPM vazio ou zero
    df = df.dropna(subset=['RPM'])
    df = df[df['RPM'] > 0]
    
    # Remove linhas com Time vazio
    df = df.dropna(subset=['Time'])
    
    print(f"MSL carregado: {len(df)} registros")
    print(f"Faixa de RPM: {df['RPM'].min():.0f} - {df['RPM'].max():.0f}")
    print(f"Período: {df['Time'].min():.1f}s - {df['Time'].max():.1f}s")
    
    return df

def synchronize_logs(csv_df, msl_df, rpm_tolerance=50):
    """
    Sincroniza os logs usando RPM como referência.
    
    Args:
        csv_df: DataFrame do log CSV
        msl_df: DataFrame do log MSL
        rpm_tolerance: Tolerância em RPM para considerar pontos próximos
        
    Returns:
        DataFrame sincronizado com dados de ambos os logs
    """
    print(f"\nSincronizando logs com tolerância de RPM: ±{rpm_tolerance}")
    
    synchronized_data = []
    
    # Para cada ponto no MSL, encontra o ponto correspondente no CSV
    for idx, msl_row in msl_df.iterrows():
        msl_rpm = msl_row['RPM']
        msl_time = msl_row['Time']
        
        # Encontra pontos no CSV com RPM similar
        csv_matches = csv_df[
            (csv_df['RPM'] >= msl_rpm - rpm_tolerance) & 
            (csv_df['RPM'] <= msl_rpm + rpm_tolerance)
        ]
        
        if len(csv_matches) > 0:
            # Pega o ponto mais próximo em RPM
            csv_match = csv_matches.iloc[
                (csv_matches['RPM'] - msl_rpm).abs().argsort().iloc[0]
            ]
            
            # Cria registro sincronizado
            sync_record = {
                'msl_time': msl_time,
                'csv_timestamp': csv_match['timestamp'],
                'csv_datetime': csv_match['datetime'],
                'rpm_msl': msl_rpm,
                'rpm_csv': csv_match['RPM'],
                'rpm_diff': abs(msl_rpm - csv_match['RPM']),
                'lambda_msl_original': msl_row.get('Lambda', np.nan),
                'lambda_csv_calculated': csv_match['lambda_calculated'],
                'o2_current': csv_match['O2_S5_WR_CURRENT'],
                'timing_advance': csv_match.get('TIMING_ADVANCE', np.nan),
                'afr_msl': msl_row.get('AFR', np.nan),
                'map_msl': msl_row.get('MAP', np.nan),
                'tps_msl': msl_row.get('TPS', np.nan)
            }
            
            synchronized_data.append(sync_record)
    
    sync_df = pd.DataFrame(synchronized_data)
    
    print(f"Pontos sincronizados: {len(sync_df)}")
    if len(sync_df) > 0:
        print(f"Diferença média de RPM: {sync_df['rpm_diff'].mean():.1f}")
        print(f"Diferença máxima de RPM: {sync_df['rpm_diff'].max():.1f}")
    
    return sync_df

def create_updated_msl(original_msl_file, sync_df, output_file):
    """
    Cria um novo arquivo MSL com os valores de lambda atualizados.
    
    Args:
        original_msl_file: Arquivo MSL original
        sync_df: DataFrame com dados sincronizados
        output_file: Arquivo MSL de saída
    """
    print(f"\nCriando arquivo MSL atualizado: {output_file}")
    
    # Lê o arquivo MSL original
    with open(original_msl_file, 'r') as f:
        lines = f.readlines()
    
    # Encontra onde começam os dados
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('Time\t'):
            data_start = i + 2
            break
    
    if data_start is None:
        raise ValueError("Não foi possível encontrar o início dos dados no MSL")
    
    # Carrega o MSL original como DataFrame
    headers = lines[data_start-2].strip().split('\t')
    msl_df = pd.read_csv(original_msl_file, sep='\t', skiprows=data_start, names=headers, low_memory=False)
    
    # Converte colunas numéricas importantes
    numeric_cols = ['Time', 'RPM', 'Lambda', 'AFR', 'MAP', 'TPS']
    for col in numeric_cols:
        if col in msl_df.columns:
            msl_df[col] = pd.to_numeric(msl_df[col], errors='coerce')
    
    # Cria interpolador para os valores de lambda calculados
    if len(sync_df) > 1:
        # Remove valores NaN do sync_df para interpolação
        sync_clean = sync_df.dropna(subset=['msl_time', 'lambda_csv_calculated'])
        
        if len(sync_clean) > 1:
            # Usa o tempo MSL como referência
            lambda_interp = interpolate.interp1d(
                sync_clean['msl_time'], 
                sync_clean['lambda_csv_calculated'],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # Atualiza valores de lambda no MSL
            msl_df['Lambda'] = lambda_interp(msl_df['Time'].fillna(0))
            
            # Também atualiza AFR (AFR = Lambda * 14.7 para gasolina)
            msl_df['AFR'] = msl_df['Lambda'] * 14.7
            
            print(f"Lambda atualizado para {len(msl_df)} registros")
        else:
            print("Não há dados suficientes para interpolação")
    
    # Escreve o arquivo atualizado
    with open(output_file, 'w') as f:
        # Copia cabeçalho original
        for i in range(data_start):
            f.write(lines[i])
        
        # Escreve dados atualizados
        for _, row in msl_df.iterrows():
            row_data = []
            for col in msl_df.columns:
                value = row[col]
                if pd.isna(value):
                    row_data.append('')
                elif isinstance(value, (int, float)):
                    if col in ['Lambda', 'AFR', 'Time']:
                        row_data.append(f"{value:.6f}")
                    else:
                        row_data.append(str(value))
                else:
                    row_data.append(str(value))
            f.write('\t'.join(row_data) + '\n')
    
    print(f"Arquivo MSL atualizado salvo: {output_file}")

def plot_comparison(sync_df, output_dir):
    """Cria gráficos comparativos dos dados sincronizados."""
    if len(sync_df) == 0:
        print("Nenhum dado sincronizado para plotar")
        return
    
    print(f"\nCriando gráficos comparativos...")
    
    # Configurar matplotlib
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparação de Logs Sincronizados', fontsize=14, fontweight='bold')
    
    # 1. Lambda vs RPM
    ax1 = axes[0, 0]
    ax1.scatter(sync_df['rpm_msl'], sync_df['lambda_msl_original'], 
               alpha=0.4, label='Lambda MSL (IGNORADO)', s=15, color='gray', marker='x')
    ax1.scatter(sync_df['rpm_msl'], sync_df['lambda_csv_calculated'], 
               alpha=0.8, label='Lambda CSV (CORRETO)', s=25, color='red', marker='o')
    ax1.set_xlabel('RPM')
    ax1.set_ylabel('Lambda')
    ax1.set_title('Lambda vs RPM - Comparação')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Estequiométrico')
    
    # 2. Sinal O2 vs RPM
    ax2 = axes[0, 1]
    scatter = ax2.scatter(sync_df['rpm_msl'], sync_df['o2_current'], 
                         c=sync_df['lambda_csv_calculated'], 
                         cmap='RdYlBu_r', alpha=0.6, s=20)
    ax2.set_xlabel('RPM')
    ax2.set_ylabel('Corrente O2 (A)')
    ax2.set_title('Sinal da Sonda vs RPM')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7)
    plt.colorbar(scatter, ax=ax2, label='Lambda')
    
    # 3. Comparação Lambda ao longo do tempo
    ax3 = axes[1, 0]
    ax3.plot(sync_df['msl_time'], sync_df['lambda_msl_original'], 
             'gray', alpha=0.4, label='MSL (IGNORADO)', linewidth=1, linestyle='--')
    ax3.plot(sync_df['msl_time'], sync_df['lambda_csv_calculated'], 
             'red', alpha=0.8, label='CSV (CORRETO)', linewidth=2)
    ax3.set_xlabel('Tempo MSL (s)')
    ax3.set_ylabel('Lambda')
    ax3.set_title('Lambda vs Tempo - Comparação')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
    
    # 4. Diferenças de RPM na sincronização
    ax4 = axes[1, 1]
    ax4.hist(sync_df['rpm_diff'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_xlabel('Diferença de RPM')
    ax4.set_ylabel('Frequência')
    ax4.set_title('Distribuição das Diferenças de RPM')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva o gráfico
    plot_file = os.path.join(output_dir, 'sync_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo: {plot_file}")
    
    # Mostra estatísticas
    print("\n" + "="*50)
    print("ESTATÍSTICAS DOS DADOS SINCRONIZADOS")
    print("="*50)
    print(f"Lambda CSV (CORRETO) - Min: {sync_df['lambda_csv_calculated'].min():.3f}, Max: {sync_df['lambda_csv_calculated'].max():.3f}, Média: {sync_df['lambda_csv_calculated'].mean():.3f}")
    if not sync_df['lambda_msl_original'].isna().all():
        print(f"Lambda MSL (IGNORADO) - Min: {sync_df['lambda_msl_original'].min():.3f}, Max: {sync_df['lambda_msl_original'].max():.3f}, Média: {sync_df['lambda_msl_original'].mean():.3f}")
    print(f"Corrente O2 - Min: {sync_df['o2_current'].min():.3f}A, Max: {sync_df['o2_current'].max():.3f}A")
    print("\nNOTA IMPORTANTE:")
    print("- Os valores ORIGINAIS de lambda do MSL foram IGNORADOS completamente")
    print("- Apenas os valores calculados da corrente da sonda (CSV) foram utilizados")
    print("- O arquivo MSL foi atualizado com os valores CORRETOS de lambda")

def main():
    parser = argparse.ArgumentParser(description='Sincroniza logs OBD2 (CSV) com logs de injeção programável (MSL)')
    parser.add_argument('csv_file', help='Arquivo CSV com dados OBD2')
    parser.add_argument('msl_file', help='Arquivo MSL da injeção programável')
    parser.add_argument('-o', '--output', help='Diretório de saída (padrão: mesmo do MSL)')
    parser.add_argument('-t', '--tolerance', type=int, default=50, 
                       help='Tolerância de RPM para sincronização (padrão: 50)')
    
    args = parser.parse_args()
    
    # Verifica se os arquivos existem
    if not os.path.exists(args.csv_file):
        print(f"Erro: Arquivo CSV não encontrado: {args.csv_file}")
        return 1
        
    if not os.path.exists(args.msl_file):
        print(f"Erro: Arquivo MSL não encontrado: {args.msl_file}")
        return 1
    
    # Define diretório de saída
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.dirname(args.msl_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Carrega os logs
        csv_df = load_csv_log(args.csv_file)
        msl_df = load_msl_log(args.msl_file)
        
        # Sincroniza os dados
        sync_df = synchronize_logs(csv_df, msl_df, args.tolerance)
        
        if len(sync_df) == 0:
            print("Erro: Nenhum ponto foi sincronizado. Verifique os dados ou aumente a tolerância de RPM.")
            return 1
        
        # Salva dados sincronizados
        sync_file = os.path.join(output_dir, 'synchronized_data.csv')
        sync_df.to_csv(sync_file, index=False)
        print(f"\nDados sincronizados salvos: {sync_file}")
        
        # Cria MSL atualizado
        base_name = os.path.splitext(os.path.basename(args.msl_file))[0]
        output_msl = os.path.join(output_dir, f"{base_name}_lambda_updated.msl")
        create_updated_msl(args.msl_file, sync_df, output_msl)
        
        # Cria gráficos
        plot_comparison(sync_df, output_dir)
        
        print(f"\nProcessamento concluído!")
        print(f"Arquivo MSL atualizado: {output_msl}")
        print(f"Dados sincronizados: {sync_file}")
        print(f"Gráficos: {output_dir}/sync_comparison.png")
        
    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
