#!/usr/bin/env python3
"""
Conversor e Analisador de Logs MLG do TunerStudio
Versão melhorada com filtros de qualidade para análise de lambda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xml.etree.ElementTree as ET
from scipy import interpolate, signal
import struct
import warnings
warnings.filterwarnings('ignore')

class MLGLogReader:
    """Leitor de arquivos MLG (MegaLogViewer) do TunerStudio"""
    
    def __init__(self):
        self.header_info = {}
        self.channels = []
        self.data = None
        
    def read_mlg_file(self, file_path):
        """Lê arquivo MLG binário e extrai dados"""
        try:
            print(f"📂 Lendo arquivo MLG: {file_path}")
            
            with open(file_path, 'rb') as f:
                # Lê header MLG
                header = f.read(6)
                if header[:5] != b'MLVLG':
                    print("❌ Arquivo não é um MLG válido")
                    return None
                
                # Pula para área de dados (offset aproximado)
                # MLG tem estrutura complexa, vamos usar uma abordagem simples
                f.seek(0)
                content = f.read()
                
                # Procura por padrões de dados conhecidos
                data_start = self._find_data_section(content)
                if data_start == -1:
                    print("❌ Não foi possível encontrar seção de dados")
                    return None
                
                # Extrai dados usando heurística
                df = self._extract_data_heuristic(content, data_start)
                
                if df is not None:
                    print(f"✅ Dados extraídos: {len(df)} registros")
                    return df
                else:
                    print("❌ Falha na extração de dados")
                    return None
                    
        except Exception as e:
            print(f"❌ Erro ao ler MLG: {e}")
            return None
    
    def _find_data_section(self, content):
        """Encontra início da seção de dados numéricos"""
        # Procura por padrões que indicam início de dados
        patterns = [b'Time', b'RPM', b'Lambda', b'TPS']
        
        for pattern in patterns:
            pos = content.find(pattern)
            if pos != -1:
                # Procura início de dados numéricos após os headers
                for i in range(pos, min(pos + 10000, len(content) - 4), 4):
                    # Tenta ler como float
                    try:
                        val = struct.unpack('<f', content[i:i+4])[0]
                        if 500 < val < 8000:  # Possível RPM
                            return i
                    except:
                        continue
        
        return -1
    
    def _extract_data_heuristic(self, content, start_pos):
        """Extrai dados usando heurística baseada em valores típicos"""
        try:
            # Estima número de canais baseado no arquivo de exemplo
            estimated_channels = 14  # Baseado no example.log
            
            # Calcula número de registros
            remaining_bytes = len(content) - start_pos
            record_size = estimated_channels * 4  # 4 bytes por float
            num_records = remaining_bytes // record_size
            
            if num_records < 100:  # Muito poucos dados
                return None
            
            # Limita para evitar problemas de memória
            num_records = min(num_records, 50000)
            
            print(f"📊 Estimativa: {estimated_channels} canais, {num_records} registros")
            
            # Extrai dados como floats
            data_matrix = []
            pos = start_pos
            
            for i in range(num_records):
                if pos + record_size > len(content):
                    break
                
                record = []
                for j in range(estimated_channels):
                    if pos + 4 <= len(content):
                        try:
                            val = struct.unpack('<f', content[pos:pos+4])[0]
                            record.append(val)
                            pos += 4
                        except:
                            record.append(np.nan)
                            pos += 4
                    else:
                        break
                
                if len(record) == estimated_channels:
                    data_matrix.append(record)
            
            # Cria DataFrame com nomes estimados
            columns = ['RPM', 'COOLANT_TEMP', 'MAF', 'THROTTLE_POS', 'INTAKE_TEMP', 
                      'TIMING_ADVANCE', 'ENGINE_LOAD', 'ELM_VOLTAGE', 'SPEED', 
                      'O2_S1_WR_CURRENT', 'O2_S5_WR_CURRENT', 'O2_B2S2', 
                      'SHORT_FUEL_TRIM_1', 'SHORT_FUEL_TRIM_2']
            
            df = pd.DataFrame(data_matrix, columns=columns[:len(data_matrix[0])])
            
            # Filtra valores absurdos
            df = self._clean_extracted_data(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Erro na extração heurística: {e}")
            return None
    
    def _clean_extracted_data(self, df):
        """Limpa dados extraídos removendo valores absurdos"""
        
        # Define ranges válidos
        valid_ranges = {
            'RPM': (300, 8000),
            'COOLANT_TEMP': (0, 150),
            'THROTTLE_POS': (0, 100),
            'INTAKE_TEMP': (-20, 80),
            'TIMING_ADVANCE': (-10, 60),
            'ENGINE_LOAD': (0, 150),
            'ELM_VOLTAGE': (10, 16),
            'SPEED': (0, 300),
            'O2_B2S2': (0.4, 1.5)  # AFR como lambda aproximado
        }
        
        initial_count = len(df)
        
        for col, (min_val, max_val) in valid_ranges.items():
            if col in df.columns:
                mask = (df[col] >= min_val) & (df[col] <= max_val)
                df = df[mask]
        
        print(f"🧹 Limpeza: {initial_count} -> {len(df)} registros ({len(df)/initial_count*100:.1f}% retidos)")
        
        return df.reset_index(drop=True)

class AdvancedVEOptimizer:
    """Otimizador VE avançado com filtros de qualidade"""
    
    def __init__(self):
        self.engine_specs = {
            'displacement': 2000,
            'cylinders': 4,
            'injectors': 4,
            'injector_flow': 19,
            'injection_mode': 'bank_to_bank',
            'stoich_afr': 14.7
        }
        
        self.quality_filters = {
            'tps_stability_threshold': 2.0,    # %/s - mudança máxima TPS para estabilidade
            'tps_stability_window': 10,        # amostras para avaliar estabilidade
            'min_pw_for_lambda': 1.0,         # ms - PW mínimo para leitura lambda válida
            'lambda_valid_range': (0.7, 1.3), # Range válido de lambda
            'min_coolant_temp': 75,           # °C - motor aquecido
            'voltage_range': (11.0, 15.0)    # V - voltagem estável
        }
        
    def load_and_process_log(self, file_path):
        """Carrega e processa log com filtros de qualidade"""
        
        # Determina tipo de arquivo e carrega
        if file_path.endswith('.mlg'):
            reader = MLGLogReader()
            df = reader.read_mlg_file(file_path)
        else:
            # Arquivo CSV/texto
            df = self._load_text_log(file_path)
        
        if df is None:
            return None
        
        # Processa dados
        df = self._add_calculated_fields(df)
        df = self._apply_quality_filters(df)
        
        return df
    
    def _load_text_log(self, file_path):
        """Carrega log em formato texto/CSV"""
        try:
            columns = ['RPM', 'COOLANT_TEMP', 'MAF', 'THROTTLE_POS', 'INTAKE_TEMP', 
                      'TIMING_ADVANCE', 'ENGINE_LOAD', 'ELM_VOLTAGE', 'SPEED', 
                      'O2_S1_WR_CURRENT', 'O2_S5_WR_CURRENT', 'O2_B2S2', 
                      'SHORT_FUEL_TRIM_1', 'SHORT_FUEL_TRIM_2']
            
            df = pd.read_csv(file_path, names=columns)
            print(f"📂 Log texto carregado: {len(df)} registros")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar log texto: {e}")
            return None
    
    def _add_calculated_fields(self, df):
        """Adiciona campos calculados"""
        
        # Timestamp (assumindo 10Hz)
        df['Time'] = np.arange(len(df)) * 0.1
        
        # Lambda a partir do AFR
        df['Lambda'] = df['O2_B2S2'] / self.engine_specs['stoich_afr']
        
        # FuelLoad (usando ENGINE_LOAD como proxy)
        df['FuelLoad'] = df['ENGINE_LOAD']
        
        # Taxa de mudança do TPS (suavizada)
        df['TPS_Rate'] = df['THROTTLE_POS'].diff() / 0.1  # %/s
        df['TPS_Rate'] = df['TPS_Rate'].rolling(window=3, center=True).mean()
        df['TPS_Rate'] = df['TPS_Rate'].fillna(0)
        
        # Estimativa de PW (pulso dos bicos) - simplificada
        # PW ≈ (VE × Load × RPM × Displacement) / (Injector_Flow × AFR × constant)
        df['PW_Estimated'] = (df['ENGINE_LOAD'] * df['RPM'] * 2.0) / (19 * 14.7 * 10000)
        df['PW_Estimated'] = np.clip(df['PW_Estimated'], 0, 20)  # Limita a range razoável
        
        # Detecta fuel cut (DFCO)
        df['Is_FuelCut'] = (df['PW_Estimated'] < self.quality_filters['min_pw_for_lambda']) | \
                          (df['Lambda'] > 1.1)  # Lambda muito alto indica corte
        
        print(f"📊 Campos calculados adicionados")
        return df
    
    def _apply_quality_filters(self, df):
        """Aplica filtros de qualidade para análise de lambda"""
        
        print("🔍 Aplicando filtros de qualidade...")
        
        initial_count = len(df)
        
        # 1. Filtro básico de qualidade
        basic_mask = (
            (df['COOLANT_TEMP'] >= self.quality_filters['min_coolant_temp']) &
            (df['ELM_VOLTAGE'] >= self.quality_filters['voltage_range'][0]) &
            (df['ELM_VOLTAGE'] <= self.quality_filters['voltage_range'][1]) &
            (df['RPM'] > 600) & (df['RPM'] < 7000) &
            (df['Lambda'] >= self.quality_filters['lambda_valid_range'][0]) &
            (df['Lambda'] <= self.quality_filters['lambda_valid_range'][1])
        )
        
        df_filtered = df[basic_mask].copy().reset_index(drop=True)
        print(f"   Filtro básico: {len(df_filtered)}/{initial_count} ({len(df_filtered)/initial_count*100:.1f}%)")
        
        # 2. Filtro de estabilidade do TPS
        df_filtered['TPS_Stable'] = self._calculate_tps_stability(df_filtered)
        
        stable_mask = df_filtered['TPS_Stable']
        df_stable = df_filtered[stable_mask].copy().reset_index(drop=True)
        print(f"   TPS estável: {len(df_stable)}/{len(df_filtered)} ({len(df_stable)/len(df_filtered)*100:.1f}%)")
        
        # 3. Filtro de fuel cut (exclui momentos de corte)
        no_cut_mask = ~df_stable['Is_FuelCut']
        df_final = df_stable[no_cut_mask].copy().reset_index(drop=True)
        print(f"   Sem fuel cut: {len(df_final)}/{len(df_stable)} ({len(df_final)/len(df_stable)*100:.1f}%)")
        
        # 4. Remove outliers de lambda (IQR method)
        Q1 = df_final['Lambda'].quantile(0.25)
        Q3 = df_final['Lambda'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df_final['Lambda'] >= lower_bound) & (df_final['Lambda'] <= upper_bound)
        df_clean = df_final[outlier_mask].copy().reset_index(drop=True)
        print(f"   Sem outliers: {len(df_clean)}/{len(df_final)} ({len(df_clean)/len(df_final)*100:.1f}%)")
        
        print(f"✅ Filtros aplicados: {initial_count} -> {len(df_clean)} registros finais ({len(df_clean)/initial_count*100:.1f}%)")
        
        return df_clean
    
    def _calculate_tps_stability(self, df):
        """Calcula se TPS está estável em cada ponto"""
        
        window = self.quality_filters['tps_stability_window']
        threshold = self.quality_filters['tps_stability_threshold']
        
        stability = np.zeros(len(df), dtype=bool)
        
        for i in range(window, len(df) - window):
            # Janela de análise centrada no ponto
            window_data = df['TPS_Rate'].iloc[i-window:i+window+1]
            
            # Verifica se todas as mudanças na janela são pequenas
            max_change = abs(window_data).max()
            
            # Considera estável se mudança máxima < threshold
            stability[i] = max_change < threshold
        
        return stability
    
    def visualize_quality_filters(self, df_original, df_filtered):
        """Visualiza efeito dos filtros de qualidade"""
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. TPS vs Tempo (original vs filtrado)
        axes[0,0].plot(df_original['Time'], df_original['THROTTLE_POS'], alpha=0.5, label='Original', color='red')
        if len(df_filtered) > 0:
            axes[0,0].plot(df_filtered['Time'], df_filtered['THROTTLE_POS'], alpha=0.8, label='Filtrado', color='blue')
        axes[0,0].set_xlabel('Tempo (s)')
        axes[0,0].set_ylabel('TPS (%)')
        axes[0,0].set_title('TPS vs Tempo')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Lambda vs Tempo
        axes[0,1].plot(df_original['Time'], df_original['Lambda'], alpha=0.5, label='Original', color='red')
        if len(df_filtered) > 0:
            axes[0,1].plot(df_filtered['Time'], df_filtered['Lambda'], alpha=0.8, label='Filtrado', color='blue')
        axes[0,1].set_xlabel('Tempo (s)')
        axes[0,1].set_ylabel('Lambda')
        axes[0,1].set_title('Lambda vs Tempo')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. TPS Rate (estabilidade)
        axes[1,0].plot(df_original['Time'], abs(df_original['TPS_Rate']), alpha=0.5, label='|TPS Rate|', color='orange')
        axes[1,0].axhline(y=self.quality_filters['tps_stability_threshold'], color='red', 
                         linestyle='--', label=f'Threshold ({self.quality_filters["tps_stability_threshold"]}%/s)')
        if len(df_filtered) > 0:
            stable_points = df_filtered[df_filtered['TPS_Stable']] if 'TPS_Stable' in df_filtered.columns else df_filtered
            axes[1,0].scatter(stable_points['Time'], abs(stable_points['TPS_Rate']), 
                             alpha=0.6, s=1, color='green', label='Pontos estáveis')
        axes[1,0].set_xlabel('Tempo (s)')
        axes[1,0].set_ylabel('|TPS Rate| (%/s)')
        axes[1,0].set_title('Estabilidade do TPS')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Fuel Cut Detection
        axes[1,1].plot(df_original['Time'], df_original['PW_Estimated'], alpha=0.5, label='PW Estimado', color='blue')
        axes[1,1].axhline(y=self.quality_filters['min_pw_for_lambda'], color='red', 
                         linestyle='--', label=f'Min PW ({self.quality_filters["min_pw_for_lambda"]} ms)')
        
        fuel_cut_points = df_original[df_original['Is_FuelCut']]
        if len(fuel_cut_points) > 0:
            axes[1,1].scatter(fuel_cut_points['Time'], fuel_cut_points['PW_Estimated'], 
                             alpha=0.6, s=5, color='red', label='Fuel Cut')
        
        axes[1,1].set_xlabel('Tempo (s)')
        axes[1,1].set_ylabel('PW Estimado (ms)')
        axes[1,1].set_title('Detecção de Fuel Cut')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 5. Distribuição Lambda (antes/depois)
        axes[2,0].hist(df_original['Lambda'], bins=50, alpha=0.5, label='Original', color='red', density=True)
        if len(df_filtered) > 0:
            axes[2,0].hist(df_filtered['Lambda'], bins=50, alpha=0.7, label='Filtrado', color='blue', density=True)
        axes[2,0].set_xlabel('Lambda')
        axes[2,0].set_ylabel('Densidade')
        axes[2,0].set_title('Distribuição Lambda')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        
        # 6. RPM vs Load (pontos válidos)
        axes[2,1].scatter(df_original['RPM'], df_original['FuelLoad'], alpha=0.3, s=1, 
                         label='Original', color='red')
        if len(df_filtered) > 0:
            axes[2,1].scatter(df_filtered['RPM'], df_filtered['FuelLoad'], alpha=0.6, s=1, 
                             label='Filtrado', color='blue')
        axes[2,1].set_xlabel('RPM')
        axes[2,1].set_ylabel('Fuel Load (kPa)')
        axes[2,1].set_title('Cobertura RPM vs Load')
        axes[2,1].legend()
        axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/robca/obd2/quality_filters_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Estatísticas
        print(f"\n📊 Estatísticas de qualidade:")
        print(f"   Dados originais: {len(df_original)} registros")
        print(f"   Dados filtrados: {len(df_filtered)} registros")
        print(f"   Taxa de retenção: {len(df_filtered)/len(df_original)*100:.1f}%")
        
        if len(df_filtered) > 0:
            print(f"\n📈 Estatísticas dos dados filtrados:")
            print(f"   Lambda: {df_filtered['Lambda'].mean():.3f} ± {df_filtered['Lambda'].std():.3f}")
            print(f"   RPM: {df_filtered['RPM'].mean():.0f} ± {df_filtered['RPM'].std():.0f}")
            print(f"   Load: {df_filtered['FuelLoad'].mean():.1f} ± {df_filtered['FuelLoad'].std():.1f}")
            print(f"   TPS: {df_filtered['THROTTLE_POS'].mean():.1f} ± {df_filtered['THROTTLE_POS'].std():.1f}")

def main():
    """Função principal para testar o sistema"""
    
    print("🚗 Analisador Avançado de Logs VE - Versão com Filtros de Qualidade")
    print("="*70)
    
    # Inicializa otimizador
    optimizer = AdvancedVEOptimizer()
    
    # Lista arquivos disponíveis
    import os
    log_dir = "/home/robca/obd2/logs"
    log_files = [f for f in os.listdir(log_dir) if f.endswith(('.mlg', '.log'))]
    
    print(f"\n📁 Arquivos de log disponíveis:")
    for i, file in enumerate(log_files, 1):
        size = os.path.getsize(os.path.join(log_dir, file)) / (1024*1024)
        print(f"   {i}. {file} ({size:.1f} MB)")
    
    # Testa com diferentes arquivos
    test_files = [
        "2025-08-06_00.09.39.mlg",  # Arquivo maior
        "example.log"               # Arquivo texto de referência
    ]
    
    for file_name in test_files:
        file_path = os.path.join(log_dir, file_name)
        if os.path.exists(file_path):
            print(f"\n" + "="*50)
            print(f"🔍 Analisando: {file_name}")
            print("="*50)
            
            # Carrega dados originais para comparação
            if file_name.endswith('.mlg'):
                reader = MLGLogReader()
                df_raw = reader.read_mlg_file(file_path)
            else:
                try:
                    columns = ['RPM', 'COOLANT_TEMP', 'MAF', 'THROTTLE_POS', 'INTAKE_TEMP', 
                              'TIMING_ADVANCE', 'ENGINE_LOAD', 'ELM_VOLTAGE', 'SPEED', 
                              'O2_S1_WR_CURRENT', 'O2_S5_WR_CURRENT', 'O2_B2S2', 
                              'SHORT_FUEL_TRIM_1', 'SHORT_FUEL_TRIM_2']
                    df_raw = pd.read_csv(file_path, names=columns)
                    df_raw['Time'] = np.arange(len(df_raw)) * 0.1
                    df_raw['Lambda'] = df_raw['O2_B2S2'] / 14.7
                    df_raw['FuelLoad'] = df_raw['ENGINE_LOAD']
                    df_raw['TPS_Rate'] = df_raw['THROTTLE_POS'].diff() / 0.1
                    df_raw['TPS_Rate'] = df_raw['TPS_Rate'].fillna(0)
                    df_raw['PW_Estimated'] = (df_raw['ENGINE_LOAD'] * df_raw['RPM'] * 2.0) / (19 * 14.7 * 10000)
                    df_raw['Is_FuelCut'] = (df_raw['PW_Estimated'] < 1.0) | (df_raw['Lambda'] > 1.1)
                except:
                    continue
            
            if df_raw is not None:
                # Processa com filtros
                df_processed = optimizer.load_and_process_log(file_path)
                
                if df_processed is not None:
                    # Visualiza resultados dos filtros
                    optimizer.visualize_quality_filters(df_raw, df_processed)
                else:
                    print("❌ Falha no processamento")
            else:
                print("❌ Falha na leitura do arquivo")

if __name__ == "__main__":
    main()
