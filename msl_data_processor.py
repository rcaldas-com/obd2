#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processador de dados MSL melhorado com análise de confiabilidade
Foca apenas em células com dados suficientes e condições estáveis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MSLDataProcessor:
    """
    Classe para processar dados do TunerStudio em formato .msl
    com foco em confiabilidade e qualidade dos dados
    """
    
    def __init__(self):
        # Especificações do motor
        self.engine_displacement = 2000  # cc
        self.cylinders = 4
        self.injectors = 4
        self.injector_flow = 18.27  # lb/h
        self.stoich_afr = 13.5  # AFR estequiométrico para gasolina com 30% de etanol
        
        # Dados carregados
        self.log_data = None
        self.ve_table = None
        self.lambda_table = None
        self.rpm_bins_ve = None
        self.load_bins_ve = None
        self.rpm_bins_lambda = None
        self.load_bins_lambda = None
        
        # Dados de confiabilidade
        self.cell_coverage = None  # Matriz de cobertura de dados por célula
        self.stable_data = None    # Dados filtrados para condições estáveis
        
        print("🚗 MSL Data Processor inicializado (Versão Confiabilidade)")
        print(f"   Motor: {self.engine_displacement}cc, {self.cylinders} cilindros")
        print(f"   Injetores: {self.injectors}x {self.injector_flow}lb/h")
        print(f"   Combustível: E30 (AFR estequio: {self.stoich_afr})")
    
    def load_msl_data(self, msl_file_path, sample_size=None):
        """
        Carrega dados do arquivo .msl (formato texto delimitado por tab)
        
        Args:
            msl_file_path: Caminho para o arquivo .msl
            sample_size: Número máximo de linhas para carregar (None = todas)
        """
        print(f"📂 Carregando dados MSL de: {msl_file_path}")
        
        try:
            # Lê o arquivo .msl
            with open(msl_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Encontra a linha de cabeçalho
            header_line = None
            data_start = 0
            
            for i, line in enumerate(lines):
                if line.startswith('Time\t'):
                    header_line = line.strip()
                    data_start = i + 1
                    break
            
            if header_line is None:
                raise ValueError("Não foi possível encontrar o cabeçalho dos dados")
            
            # Extrai os nomes das colunas
            columns = header_line.split('\t')
            print(f"📊 Encontradas {len(columns)} colunas de dados")
            
            # Carrega os dados
            data_lines = lines[data_start:]
            if sample_size:
                data_lines = data_lines[:sample_size]
                print(f"📝 Limitando análise a {sample_size} amostras")
            
            # Converte para DataFrame
            data_rows = []
            for line in data_lines:
                if line.strip():
                    values = line.strip().split('\t')
                    if len(values) == len(columns):
                        data_rows.append(values)
            
            self.log_data = pd.DataFrame(data_rows, columns=columns)
            
            # Converte colunas numéricas
            numeric_columns = [
                'Time', 'SecL', 'RPM', 'MAP', 'MAPxRPM', 'TPS', 'AFR', 'Lambda', 
                'MAT', 'CLT', 'VE _Current', 'VE1', 'VE2', 'PW', 'PW2', 'PW3', 'PW4',
                'AFR Target', 'Lambda Target', 'Duty_Cycle', 'TPS DOT', 'Advance _Current',
                'Battery V', 'FuelLoad', 'IgnitionLoad', 'DFCO', 'Engine'
            ]
            
            for col in numeric_columns:
                if col in self.log_data.columns:
                    self.log_data[col] = pd.to_numeric(self.log_data[col], errors='coerce')
            
            # Remove linhas com valores críticos faltando
            critical_columns = ['RPM', 'MAP', 'Lambda', 'VE _Current']
            available_critical = [col for col in critical_columns if col in self.log_data.columns]
            self.log_data = self.log_data.dropna(subset=available_critical)
            
            print(f"✅ Dados carregados: {len(self.log_data)} registros válidos")
            print(f"📈 Faixa RPM: {self.log_data['RPM'].min():.0f} - {self.log_data['RPM'].max():.0f}")
            print(f"📈 Faixa MAP: {self.log_data['MAP'].min():.1f} - {self.log_data['MAP'].max():.1f} kPa")
            print(f"📈 Faixa Lambda: {self.log_data['Lambda'].min():.3f} - {self.log_data['Lambda'].max():.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados MSL: {e}")
            return False
    
    def load_ve_table(self, ve_table_path):
        """Carrega a tabela VE do arquivo XML"""
        print(f"📊 Carregando tabela VE de: {ve_table_path}")
        
        try:
            tree = ET.parse(ve_table_path)
            root = tree.getroot()
            
            # Encontra a tabela
            table = root.find('.//table') or root.find('.//{http://www.EFIAnalytics.com/:table}table')
            
            if table is None:
                raise ValueError("Estrutura de tabela não encontrada no XML")
            
            # Extrai eixo X (RPM)
            x_axis = table.find('.//xAxis') or table.find('.//{http://www.EFIAnalytics.com/:table}xAxis')
            rpm_text = x_axis.text.strip().split()
            self.rpm_bins_ve = [float(x) for x in rpm_text]
            
            # Extrai eixo Y (FuelLoad)
            y_axis = table.find('.//yAxis') or table.find('.//{http://www.EFIAnalytics.com/:table}yAxis')
            load_text = y_axis.text.strip().split()
            self.load_bins_ve = [float(x) for x in load_text]
            
            # Extrai valores Z (VE)
            z_values = table.find('.//zValues') or table.find('.//{http://www.EFIAnalytics.com/:table}zValues')
            ve_text = z_values.text.strip().split()
            ve_values = [float(x) for x in ve_text]
            
            # Converte para matriz
            rows = len(self.load_bins_ve)
            cols = len(self.rpm_bins_ve)
            self.ve_table = np.array(ve_values).reshape(rows, cols)
            
            # Inicializa matriz de cobertura
            self.cell_coverage = np.zeros((rows, cols))
            
            print(f"✅ Tabela VE carregada: {rows}x{cols}")
            print(f"📈 RPM: {min(self.rpm_bins_ve)} - {max(self.rpm_bins_ve)}")
            print(f"📈 Load: {min(self.load_bins_ve)} - {max(self.load_bins_ve)}")
            print(f"📈 VE: {self.ve_table.min():.1f}% - {self.ve_table.max():.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar tabela VE: {e}")
            return False
    
    def load_lambda_table(self, lambda_table_path):
        """Carrega a tabela Lambda Target do arquivo XML"""
        print(f"🎯 Carregando tabela Lambda Target de: {lambda_table_path}")
        
        try:
            tree = ET.parse(lambda_table_path)
            root = tree.getroot()
            
            # Encontra a tabela
            table = root.find('.//table') or root.find('.//{http://www.EFIAnalytics.com/:table}table')
            
            if table is None:
                raise ValueError("Estrutura de tabela não encontrada no XML")
            
            # Extrai eixo X (RPM)
            x_axis = table.find('.//xAxis') or table.find('.//{http://www.EFIAnalytics.com/:table}xAxis')
            rpm_text = x_axis.text.strip().split()
            self.rpm_bins_lambda = [float(x) for x in rpm_text]
            
            # Extrai eixo Y (FuelLoad)
            y_axis = table.find('.//yAxis') or table.find('.//{http://www.EFIAnalytics.com/:table}yAxis')
            load_text = y_axis.text.strip().split()
            self.load_bins_lambda = [float(x) for x in load_text]
            
            # Extrai valores Z (Lambda)
            z_values = table.find('.//zValues') or table.find('.//{http://www.EFIAnalytics.com/:table}zValues')
            lambda_text = z_values.text.strip().split()
            lambda_values = [float(x) for x in lambda_text]
            
            # Converte para matriz
            rows = len(self.load_bins_lambda)
            cols = len(self.rpm_bins_lambda)
            self.lambda_table = np.array(lambda_values).reshape(rows, cols)
            
            print(f"✅ Tabela Lambda carregada: {rows}x{cols}")
            print(f"📈 RPM: {min(self.rpm_bins_lambda)} - {max(self.rpm_bins_lambda)}")
            print(f"📈 Load: {min(self.load_bins_lambda)} - {max(self.load_bins_lambda)}")
            print(f"🎯 Lambda: {self.lambda_table.min():.3f} - {self.lambda_table.max():.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar tabela Lambda: {e}")
            return False
    
    def estimate_lambda_delay_from_dfco(self):
        """
        Estima delay da sonda usando eventos de DFCO (cutoff)
        DFCO causa mudanças rápidas e previsíveis no lambda
        """
        if self.log_data is None:
            return 0.8
        
        print("🔍 Estimando delay da sonda usando eventos DFCO...")
        
        # Verifica se temos coluna DFCO
        if 'DFCO' not in self.log_data.columns:
            print("⚠️  Coluna DFCO não encontrada, usando delay padrão")
            return 0.8
        
        # Encontra transições DFCO
        dfco_series = self.log_data['DFCO'].fillna(0)
        dfco_changes = dfco_series.diff().abs() > 0.5  # Mudança de estado
        
        dfco_events = self.log_data[dfco_changes].copy()
        
        if len(dfco_events) < 2:
            print("📊 Poucos eventos DFCO encontrados, usando delay padrão")
            return 0.8
        
        delays = []
        
        for idx in dfco_events.index[:10]:  # Analisa até 10 eventos
            event_time = self.log_data.loc[idx, 'Time']
            dfco_state = self.log_data.loc[idx, 'DFCO']
            
            # Janela de 3 segundos após o evento
            window_mask = (self.log_data['Time'] >= event_time) & (self.log_data['Time'] <= event_time + 3.0)
            window = self.log_data[window_mask]
            
            if len(window) > 10:
                # Se DFCO ativou, lambda deve ficar pobre (>1.2)
                # Se DFCO desativou, lambda deve normalizar
                if dfco_state > 0.5:  # DFCO ativado
                    target_condition = window['Lambda'] > 1.5
                else:  # DFCO desativado
                    target_condition = window['Lambda'] < 1.2
                
                if target_condition.any():
                    first_reaction_idx = window[target_condition].index[0]
                    delay_time = self.log_data.loc[first_reaction_idx, 'Time'] - event_time
                    
                    if 0.1 <= delay_time <= 2.0:  # Delay razoável
                        delays.append(delay_time)
        
        if delays:
            estimated_delay = np.median(delays)
            print(f"📡 Delay da sonda estimado: {estimated_delay:.3f}s (baseado em {len(delays)} eventos DFCO)")
            return estimated_delay
        else:
            print("📊 Não foi possível estimar delay com DFCO, usando padrão")
            return 0.8
    
    def filter_stable_conditions(self, min_stable_time=2.0, max_tps_change=2.0):
        """
        Filtra dados para condições estáveis (TPS estável, sem DFCO)
        
        Args:
            min_stable_time: Tempo mínimo de estabilidade (segundos)
            max_tps_change: Máxima variação de TPS permitida (%/s)
        """
        if self.log_data is None:
            return False
        
        print(f"🔍 Filtrando condições estáveis...")
        print(f"   Tempo mínimo estável: {min_stable_time}s")
        print(f"   Máxima variação TPS: {max_tps_change}%/s")
        
        log_data = self.log_data.copy()
        
        # Calcula derivada do TPS
        if 'TPS' in log_data.columns:
            time_diff = log_data['Time'].diff()
            tps_diff = log_data['TPS'].diff()
            log_data['TPS_Rate'] = tps_diff / time_diff
            log_data['TPS_Rate'] = log_data['TPS_Rate'].fillna(0)
        else:
            log_data['TPS_Rate'] = 0
        
        # Marca pontos estáveis
        conditions = [
            log_data['CLT'] > 75,  # Motor aquecido
            log_data['RPM'] > 800,  # Fora da marcha lenta instável
            abs(log_data['TPS_Rate']) < max_tps_change,  # TPS estável
        ]
        
        # Adiciona condição DFCO se disponível
        if 'DFCO' in log_data.columns:
            conditions.append(log_data['DFCO'] < 0.5)  # Sem cutoff
        
        # Combina todas as condições
        stable_mask = pd.Series(True, index=log_data.index)
        for condition in conditions:
            stable_mask &= condition
        
        # Filtra por tempo mínimo de estabilidade
        log_data['is_stable'] = stable_mask
        
        # Encontra segmentos estáveis contínuos
        stable_segments = []
        current_segment_start = None
        
        for i, is_stable in enumerate(log_data['is_stable']):
            if is_stable and current_segment_start is None:
                current_segment_start = i
            elif not is_stable and current_segment_start is not None:
                segment_duration = log_data.iloc[i-1]['Time'] - log_data.iloc[current_segment_start]['Time']
                if segment_duration >= min_stable_time:
                    stable_segments.append((current_segment_start, i-1))
                current_segment_start = None
        
        # Último segmento
        if current_segment_start is not None:
            segment_duration = log_data.iloc[-1]['Time'] - log_data.iloc[current_segment_start]['Time']
            if segment_duration >= min_stable_time:
                stable_segments.append((current_segment_start, len(log_data)-1))
        
        # Seleciona apenas pontos dos segmentos estáveis
        stable_indices = []
        for start_idx, end_idx in stable_segments:
            stable_indices.extend(range(start_idx, end_idx + 1))
        
        self.stable_data = log_data.iloc[stable_indices].copy()
        
        print(f"✅ Condições estáveis identificadas:")
        print(f"   Total de pontos: {len(log_data)}")
        print(f"   Pontos estáveis: {len(self.stable_data)} ({100*len(self.stable_data)/len(log_data):.1f}%)")
        print(f"   Segmentos estáveis: {len(stable_segments)}")
        
        return True
    
    def calculate_cell_coverage(self, min_points_per_cell=10):
        """
        Calcula quantos pontos estáveis cada célula da tabela VE possui
        
        Args:
            min_points_per_cell: Mínimo de pontos para considerar célula confiável
        """
        if self.stable_data is None or self.ve_table is None:
            print("❌ Dados estáveis ou tabela VE não carregados")
            return False
        
        print(f"📊 Calculando cobertura de células...")
        print(f"   Mínimo de pontos por célula: {min_points_per_cell}")
        
        # Inicializa matriz de cobertura
        rows, cols = self.ve_table.shape
        self.cell_coverage = np.zeros((rows, cols))
        
        # Para cada ponto estável, encontra a célula correspondente
        for _, row in self.stable_data.iterrows():
            rpm = row['RPM']
            load = row.get('FuelLoad', row.get('MAP', 0))
            
            # Encontra índices da célula mais próxima
            rpm_idx = np.argmin(np.abs(np.array(self.rpm_bins_ve) - rpm))
            load_idx = np.argmin(np.abs(np.array(self.load_bins_ve) - load))
            
            # Verifica se está dentro dos limites razoáveis
            rpm_error = abs(self.rpm_bins_ve[rpm_idx] - rpm) / max(self.rpm_bins_ve) * 100
            load_error = abs(self.load_bins_ve[load_idx] - load) / max(self.load_bins_ve) * 100
            
            # Aceita se erro < 20% da faixa total
            if rpm_error < 20 and load_error < 20:
                self.cell_coverage[load_idx, rpm_idx] += 1
        
        # Estatísticas de cobertura
        total_cells = rows * cols
        covered_cells = np.sum(self.cell_coverage >= min_points_per_cell)
        coverage_percentage = (covered_cells / total_cells) * 100
        
        print(f"✅ Cobertura calculada:")
        print(f"   Total de células: {total_cells}")
        print(f"   Células com dados suficientes: {covered_cells} ({coverage_percentage:.1f}%)")
        print(f"   Pontos médios por célula coberta: {self.cell_coverage[self.cell_coverage >= min_points_per_cell].mean():.1f}")
        print(f"   Máximo de pontos em uma célula: {self.cell_coverage.max():.0f}")
        
        return True
    
    def get_ve_interpolated(self, rpm, load):
        """
        Interpola valor VE da tabela para RPM e Load específicos
        """
        if self.ve_table is None:
            return None
        
        try:
            from scipy.interpolate import RegularGridInterpolator
            
            interpolator = RegularGridInterpolator(
                (self.load_bins_ve, self.rpm_bins_ve), 
                self.ve_table, 
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            
            result = interpolator([load, rpm])
            return float(result[0]) if not np.isnan(result[0]) else None
            
        except Exception as e:
            print(f"Erro na interpolação VE: {e}")
            return None
    
    def get_lambda_target_interpolated(self, rpm, load):
        """
        Interpola valor Lambda Target da tabela para RPM e Load específicos
        """
        if self.lambda_table is None:
            return None
        
        try:
            from scipy.interpolate import RegularGridInterpolator
            
            interpolator = RegularGridInterpolator(
                (self.load_bins_lambda, self.rpm_bins_lambda), 
                self.lambda_table, 
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            
            result = interpolator([load, rpm])
            return float(result[0]) if not np.isnan(result[0]) else None
            
        except Exception as e:
            print(f"Erro na interpolação Lambda: {e}")
            return None
    
    def analyze_data_quality(self):
        """Analisa a qualidade dos dados carregados"""
        if self.log_data is None:
            print("❌ Nenhum dado carregado")
            return
        
        print("\n🔍 ANÁLISE DE QUALIDADE DOS DADOS")
        print("=" * 50)
        
        # Estatísticas básicas
        print(f"📊 Total de registros: {len(self.log_data)}")
        print(f"⏱️  Duração: {self.log_data['Time'].max():.1f} segundos")
        
        # Verifica dados essenciais
        essential_cols = ['RPM', 'MAP', 'Lambda', 'VE _Current']
        for col in essential_cols:
            if col in self.log_data.columns:
                values = self.log_data[col]
                print(f"📈 {col}: {values.min():.2f} - {values.max():.2f} (média: {values.mean():.2f})")
        
        # Verifica condições de operação
        print("\n🌡️ CONDIÇÕES DE OPERAÇÃO:")
        if 'CLT' in self.log_data.columns:
            temp_min = self.log_data['CLT'].min()
            temp_max = self.log_data['CLT'].max()
            print(f"   Temperatura motor: {temp_min:.0f}°C - {temp_max:.0f}°C")
            if temp_max < 80:
                print("   ⚠️  Motor pode não estar totalmente aquecido")
        
        # Verifica DFCO
        if 'DFCO' in self.log_data.columns:
            dfco_events = (self.log_data['DFCO'] > 0.5).sum()
            dfco_percentage = (dfco_events / len(self.log_data)) * 100
            print(f"   Eventos DFCO: {dfco_events} ({dfco_percentage:.1f}%)")
        
        # Verifica distribuição de pontos
        print("\n📊 DISTRIBUIÇÃO DE DADOS:")
        rpm_ranges = [
            (500, 1500, "Marcha lenta"),
            (1500, 3000, "Cruzeiro baixo"),
            (3000, 4500, "Cruzeiro médio"), 
            (4500, 6000, "Alta rotação"),
            (6000, 8000, "Zona vermelha")
        ]
        
        for rpm_min, rpm_max, label in rpm_ranges:
            mask = (self.log_data['RPM'] >= rpm_min) & (self.log_data['RPM'] < rpm_max)
            count = mask.sum()
            pct = (count / len(self.log_data)) * 100
            print(f"   {label:15}: {count:5d} pontos ({pct:4.1f}%)")
        
        # Se já temos dados estáveis, mostra estatísticas
        if self.stable_data is not None:
            print(f"\n✅ DADOS ESTÁVEIS FILTRADOS:")
            print(f"   Pontos estáveis: {len(self.stable_data)} ({100*len(self.stable_data)/len(self.log_data):.1f}%)")
    
    def plot_coverage_heatmap(self, save_path=None):
        """Gera heatmap da cobertura de células"""
        if self.cell_coverage is None:
            print("❌ Cobertura não calculada")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Cria heatmap
        im = plt.imshow(self.cell_coverage, aspect='auto', cmap='viridis', origin='lower')
        
        # Configura eixos
        plt.title('Cobertura de Dados por Célula VE\n(Número de pontos estáveis)', fontsize=14, fontweight='bold')
        plt.xlabel('RPM')
        plt.ylabel('FuelLoad (kPa)')
        
        # Eixos com valores reais
        rpm_ticks = np.arange(0, len(self.rpm_bins_ve), 2)
        load_ticks = np.arange(0, len(self.load_bins_ve), 2)
        
        plt.xticks(rpm_ticks, [f"{int(self.rpm_bins_ve[i])}" for i in rpm_ticks])
        plt.yticks(load_ticks, [f"{int(self.load_bins_ve[i])}" for i in load_ticks])
        
        # Colorbar
        cbar = plt.colorbar(im, label='Número de Pontos')
        
        # Adiciona valores nas células
        for i in range(len(self.load_bins_ve)):
            for j in range(len(self.rpm_bins_ve)):
                value = self.cell_coverage[i, j]
                if value > 0:
                    plt.text(j, i, f'{int(value)}', ha='center', va='center', 
                           color='white' if value > self.cell_coverage.max()/2 else 'black',
                           fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Heatmap de cobertura salvo em: {save_path}")
        
        plt.show()
    
    def get_summary_stats(self):
        """Retorna estatísticas resumidas dos dados"""
        if self.log_data is None:
            return None
        
        stats = {
            'total_points': len(self.log_data),
            'duration': self.log_data['Time'].max(),
            'rpm_range': (self.log_data['RPM'].min(), self.log_data['RPM'].max()),
            'map_range': (self.log_data['MAP'].min(), self.log_data['MAP'].max()),
            'lambda_range': (self.log_data['Lambda'].min(), self.log_data['Lambda'].max())
        }
        
        if self.stable_data is not None:
            stats['stable_points'] = len(self.stable_data)
            stats['stability_ratio'] = len(self.stable_data) / len(self.log_data)
        
        if self.cell_coverage is not None:
            stats['covered_cells'] = np.sum(self.cell_coverage >= 10)
            stats['total_cells'] = self.cell_coverage.size
            stats['coverage_ratio'] = stats['covered_cells'] / stats['total_cells']
        
        return stats

if __name__ == "__main__":
    # Teste das novas funcionalidades
    processor = MSLDataProcessor()
    
    # Carrega dados
    if processor.load_msl_data("logs/short.msl"):
        processor.analyze_data_quality()
        
        # Estima delay usando DFCO
        delay = processor.estimate_lambda_delay_from_dfco()
        
        # Filtra condições estáveis
        processor.filter_stable_conditions()
        
        # Carrega tabela VE e calcula cobertura
        if processor.load_ve_table("logs/veTable1Tbl_pre.table"):
            processor.calculate_cell_coverage(min_points_per_cell=5)
            processor.plot_coverage_heatmap("coverage_heatmap.png")
