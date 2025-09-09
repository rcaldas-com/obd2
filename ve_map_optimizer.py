#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Otimizador de VE Map com foco em confiabilidade
Modifica apenas células com dados suficientes e condições estáveis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from msl_data_processor import MSLDataProcessor

class VEMapOptimizerReliable:
    """
    Otimizador de VE Map focado em confiabilidade
    - Modifica apenas células com dados suficientes
    - Usa apenas condições estáveis (TPS estável, sem DFCO)
    - Considera delay da sonda lambda
    """
    
    def __init__(self):
        self.processor = MSLDataProcessor()
        self.model = None
        self.ve_optimized = None
        self.lambda_delay = 0.8  # Segundos
        self.min_points_per_cell = 15  # Mínimo de pontos para modificar célula
        
        print("🎯 VE Map Optimizer inicializado (Versão Confiabilidade)")
        print(f"   Mínimo de pontos por célula: {self.min_points_per_cell}")
    
    def load_data(self, msl_file, ve_table_file, lambda_table_file=None):
        """Carrega todos os dados necessários"""
        print("📂 Carregando dados para otimização...")
        
        # Carrega dados MSL
        if not self.processor.load_msl_data(msl_file):
            return False
        
        # Carrega tabela VE
        if not self.processor.load_ve_table(ve_table_file):
            return False
        
        # Carrega tabela Lambda se fornecida
        if lambda_table_file:
            self.processor.load_lambda_table(lambda_table_file)
        
        # Estima delay da sonda
        self.lambda_delay = self.processor.estimate_lambda_delay_from_dfco()
        
        # Filtra condições estáveis
        if not self.processor.filter_stable_conditions():
            return False
        
        # Calcula cobertura de células
        if not self.processor.calculate_cell_coverage(self.min_points_per_cell):
            return False
        
        print("✅ Todos os dados carregados e processados")
        return True
    
    def prepare_training_data(self):
        """
        Prepara dados de treinamento considerando delay da sonda
        """
        if self.processor.stable_data is None:
            print("❌ Dados estáveis não disponíveis")
            return None
        
        print(f"🔄 Preparando dados de treinamento...")
        print(f"   Considerando delay da sonda: {self.lambda_delay:.3f}s")
        
        data = self.processor.stable_data.copy()
        
        # Aplica delay temporal no lambda
        # O lambda medido agora reflete as condições de X segundos atrás
        data['Time_Delayed'] = data['Time'] - self.lambda_delay
        
        training_rows = []
        
        for idx, row in data.iterrows():
            # Encontra as condições operacionais no momento do delay
            delayed_time = row['Time_Delayed']
            
            # Busca condições próximas ao tempo com delay
            time_mask = abs(data['Time'] - delayed_time) <= 0.1  # ±0.1s tolerância
            candidates = data[time_mask]
            
            if len(candidates) == 0:
                continue
            
            # Pega o ponto mais próximo no tempo
            closest_idx = (abs(candidates['Time'] - delayed_time)).idxmin()
            delayed_conditions = candidates.loc[closest_idx]
            
            # RPM e Load do momento da injeção (delay aplicado)
            rpm = delayed_conditions['RPM']
            fuel_load = delayed_conditions.get('FuelLoad', delayed_conditions.get('MAP', 0))
            
            # Lambda medido agora (resultado das condições passadas)
            lambda_measured = row['Lambda']
            
            # Interpolações das tabelas para as condições passadas
            ve_table_value = self.processor.get_ve_interpolated(rpm, fuel_load)
            lambda_target_value = self.processor.get_lambda_target_interpolated(rpm, fuel_load)
            
            # Validações
            if any(x is None for x in [ve_table_value, lambda_target_value]):
                continue
            
            if not (500 <= rpm <= 8000 and 10 <= fuel_load <= 200):
                continue
                
            if not (0.6 <= lambda_measured <= 1.6):
                continue
            
            # Calcula erro lambda atual
            lambda_error = (lambda_measured - lambda_target_value) / lambda_target_value
            
            # Adiciona outras variáveis que podem influenciar
            training_row = {
                'RPM': rpm,
                'FuelLoad': fuel_load,
                'VE_Current': ve_table_value,
                'Lambda_Target': lambda_target_value,
                'Lambda_Measured': lambda_measured,
                'Lambda_Error': lambda_error,
                'CLT': delayed_conditions.get('CLT', 85),
                'MAT': delayed_conditions.get('MAT', 25),
                'Lambda_Error_Abs': abs(lambda_error)
            }
            
            training_rows.append(training_row)
        
        if len(training_rows) == 0:
            print("❌ Nenhum dado de treinamento válido gerado")
            return None
        
        df = pd.DataFrame(training_rows)
        
        # Remove outliers extremos
        lambda_error_q99 = df['Lambda_Error_Abs'].quantile(0.99)
        df = df[df['Lambda_Error_Abs'] <= lambda_error_q99]
        
        print(f"✅ Dados de treinamento preparados:")
        print(f"   Total de pontos: {len(df)}")
        print(f"   Faixa RPM: {df['RPM'].min():.0f} - {df['RPM'].max():.0f}")
        print(f"   Faixa Load: {df['FuelLoad'].min():.1f} - {df['FuelLoad'].max():.1f}")
        print(f"   Erro Lambda médio: {df['Lambda_Error_Abs'].mean():.3f}")
        print(f"   Delay aplicado: {self.lambda_delay:.3f}s")
        
        return df
    
    def train_model(self, training_data):
        """Treina modelo de machine learning"""
        if training_data is None or len(training_data) == 0:
            print("❌ Dados de treinamento inválidos")
            return False
        
        print("🤖 Treinando modelo de machine learning...")
        
        # Features
        feature_cols = ['RPM', 'FuelLoad', 'VE_Current', 'Lambda_Target', 'CLT', 'MAT']
        X = training_data[feature_cols]
        
        # Target: erro lambda
        y = training_data['Lambda_Error']
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treina Random Forest
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Avalia modelo
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"✅ Modelo treinado:")
        print(f"   R² treino: {r2_train:.4f}")
        print(f"   R² teste: {r2_test:.4f}")
        print(f"   RMSE teste: {rmse_test:.4f}")
        
        # Importância das features
        print("\n📊 Importância das variáveis:")
        for feature, importance in zip(feature_cols, self.model.feature_importances_):
            print(f"   {feature:12}: {importance:.3f}")
        
        return True
    
    def optimize_ve_table(self, conservative_factor=0.5):
        """
        Otimiza tabela VE modificando apenas células com dados suficientes
        
        Args:
            conservative_factor: Fator de conservadorismo (0.5 = aplica 50% da correção)
        """
        if self.model is None:
            print("❌ Modelo não treinado")
            return False
        
        if self.processor.cell_coverage is None:
            print("❌ Cobertura de células não calculada")
            return False
        
        print(f"🎯 Otimizando tabela VE...")
        print(f"   Fator conservador: {conservative_factor:.1f}")
        print(f"   Mínimo de pontos por célula: {self.min_points_per_cell}")
        
        # Copia tabela original
        self.ve_optimized = self.processor.ve_table.copy()
        
        modification_stats = {
            'total_cells': 0,
            'covered_cells': 0,
            'modified_cells': 0,
            'significant_changes': 0,
            'max_change': 0
        }
        
        rows, cols = self.ve_optimized.shape
        modification_stats['total_cells'] = rows * cols
        
        # Para cada célula da tabela
        for i in range(rows):
            for j in range(cols):
                # Verifica se temos dados suficientes
                if self.processor.cell_coverage[i, j] < self.min_points_per_cell:
                    continue
                
                modification_stats['covered_cells'] += 1
                
                # Condições operacionais desta célula
                rpm = self.processor.rpm_bins_ve[j]
                fuel_load = self.processor.load_bins_ve[i]
                ve_current = self.ve_optimized[i, j]
                
                # Lambda target
                lambda_target = self.processor.get_lambda_target_interpolated(rpm, fuel_load)
                if lambda_target is None:
                    lambda_target = 1.0  # Valor padrão
                
                # Condições médias (usar dados históricos se disponível)
                clt = 85  # Temperatura operacional
                mat = 25  # Temperatura do ar
                
                # Prepara dados para predição
                features = [[rpm, fuel_load, ve_current, lambda_target, clt, mat]]
                
                try:
                    # Prediz erro lambda esperado
                    predicted_error = self.model.predict(features)[0]
                    
                    # Calcula correção VE necessária
                    # Erro positivo = muito rico (precisa menos combustível = mais VE)
                    # Erro negativo = muito pobre (precisa mais combustível = menos VE)
                    ve_correction = predicted_error * 100 * conservative_factor  # Converte para %
                    
                    # Aplica correção
                    ve_new = ve_current + ve_correction
                    
                    # Limita correção a ±20%
                    max_change = ve_current * 0.20
                    ve_correction = np.clip(ve_correction, -max_change, max_change)
                    ve_new = ve_current + ve_correction
                    
                    # Limita valores finais
                    ve_new = np.clip(ve_new, 20, 150)
                    
                    # Atualiza se mudança significativa
                    if abs(ve_correction) >= 0.5:  # Mudança mínima de 0.5%
                        self.ve_optimized[i, j] = ve_new
                        modification_stats['modified_cells'] += 1
                        
                        if abs(ve_correction) >= 2.0:  # Mudança significativa
                            modification_stats['significant_changes'] += 1
                        
                        modification_stats['max_change'] = max(modification_stats['max_change'], abs(ve_correction))
                
                except Exception as e:
                    print(f"Erro ao processar célula [{i},{j}]: {e}")
                    continue
        
        # Estatísticas finais
        print(f"✅ Otimização concluída:")
        print(f"   Total de células: {modification_stats['total_cells']}")
        print(f"   Células com dados: {modification_stats['covered_cells']} ({100*modification_stats['covered_cells']/modification_stats['total_cells']:.1f}%)")
        print(f"   Células modificadas: {modification_stats['modified_cells']} ({100*modification_stats['modified_cells']/modification_stats['covered_cells']:.1f}% das cobertas)")
        print(f"   Mudanças significativas (>2%): {modification_stats['significant_changes']}")
        print(f"   Maior mudança: {modification_stats['max_change']:.1f}%")
        
        self.modification_stats = modification_stats
        return True
    
    def save_optimized_table(self, output_path, base_table_path):
        """Salva tabela otimizada mantendo estrutura XML original"""
        if self.ve_optimized is None:
            print("❌ Tabela otimizada não disponível")
            return False
        
        print(f"💾 Salvando tabela otimizada em: {output_path}")
        
        try:
            # Carrega XML original
            tree = ET.parse(base_table_path)
            root = tree.getroot()
            
            # Encontra elemento zValues
            z_values = root.find('.//zValues') or root.find('.//{http://www.EFIAnalytics.com/:table}zValues')
            
            if z_values is None:
                raise ValueError("Elemento zValues não encontrado")
            
            # Converte matriz para string
            ve_flat = self.ve_optimized.flatten()
            ve_string = ' '.join([f"{val:.1f}" for val in ve_flat])
            
            # Atualiza XML
            z_values.text = ve_string
            
            # Salva
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            print(f"✅ Tabela salva com sucesso")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar tabela: {e}")
            return False
    
    def plot_optimization_analysis(self, save_path=None):
        """Gera gráficos de análise da otimização"""
        if self.ve_optimized is None:
            print("❌ Tabela otimizada não disponível")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Tabela original
        im1 = ax1.imshow(self.processor.ve_table, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_title('VE Table Original', fontweight='bold')
        ax1.set_xlabel('RPM')
        ax1.set_ylabel('FuelLoad')
        plt.colorbar(im1, ax=ax1, label='VE (%)')
        
        # 2. Tabela otimizada
        im2 = ax2.imshow(self.ve_optimized, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_title('VE Table Otimizada', fontweight='bold')
        ax2.set_xlabel('RPM')
        ax2.set_ylabel('FuelLoad')
        plt.colorbar(im2, ax=ax2, label='VE (%)')
        
        # 3. Diferenças
        diff_table = self.ve_optimized - self.processor.ve_table
        im3 = ax3.imshow(diff_table, aspect='auto', cmap='RdBu_r', origin='lower', 
                         vmin=-10, vmax=10)
        ax3.set_title('Diferenças (Otimizada - Original)', fontweight='bold')
        ax3.set_xlabel('RPM')
        ax3.set_ylabel('FuelLoad')
        plt.colorbar(im3, ax=ax3, label='Δ VE (%)')
        
        # 4. Cobertura de dados
        # Máscara para células modificadas
        coverage_display = self.processor.cell_coverage.copy()
        coverage_display[coverage_display < self.min_points_per_cell] = 0
        
        im4 = ax4.imshow(coverage_display, aspect='auto', cmap='plasma', origin='lower')
        ax4.set_title(f'Cobertura de Dados\n(Min: {self.min_points_per_cell} pontos)', fontweight='bold')
        ax4.set_xlabel('RPM')
        ax4.set_ylabel('FuelLoad')
        plt.colorbar(im4, ax=ax4, label='Pontos Estáveis')
        
        # Ajusta eixos para todos os subplots
        for ax in [ax1, ax2, ax3, ax4]:
            rpm_ticks = np.arange(0, len(self.processor.rpm_bins_ve), 3)
            load_ticks = np.arange(0, len(self.processor.load_bins_ve), 2)
            
            ax.set_xticks(rpm_ticks)
            ax.set_xticklabels([f"{int(self.processor.rpm_bins_ve[i])}" for i in rpm_ticks])
            ax.set_yticks(load_ticks)
            ax.set_yticklabels([f"{int(self.processor.load_bins_ve[i])}" for i in load_ticks])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Análise salva em: {save_path}")
        
        plt.show()

if __name__ == "__main__":
    # Teste do otimizador
    optimizer = VEMapOptimizerReliable()
    
    if optimizer.load_data("logs/short.msl", "logs/veTable1Tbl_pre.table", "logs/lambdaTable1Tbl_.table"):
        training_data = optimizer.prepare_training_data()
        
        if training_data is not None and optimizer.train_model(training_data):
            if optimizer.optimize_ve_table(conservative_factor=0.7):
                optimizer.save_optimized_table("ve_optimized_reliable.table", "logs/veTable1Tbl_pre.table")
                optimizer.plot_optimization_analysis("ve_analysis_reliable.png")
