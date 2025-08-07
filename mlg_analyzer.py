#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisador aprimorado para arquivos MLG do TunerStudio
"""

import struct
import pandas as pd
import numpy as np

class MLGAnalyzer:
    """Analisador especializado para arquivos MLG"""
    
    def __init__(self):
        self.header_info = {}
        self.channels = []
        self.data_start = 0
        
    def analyze_mlg_structure(self, file_path):
        """Analisa a estrutura completa do arquivo MLG"""
        print(f"🔍 Analisando estrutura de {file_path}")
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Verifica header
        if content[:5] != b'MLVLG':
            print("❌ Não é um arquivo MLG válido")
            return None
        
        print(f"✅ Arquivo MLG válido ({len(content)} bytes)")
        
        # Extrai informações do header
        header_version = struct.unpack('<H', content[6:8])[0]
        print(f"📋 Versão do header: {header_version}")
        
        # Procura por nomes de canais
        self._extract_channel_names(content)
        
        # Encontra início dos dados
        self._find_data_section(content)
        
        return True
    
    def _extract_channel_names(self, content):
        """Extrai nomes dos canais do arquivo"""
        # Procura por strings conhecidas
        known_channels = [
            b'Time', b'SecL', b'RPM', b'MAP', b'MAPxRPM', b'Lambda', 
            b'Engine', b'DFCO', b'Gego', b'Gair', b'Gbattery', b'Gwarm',
            b'Gbaro', b'Gammae', b'Accel Enrich', b'VE _Current', 
            b'AFR Target', b'Lambda Target', b'Duty Cycle'
        ]
        
        found_channels = []
        for channel in known_channels:
            pos = content.find(channel)
            if pos != -1:
                found_channels.append((channel.decode('utf-8'), pos))
        
        found_channels.sort(key=lambda x: x[1])  # Ordena por posição
        self.channels = [ch[0] for ch in found_channels]
        
        print(f"📊 Canais encontrados ({len(self.channels)}):")
        for i, ch in enumerate(self.channels):
            print(f"  {i+1:2d}. {ch}")
    
    def _find_data_section(self, content):
        """Encontra onde começam os dados numéricos usando múltiplas estratégias"""
        print("🔍 Procurando seção de dados...")
        
        # Estratégia 1: Procura por valores RPM típicos
        rpm_candidates = []
        for i in range(50000, min(200000, len(content) - 400), 100):
            try:
                # Testa sequências de 20 valores
                valid_count = 0
                values = []
                for j in range(20):
                    val = struct.unpack('<f', content[i + j*4:i + j*4 + 4])[0]
                    values.append(val)
                    if 400 <= val <= 8000:  # RPM válido
                        valid_count += 1
                
                if valid_count >= 8:  # Pelo menos 40% dos valores são RPM válidos
                    rpm_candidates.append((i, valid_count, values[:5]))
                    
            except:
                continue
        
        # Estratégia 2: Procura por valores Lambda típicos
        lambda_candidates = []
        for i in range(50000, min(200000, len(content) - 400), 100):
            try:
                valid_count = 0
                values = []
                for j in range(20):
                    val = struct.unpack('<f', content[i + j*4:i + j*4 + 4])[0]
                    values.append(val)
                    if 0.4 <= val <= 2.5:  # Lambda válido
                        valid_count += 1
                
                if valid_count >= 6:  # Pelo menos 30% são lambda válidos
                    lambda_candidates.append((i, valid_count, values[:5]))
                    
            except:
                continue
        
        # Estratégia 3: Procura por valores de pressão (MAP)
        map_candidates = []
        for i in range(50000, min(200000, len(content) - 400), 100):
            try:
                valid_count = 0
                values = []
                for j in range(20):
                    val = struct.unpack('<f', content[i + j*4:i + j*4 + 4])[0]
                    values.append(val)
                    if 10 <= val <= 300:  # MAP/pressão válido
                        valid_count += 1
                
                if valid_count >= 8:
                    map_candidates.append((i, valid_count, values[:5]))
                    
            except:
                continue
        
        # Analisa os candidatos
        all_candidates = []
        
        for offset, count, values in rpm_candidates:
            all_candidates.append((offset, count, 'RPM', values))
            
        for offset, count, values in lambda_candidates:
            all_candidates.append((offset, count, 'Lambda', values))
            
        for offset, count, values in map_candidates:
            all_candidates.append((offset, count, 'MAP', values))
        
        if all_candidates:
            # Ordena por número de valores válidos
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            
            print(f"🎯 Candidatos encontrados ({len(all_candidates)}):")
            for i, (offset, count, type_hint, values) in enumerate(all_candidates[:5]):
                print(f"  {i+1}. Offset: 0x{offset:08x} ({offset}), {type_hint}: {count} válidos")
                print(f"     Valores: {[f'{v:.3f}' for v in values]}")
            
            # Escolhe o melhor candidato
            best_offset = all_candidates[0][0]
            self.data_start = best_offset
            print(f"✅ Melhor candidato selecionado: offset 0x{best_offset:08x}")
            return best_offset
        
        print("❌ Nenhum candidato válido encontrado")
        return -1
    
    def extract_data(self, file_path, max_records=50000):
        """Extrai dados do arquivo MLG com múltiplas tentativas"""
        if not self.analyze_mlg_structure(file_path):
            return None
        
        if self.data_start <= 0:
            print("❌ Início dos dados não encontrado")
            return None
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Testa diferentes números de canais
        channel_options = [16, 18, 20, 22, 24, 14, 12, 10, 8]
        
        best_df = None
        best_score = 0
        
        for num_channels in channel_options:
            print(f"\n🧪 Testando com {num_channels} canais...")
            
            df = self._try_extract_with_channels(content, num_channels, max_records)
            
            if df is not None:
                score = self._score_data_quality(df)
                print(f"   Pontuação de qualidade: {score}")
                
                if score > best_score:
                    best_score = score
                    best_df = df
                    print(f"   ✅ Nova melhor opção!")
        
        if best_df is not None:
            print(f"\n🎯 Melhor resultado:")
            print(f"   Canais: {len(best_df.columns)}")
            print(f"   Registros: {len(best_df)}")
            print(f"   Pontuação: {best_score}")
            
            # Renomeia colunas se possível
            best_df = self._assign_column_names(best_df)
            
            self._show_data_stats(best_df)
            return best_df
        
        print("❌ Nenhuma configuração válida encontrada")
        return None
    
    def _try_extract_with_channels(self, content, num_channels, max_records):
        """Tenta extrair dados com número específico de canais"""
        try:
            record_size = num_channels * 4
            remaining_bytes = len(content) - self.data_start
            estimated_records = remaining_bytes // record_size
            
            if estimated_records < 100:
                return None
            
            actual_records = min(estimated_records, max_records)
            
            data_matrix = []
            pos = self.data_start
            
            for i in range(actual_records):
                if pos + record_size > len(content):
                    break
                
                record = []
                for j in range(num_channels):
                    if pos + 4 <= len(content):
                        try:
                            val = struct.unpack('<f', content[pos:pos+4])[0]
                            
                            # Filtra valores extremamente inválidos
                            if abs(val) > 1e6 or val != val:  # NaN
                                val = 0.0
                            
                            record.append(val)
                            pos += 4
                        except:
                            record.append(0.0)
                            pos += 4
                    else:
                        break
                
                if len(record) == num_channels:
                    data_matrix.append(record)
                else:
                    break
            
            if len(data_matrix) < 100:
                return None
            
            columns = [f'Channel_{i+1}' for i in range(num_channels)]
            df = pd.DataFrame(data_matrix, columns=columns)
            
            return df
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            return None
    
    def _score_data_quality(self, df):
        """Pontua a qualidade dos dados extraídos"""
        if df is None or len(df) < 100:
            return 0
        
        score = 0
        
        for col in df.columns:
            values = df[col]
            
            # Evita colunas com muitos zeros
            if (values == 0).sum() > len(values) * 0.8:
                continue
            
            # Pontos por ranges válidos conhecidos
            if values.min() >= 300 and values.max() <= 8000 and values.std() > 100:
                score += 100  # Possível RPM
                
            elif values.min() >= 0.3 and values.max() <= 3.0 and values.std() > 0.05:
                score += 80   # Possível Lambda
                
            elif values.min() >= 5 and values.max() <= 400 and values.std() > 5:
                score += 60   # Possível MAP/pressão
                
            elif values.min() >= -10 and values.max() <= 120 and values.std() > 2:
                score += 40   # Possível TPS ou temperatura
                
            elif values.std() > 0.1:  # Qualquer variação
                score += 10
        
        return score
    
    def _assign_column_names(self, df):
        """Tenta atribuir nomes significativos às colunas"""
        if df is None:
            return df
        
        new_names = {}
        
        for col in df.columns:
            values = df[col]
            
            # RPM
            if (values.min() >= 300 and values.max() <= 8000 and 
                values.std() > 100 and values.mean() > 500):
                new_names[col] = 'RPM'
                
            # Lambda
            elif (values.min() >= 0.3 and values.max() <= 3.0 and 
                  values.std() > 0.05 and values.mean() > 0.5):
                new_names[col] = 'Lambda'
                
            # MAP
            elif (values.min() >= 5 and values.max() <= 400 and 
                  values.std() > 5 and values.mean() > 20):
                new_names[col] = 'MAP'
                
            # TPS
            elif (values.min() >= -5 and values.max() <= 110 and 
                  values.std() > 2):
                new_names[col] = 'TPS'
        
        if new_names:
            print(f"🏷️ Colunas identificadas: {list(new_names.values())}")
            df = df.rename(columns=new_names)
        
        return df
    
    def _show_data_stats(self, df):
        """Mostra estatísticas básicas dos dados"""
        print(f"\n📊 Estatísticas dos dados:")
        print(f"{'Canal':<15} {'Min':<10} {'Max':<10} {'Média':<10} {'Std':<10}")
        print("-" * 60)
        
        for col in df.columns[:10]:  # Primeiros 10 canais
            if df[col].std() > 0:  # Apenas canais com variação
                print(f"{col:<15} {df[col].min():<10.2f} {df[col].max():<10.2f} "
                      f"{df[col].mean():<10.2f} {df[col].std():<10.2f}")

def test_mlg_analyzer():
    """Testa o analisador MLG"""
    analyzer = MLGAnalyzer()
    file_path = '/home/robca/obd2/logs/2025-08-06_00.09.39.mlg'
    
    df = analyzer.extract_data(file_path)
    
    if df is not None:
        print(f"\n🎯 Dados finais:")
        print(f"Shape: {df.shape}")
        print(f"Primeiras linhas:")
        print(df.head())
        
        # Salva para debug
        df.to_csv('/home/robca/obd2/mlg_extracted_data.csv', index=False)
        print(f"💾 Dados salvos em: mlg_extracted_data.csv")
    
    return df

if __name__ == "__main__":
    test_mlg_analyzer()
