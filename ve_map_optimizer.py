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
from scipy import interpolate
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class VEMapOptimizer:
    """
    Classe para otimização do mapa VE baseado em dados de lambda do sensor O2 wideband
    
    Especificações do motor:
    - Tamanho: 2000cc
    - Cilindros: 4
    - Bicos: 4 (19lb/h cada)
    - Modo de injeção: Banco a banco (2 injeções por ciclo por banco)
    """
    
    def __init__(self):
        self.engine_displacement = 2000  # cc
        self.cylinders = 4
        self.injectors = 4
        self.injector_flow = 19  # lb/h
        self.injection_mode = "bank_to_bank"  # 2 injeções por ciclo por banco
        
        # Parâmetros para estimativa de delay da sonda
        self.estimated_delay = 0.0  # será calculado
        self.delay_samples = 0
        
        # Dados carregados
        self.log_data = None
        self.ve_table = None
        self.rpm_bins = None
        self.load_bins = None
        
        # Modelo treinado
        self.model = None
        self.scaler = StandardScaler()
        
    def load_log_data(self, log_file_path):
        """Carrega dados do arquivo de log CSV"""
        try:
            # Lê o arquivo CSV
            columns = ['RPM', 'COOLANT_TEMP', 'MAF', 'THROTTLE_POS', 'INTAKE_TEMP', 
                      'TIMING_ADVANCE', 'ENGINE_LOAD', 'ELM_VOLTAGE', 'SPEED', 
                      'O2_S1_WR_CURRENT', 'O2_S5_WR_CURRENT', 'O2_B2S2', 
                      'SHORT_FUEL_TRIM_1', 'SHORT_FUEL_TRIM_2']
            
            self.log_data = pd.read_csv(log_file_path, names=columns)
            
            # Calcula lambda a partir do AFR (assumindo O2_B2S2 como AFR)
            # Lambda = AFR_atual / AFR_estequiométrico (14.7 para gasolina)
            self.log_data['Lambda'] = self.log_data['O2_B2S2'] / 14.7
            
            # Calcula FuelLoad (pressão MAP como proxy)
            # Para simplificação, usaremos ENGINE_LOAD como FuelLoad
            self.log_data['FuelLoad'] = self.log_data['ENGINE_LOAD']
            
            # Adiciona timestamp para análise de delay
            self.log_data['Time'] = np.arange(len(self.log_data)) * 0.1  # assumindo 10Hz
            
            print(f"Log carregado com {len(self.log_data)} registros")
            return True
            
        except Exception as e:
            print(f"Erro ao carregar log: {e}")
            return False
    
    def load_ve_table(self, ve_table_path):
        """Carrega a tabela VE do arquivo XML"""
        try:
            tree = ET.parse(ve_table_path)
            root = tree.getroot()
            
            # Namespace do XML
            ns = {'table': 'http://www.EFIAnalytics.com/:table'}
            
            # Extrai eixo X (RPM)
            x_axis = root.find('.//table:xAxis', ns)
            rpm_text = x_axis.text.strip().split()
            self.rpm_bins = [float(x) for x in rpm_text]
            
            # Extrai eixo Y (FuelLoad)
            y_axis = root.find('.//table:yAxis', ns)
            load_text = y_axis.text.strip().split()
            self.load_bins = [float(x) for x in load_text]
            
            # Extrai valores Z (VE)
            z_values = root.find('.//table:zValues', ns)
            ve_text = z_values.text.strip().split()
            
            # Converte para matriz
            ve_matrix = []
            idx = 0
            for row in range(len(self.load_bins)):
                ve_row = []
                for col in range(len(self.rpm_bins)):
                    ve_row.append(float(ve_text[idx]))
                    idx += 1
                ve_matrix.append(ve_row)
            
            self.ve_table = np.array(ve_matrix)
            
            print(f"Tabela VE carregada: {len(self.load_bins)}x{len(self.rpm_bins)}")
            return True
            
        except Exception as e:
            print(f"Erro ao carregar tabela VE: {e}")
            return False
    
    def estimate_sensor_delay(self):
        """Estima o delay da sonda O2 baseado em eventos de mudança rápida no TPS"""
        if self.log_data is None:
            return 0
        
        # Calcula taxa de mudança do TPS
        self.log_data['TPS_Rate'] = self.log_data['THROTTLE_POS'].diff() / 0.1  # por segundo
        
        # Identifica eventos de mudança rápida (>50%/s)
        rapid_events = self.log_data[abs(self.log_data['TPS_Rate']) > 50]
        
        if len(rapid_events) == 0:
            print("Nenhum evento rápido encontrado para estimar delay")
            return 0
        
        # Para cada evento, procura o pico de resposta no lambda
        delays = []
        
        for idx in rapid_events.index[:10]:  # Analisa até 10 eventos
            if idx + 50 < len(self.log_data):  # Garante janela de análise
                # Janela de 5 segundos após o evento
                window = self.log_data.loc[idx:idx+50]
                
                # Procura maior variação no lambda
                lambda_change = abs(window['Lambda'].diff()).max()
                if lambda_change > 0.02:  # Mudança significativa
                    peak_idx = abs(window['Lambda'].diff()).idxmax()
                    delay_samples = peak_idx - idx
                    delays.append(delay_samples)
        
        if delays:
            self.delay_samples = int(np.median(delays))
            self.estimated_delay = self.delay_samples * 0.1  # converte para segundos
            print(f"Delay estimado da sonda: {self.estimated_delay:.1f}s ({self.delay_samples} amostras)")
        else:
            print("Não foi possível estimar delay da sonda")
            
        return self.estimated_delay
    
    def interpolate_ve_current(self, rpm, fuel_load):
        """Interpola o valor VE atual da tabela para RPM e FuelLoad específicos"""
        if self.ve_table is None:
            return 50.0  # valor padrão
        
        # Interpolação bilinear
        interp_func = interpolate.interp2d(self.rpm_bins, self.load_bins, 
                                         self.ve_table, kind='linear')
        
        # Garante que os valores estão dentro dos limites
        rpm = np.clip(rpm, min(self.rpm_bins), max(self.rpm_bins))
        fuel_load = np.clip(fuel_load, min(self.load_bins), max(self.load_bins))
        
        return float(interp_func(rpm, fuel_load)[0])
    
    def calculate_lambda_target(self, rpm, fuel_load):
        """Calcula lambda target baseado em RPM e carga"""
        # Lambda target típico para motores aspirados:
        # - Marcha lenta e baixa carga: ~1.0 (estequiométrico)
        # - Carga média: 0.95-1.0
        # - Carga alta/WOT: 0.85-0.90 (rico para proteção)
        
        if fuel_load < 30:
            return 1.0  # Baixa carga - estequiométrico
        elif fuel_load < 70:
            return 0.95  # Carga média
        else:
            return 0.87  # Alta carga - rico
    
    def prepare_training_data(self):
        """Prepara dados para treinamento do modelo"""
        if self.log_data is None:
            raise ValueError("Dados de log não carregados")
        
        # Remove outliers
        data = self.log_data.copy()
        
        # Filtros de qualidade dos dados
        data = data[
            (data['RPM'] > 500) & (data['RPM'] < 7000) &
            (data['FuelLoad'] > 10) & (data['FuelLoad'] < 120) &
            (data['Lambda'] > 0.7) & (data['Lambda'] < 1.3) &
            (data['COOLANT_TEMP'] > 80)  # Motor aquecido
        ]
        
        # Aplica delay da sonda se estimado
        if self.delay_samples > 0:
            data['Lambda_Delayed'] = data['Lambda'].shift(-self.delay_samples)
            data = data[:-self.delay_samples]  # Remove registros sem lambda delayed
        else:
            data['Lambda_Delayed'] = data['Lambda']
        
        # Calcula VE atual da tabela
        data['VE_Current'] = data.apply(
            lambda row: self.interpolate_ve_current(row['RPM'], row['FuelLoad']), 
            axis=1
        )
        
        # Calcula lambda target
        data['Lambda_Target'] = data.apply(
            lambda row: self.calculate_lambda_target(row['RPM'], row['FuelLoad']),
            axis=1
        )
        
        # Calcula erro de lambda
        data['Lambda_Error'] = data['Lambda_Delayed'] - data['Lambda_Target']
        
        # Calcula correção necessária no VE
        # Correção baseada na teoria: se lambda é alto (pobre), precisa mais combustível (VE maior)
        data['VE_Correction_Factor'] = data['Lambda_Target'] / data['Lambda_Delayed']
        data['VE_New'] = data['VE_Current'] * data['VE_Correction_Factor']
        
        # Limita correções extremas
        data['VE_New'] = np.clip(data['VE_New'], data['VE_Current'] * 0.8, 
                                data['VE_Current'] * 1.2)
        
        return data
    
    def train_model(self, data):
        """Treina modelo de machine learning para prever ajustes VE"""
        # Features para o modelo
        features = ['RPM', 'FuelLoad', 'VE_Current', 'Lambda_Delayed', 'Lambda_Target',
                   'COOLANT_TEMP', 'INTAKE_TEMP', 'TIMING_ADVANCE']
        
        X = data[features].copy()
        y = data['VE_New'].copy()
        
        # Remove NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Normalização
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Treina diferentes modelos
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear': LinearRegression()
        }
        
        best_score = -np.inf
        best_model = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            mean_score = cv_scores.mean()
            
            print(f"{name}: CV Score = {-mean_score:.3f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
        
        # Treina o melhor modelo
        self.model = best_model
        self.model.fit(X_train, y_train)
        
        # Avaliação no conjunto de teste
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nMelhor modelo: {type(best_model).__name__}")
        print(f"MSE: {mse:.3f}")
        print(f"R²: {r2:.3f}")
        print(f"MAE: {mae:.3f}")
        
        return X, y, features
    
    def predict_ve_adjustment(self, rpm, fuel_load, current_ve, lambda_measured, 
                            lambda_target, coolant_temp=90, intake_temp=25, timing=25):
        """Prediz ajuste necessário no VE para um ponto específico"""
        if self.model is None:
            raise ValueError("Modelo não treinado")
        
        # Prepara features
        features = np.array([[rpm, fuel_load, current_ve, lambda_measured, 
                            lambda_target, coolant_temp, intake_temp, timing]])
        
        # Normaliza
        features_scaled = self.scaler.transform(features)
        
        # Predição
        new_ve = self.model.predict(features_scaled)[0]
        
        # Calcula fator de correção
        correction_factor = new_ve / current_ve
        
        return {
            'current_ve': current_ve,
            'predicted_ve': new_ve,
            'correction_factor': correction_factor,
            'percent_change': (correction_factor - 1) * 100
        }
    
    def generate_optimized_ve_table(self, data):
        """Gera nova tabela VE otimizada baseada nos dados de treinamento"""
        if self.ve_table is None or self.model is None:
            raise ValueError("Tabela VE ou modelo não carregados")
        
        new_ve_table = self.ve_table.copy()
        
        # Para cada célula da tabela
        for i, load in enumerate(self.load_bins):
            for j, rpm in enumerate(self.rpm_bins):
                # Encontra dados próximos a este ponto
                nearby_data = data[
                    (abs(data['RPM'] - rpm) < 200) & 
                    (abs(data['FuelLoad'] - load) < 10)
                ]
                
                if len(nearby_data) > 5:  # Dados suficientes
                    # Usa a média das correções preditas
                    lambda_target = self.calculate_lambda_target(rpm, load)
                    current_ve = self.ve_table[i, j]
                    
                    # Média das leituras lambda
                    avg_lambda = nearby_data['Lambda_Delayed'].mean()
                    
                    # Prediz nova VE
                    prediction = self.predict_ve_adjustment(
                        rpm, load, current_ve, avg_lambda, lambda_target
                    )
                    
                    new_ve_table[i, j] = prediction['predicted_ve']
        
        return new_ve_table
    
    def plot_analysis(self, data):
        """Gera gráficos de análise"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Lambda vs Lambda Target
        axes[0,0].scatter(data['Lambda_Target'], data['Lambda_Delayed'], alpha=0.5)
        axes[0,0].plot([0.8, 1.1], [0.8, 1.1], 'r--', label='Ideal')
        axes[0,0].set_xlabel('Lambda Target')
        axes[0,0].set_ylabel('Lambda Medido')
        axes[0,0].set_title('Lambda Medido vs Target')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # 2. Erro de Lambda por RPM
        axes[0,1].scatter(data['RPM'], data['Lambda_Error'], alpha=0.5)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('RPM')
        axes[0,1].set_ylabel('Erro Lambda (Medido - Target)')
        axes[0,1].set_title('Erro Lambda por RPM')
        axes[0,1].grid(True)
        
        # 3. Erro de Lambda por Carga
        axes[0,2].scatter(data['FuelLoad'], data['Lambda_Error'], alpha=0.5)
        axes[0,2].axhline(y=0, color='r', linestyle='--')
        axes[0,2].set_xlabel('Fuel Load')
        axes[0,2].set_ylabel('Erro Lambda (Medido - Target)')
        axes[0,2].set_title('Erro Lambda por Carga')
        axes[0,2].grid(True)
        
        # 4. VE Atual vs VE Predita
        axes[1,0].scatter(data['VE_Current'], data['VE_New'], alpha=0.5)
        min_ve = min(data['VE_Current'].min(), data['VE_New'].min())
        max_ve = max(data['VE_Current'].max(), data['VE_New'].max())
        axes[1,0].plot([min_ve, max_ve], [min_ve, max_ve], 'r--', label='Sem mudança')
        axes[1,0].set_xlabel('VE Atual')
        axes[1,0].set_ylabel('VE Predita')
        axes[1,0].set_title('VE Atual vs Predita')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # 5. Fator de Correção por RPM
        axes[1,1].scatter(data['RPM'], data['VE_Correction_Factor'], alpha=0.5)
        axes[1,1].axhline(y=1.0, color='r', linestyle='--')
        axes[1,1].set_xlabel('RPM')
        axes[1,1].set_ylabel('Fator Correção VE')
        axes[1,1].set_title('Fator Correção VE por RPM')
        axes[1,1].grid(True)
        
        # 6. Fator de Correção por Carga
        axes[1,2].scatter(data['FuelLoad'], data['VE_Correction_Factor'], alpha=0.5)
        axes[1,2].axhline(y=1.0, color='r', linestyle='--')
        axes[1,2].set_xlabel('Fuel Load')
        axes[1,2].set_ylabel('Fator Correção VE')
        axes[1,2].set_title('Fator Correção VE por Carga')
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/robca/obd2/ve_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_optimized_table(self, new_ve_table, output_path):
        """Salva a nova tabela VE otimizada em formato XML"""
        # Cria estrutura XML
        root = ET.Element('tableData')
        root.set('xmlns', 'http://www.EFIAnalytics.com/:table')
        
        # Bibliografia
        bib = ET.SubElement(root, 'bibliography')
        bib.set('author', 'VE Map Optimizer - AI Generated')
        bib.set('company', 'Auto-generated optimization based on O2 sensor data')
        bib.set('writeDate', pd.Timestamp.now().strftime('%a %b %d %H:%M:%S %Z %Y'))
        
        # Versão
        version = ET.SubElement(root, 'versionInfo')
        version.set('fileFormat', '1.0')
        
        # Tabela
        table = ET.SubElement(root, 'table')
        table.set('cols', str(len(self.rpm_bins)))
        table.set('rows', str(len(self.load_bins)))
        
        # Eixo X (RPM)
        x_axis = ET.SubElement(table, 'xAxis')
        x_axis.set('cols', str(len(self.rpm_bins)))
        x_axis.set('name', 'rpm')
        x_axis.text = '\\n' + ' \\n'.join([f' {rpm:.0f} ' for rpm in self.rpm_bins]) + ' \\n'
        
        # Eixo Y (FuelLoad)
        y_axis = ET.SubElement(table, 'yAxis')
        y_axis.set('name', 'fuelLoad')
        y_axis.set('rows', str(len(self.load_bins)))
        y_axis.text = '\\n' + ' \\n'.join([f' {load:.0f} ' for load in self.load_bins]) + ' \\n'
        
        # Valores Z (VE)
        z_values = ET.SubElement(table, 'zValues')
        z_values.set('cols', str(len(self.rpm_bins)))
        z_values.set('rows', str(len(self.load_bins)))
        
        # Formata valores VE
        ve_text = ''
        for row in new_ve_table:
            ve_text += ' ' + ' '.join([f'{ve:.1f}' for ve in row]) + ' \\n'
        
        z_values.text = '\\n' + ve_text + '      '
        
        # Salva arquivo
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        print(f"Nova tabela VE salva em: {output_path}")


def main():
    """Função principal para executar a otimização do mapa VE"""
    
    print("=== Otimizador de Mapa VE com IA ===")
    print("Baseado em dados de sensor O2 wideband\\n")
    
    # Inicializa otimizador
    optimizer = VEMapOptimizer()
    
    # Carrega dados
    print("1. Carregando dados...")
    log_file = "/home/robca/obd2/logs/example.log"
    ve_table_file = "/home/robca/obd2/logs/ve.table"
    
    if not optimizer.load_log_data(log_file):
        print("Erro ao carregar dados de log")
        return
    
    if not optimizer.load_ve_table(ve_table_file):
        print("Erro ao carregar tabela VE")
        return
    
    # Estima delay da sonda
    print("\\n2. Estimando delay da sonda O2...")
    delay = optimizer.estimate_sensor_delay()
    
    # Prepara dados de treinamento
    print("\\n3. Preparando dados de treinamento...")
    training_data = optimizer.prepare_training_data()
    print(f"Dados válidos para treinamento: {len(training_data)}")
    
    # Treina modelo
    print("\\n4. Treinando modelo de IA...")
    X, y, features = optimizer.train_model(training_data)
    
    # Gera análises
    print("\\n5. Gerando análises...")
    optimizer.plot_analysis(training_data)
    
    # Gera nova tabela VE otimizada
    print("\\n6. Gerando nova tabela VE otimizada...")
    new_ve_table = optimizer.generate_optimized_ve_table(training_data)
    
    # Salva nova tabela
    output_file = "/home/robca/obd2/ve_optimized.table"
    optimizer.save_optimized_table(new_ve_table, output_file)
    
    # Relatório de mudanças
    print("\\n7. Relatório de mudanças:")
    total_cells = len(optimizer.load_bins) * len(optimizer.rpm_bins)
    changes = np.sum(np.abs(new_ve_table - optimizer.ve_table) > 1.0)
    avg_change = np.mean(np.abs(new_ve_table - optimizer.ve_table))
    max_change = np.max(np.abs(new_ve_table - optimizer.ve_table))
    
    print(f"Células modificadas: {changes}/{total_cells} ({changes/total_cells*100:.1f}%)")
    print(f"Mudança média: {avg_change:.1f}%")
    print(f"Mudança máxima: {max_change:.1f}%")
    
    # Exemplo de predição
    print("\\n8. Exemplo de predição:")
    example_prediction = optimizer.predict_ve_adjustment(
        rpm=2000, fuel_load=50, current_ve=45, 
        lambda_measured=0.92, lambda_target=0.87
    )
    
    print(f"RPM: 2000, Carga: 50kPa")
    print(f"VE atual: {example_prediction['current_ve']:.1f}%")
    print(f"VE predita: {example_prediction['predicted_ve']:.1f}%")
    print(f"Mudança: {example_prediction['percent_change']:.1f}%")
    
    print("\\n✅ Otimização concluída!")
    print(f"Nova tabela VE salva em: {output_file}")
    print(f"Gráficos de análise salvos em: /home/robca/obd2/ve_analysis.png")

if __name__ == "__main__":
    main()
