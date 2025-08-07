# 🚗 Otimizador de Mapa VE com Inteligência Artificial

Sistema completo para otimização automática de mapas de eficiência volumétrica (VE) em injeção eletrônica programável, usando dados de sensor O2 wideband e machine learning.

## 📋 Visão Geral

Este projeto implementa um modelo de IA que analisa dados de log da ECU para prever ajustes necessários no mapa VE, considerando:

- **Delay da sonda O2** no escapamento
- **Lambda target** por zona de operação
- **Correlação** entre VE atual e lambda medido
- **Condições de operação** do motor

### 🔧 Especificações do Motor Suportado

- **Cilindrada**: 2000cc
- **Cilindros**: 4
- **Bicos Injetores**: 4 x 19lb/h
- **Modo de Injeção**: Banco a banco (2 injeções por ciclo)
- **Combustível**: Gasolina (AFR estequiométrico: 14.7)

## 📁 Estrutura do Projeto

```
obd2/
├── VE_Map_Optimizer_Analysis.ipynb  # Notebook interativo principal
├── ve_map_optimizer.py              # Script Python standalone
├── logs/
│   ├── example.log                  # Dados de exemplo da ECU
│   ├── ve.table                     # Tabela VE original (XML)
│   └── *.mlg                        # Logs binários TunerStudio
└── README.md                        # Este arquivo
```

## 🚀 Como Usar

### Opção 1: Jupyter Notebook (Recomendado)

1. **Abra o notebook**:
   ```
   VE_Map_Optimizer_Analysis.ipynb
   ```

2. **Execute as células em ordem** para:
   - Carregar e processar dados
   - Estimar delay da sonda O2
   - Treinar modelos de IA
   - Gerar tabela VE otimizada

### Opção 2: Script Python

1. **Execute o script**:
   ```bash
   python3 ve_map_optimizer.py
   ```

2. **Saídas geradas**:
   - `ve_optimized.table` - Nova tabela VE
   - `ve_analysis.png` - Gráficos de análise

## 📊 Formato dos Dados de Entrada

### Arquivo de Log (.log/.csv)
```csv
RPM,COOLANT_TEMP,MAF,THROTTLE_POS,INTAKE_TEMP,TIMING_ADVANCE,ENGINE_LOAD,ELM_VOLTAGE,SPEED,O2_S1_WR_CURRENT,O2_S5_WR_CURRENT,O2_B2S2,SHORT_FUEL_TRIM_1,SHORT_FUEL_TRIM_2
916.0,90,4.25,12.5,27,21.5,19.2,13.7,26.0,-0.019,-0.019,0.445,-0.78,-0.78
...
```

### Tabela VE (XML)
```xml
<tableData xmlns="http://www.EFIAnalytics.com/:table">
    <table cols="16" rows="16">
        <xAxis name="rpm">500 700 900 ...</xAxis>
        <yAxis name="fuelLoad">14 20 26 ...</yAxis>
        <zValues>8.0 10.0 12.0 ...</zValues>
    </table>
</tableData>
```

## 🤖 Algoritmo de Otimização

### 1. Análise de Delay da Sonda
- Detecta eventos de mudança rápida no TPS
- Correlaciona com resposta do lambda
- Estima delay temporal em segundos

### 2. Cálculo de Lambda Target
```python
def calculate_lambda_target(fuel_load):
    if fuel_load < 25:
        return 1.0      # Baixa carga - estequiométrico
    elif fuel_load < 60:
        return 0.95     # Carga média
    elif fuel_load < 80:
        return 0.88     # Carga alta
    else:
        return 0.85     # WOT - proteção
```

### 3. Modelo de Machine Learning

**Features utilizadas**:
- RPM, FuelLoad, VE_Current
- Lambda_Delayed, Lambda_Target
- Temperaturas (motor, ar)
- Avanço de ignição
- Features de interação (RPM×Load)

**Modelos testados**:
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression

### 4. Predição de Ajustes VE

```python
VE_novo = VE_atual × (Lambda_target / Lambda_medido)
```

Com limitadores de segurança (±25% do valor original).

## 📈 Interpretação dos Resultados

### Visualizações Geradas

1. **Mapa VE Original vs Otimizado**
2. **Mapa de Diferenças** (mudanças aplicadas)
3. **Distribuição de Correções**
4. **Análise por Zona de RPM/Carga**
5. **Mapa de Confiança** (densidade de dados)

### Métricas de Avaliação

- **R² Score**: Qualidade da predição (>0.8 é bom)
- **MSE/MAE**: Erro médio das predições
- **Células modificadas**: % da tabela alterada
- **Mudanças significativas**: Ajustes >2%

## ⚠️ Avisos de Segurança

### 🔴 CRÍTICO
- **Sempre monitore EGT** (temperatura gases escape)
- **Teste em condições controladas** antes de usar na rua
- **Monitore knock/detonação** em tempo real
- **Mantenha tune original** como backup

### 🟡 Recomendações
- Aplique mudanças **incrementalmente** (25% por vez)
- Valide com **wideband O2** em tempo real
- Teste em **dinamômetro** quando possível
- Revise **pressure/temperature limits**

## 🛠️ Instalação de Dependências

```bash
# Dependências Python
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Para Jupyter Notebook
pip install jupyter ipykernel
```

## 📝 Logs de Exemplo

O arquivo `logs/example.log` contém dados reais de:
- Sessão de 10+ minutos
- Condições variadas (idle, cruise, aceleração)
- Motor aquecido (>80°C)
- Dados de O2 wideband válidos

## 🔬 Validação do Modelo

### Critérios de Qualidade
- **R² > 0.75**: Predições confiáveis
- **Resíduos normais**: Sem viés sistemático
- **Cross-validation**: Modelo generaliza bem
- **Feature importance**: Features físicas relevantes

### Limitações
- Requer **dados de qualidade** (motor aquecido, O2 funcionando)
- **Delay da sonda** pode variar com temperatura
- **Lambda targets** são estimativas (ajustar conforme necessário)
- **Não substitui** conhecimento de tuning

## 📊 Exemplo de Resultado

```
📊 RELATÓRIO FINAL DE OTIMIZAÇÃO VE
============================================================
🔢 Total de células na tabela: 256
📈 Células modificadas (>0.1%): 89 (34.8%)
⚡ Mudanças significativas (>2%): 23 (9.0%)
📊 Mudança média: +1.2% ± 3.4%
📈 Maior aumento: +8.5%
📉 Maior redução: -6.2%

🎯 Análise por zona de carga:
  Baixa (0-40 kPa): +0.5% ± 2.1%
  Média (40-70 kPa): +1.8% ± 3.9%
  Alta (70+ kPa): +2.1% ± 4.2%

🤖 Modelo utilizado: Random Forest
⏱️  Delay da sonda estimado: 0.85s
```

## 🤝 Contribuições

Contribuições são bem-vindas! Areas de interesse:
- Suporte para outros formatos de log
- Modelos de IA mais avançados
- Interface gráfica
- Validação em bancada

## 📄 Licença

Este projeto é fornecido "como está" para fins educacionais e de pesquisa. 

**Use por sua própria conta e risco!**

---

*Desenvolvido para auxiliar tuners e entusiastas de injeção eletrônica programável.*
