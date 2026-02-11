#!/usr/bin/env python3
import obd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

car = 'Omega2005'
commands = [
    "O2_S1_WR_CURRENT",  # Lambda11 - Banco 1 (valor nativo)
    "O2_S5_WR_CURRENT",  # Lambda21 - Banco 2 (valor nativo)
]

log_file_path = f"logs/{car}_lambda_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"

# Abrir o arquivo para gravação
log_file = open(log_file_path, mode="w", newline="")
writer = csv.writer(log_file)

# Escrever o cabeçalho
header = ["LAMBDA11", "LAMBDA21", "timestamp"]
writer.writerow(header)

# Conectar ao OBD2
conn = obd.OBD("/dev/ttyUSB0")

# Verificar conexão
if not conn.is_connected():
    print("Not connected to OBD2")
    exit()
else:
    print("Connected to OBD2!")
    print(f"Protocol: {conn.protocol_name()}")
    
# Verificar quais comandos são suportados
print("\nSupported lambda commands:")
for cmd in commands:
    if obd.commands[cmd] in conn.supported_commands:
        print(f"  ✓ {cmd}")
    else:
        print(f"  ✗ {cmd} - NOT SUPPORTED")

print(f"\nTotal supported commands: {len(conn.supported_commands)}")
print("\nStarting lambda data collection...")

# Dados para o gráfico - apenas lambda
data = {
    'LAMBDA11': [],  # Lambda nativo sonda 1
    'LAMBDA21': []   # Lambda nativo sonda 2
}

# Função para atualizar os dados
def update_data(frame):
    global data
    log = []
    current_time = datetime.now().timestamp()

    # Coletar lambda11 (Banco 1)
    try:
        result = conn.query(obd.commands['O2_S1_WR_CURRENT'])
        if result.value is not None:
            current11 = result.value.magnitude if hasattr(result.value, 'magnitude') else result.value
            # Converter corrente para lambda (calibração precisa)
            if current11 <= 0:
                # Lado rico: -0.4mA=0.9λ, 0mA=1.0λ
                lambda11 = 1.0 + (current11 * 0.25)  # -0.4mA * 0.25 = -0.1λ -> 0.9λ
            else:
                # Lado pobre: +0.3mA=1.15λ, +0.5mA=1.25λ
                lambda11 = 1.0 + (current11 * 0.5)   # +0.3mA * 0.5 = +0.15λ -> 1.15λ
            data['LAMBDA11'].append(lambda11)
            log.append(current11)  # Salvar corrente no log
            print(f"Lambda11: {current11:.6f}mA -> {lambda11:.3f}λ")
        else:
            data['LAMBDA11'].append(None)
            log.append(None)
            print("Lambda11: NULL")
    except Exception as ex:
        print(f"Error Lambda11: {ex}")
        data['LAMBDA11'].append(None)
        log.append(None)
    
    # Coletar lambda21 (Banco 2)
    try:
        result = conn.query(obd.commands['O2_S5_WR_CURRENT'])
        if result.value is not None:
            current21 = result.value.magnitude if hasattr(result.value, 'magnitude') else result.value
            # Converter corrente para lambda (calibração precisa)
            if current21 <= 0:
                # Lado rico: -0.4mA=0.9λ, 0mA=1.0λ
                lambda21 = 1.0 + (current21 * 0.25)  # -0.4mA * 0.25 = -0.1λ -> 0.9λ
            else:
                # Lado pobre: +0.3mA=1.15λ, +0.5mA=1.25λ
                lambda21 = 1.0 + (current21 * 0.5)   # +0.3mA * 0.5 = +0.15λ -> 1.15λ
            data['LAMBDA21'].append(lambda21)
            log.append(current21)  # Salvar corrente no log
            print(f"Lambda21: {current21:.6f}mA -> {lambda21:.3f}λ")
        else:
            data['LAMBDA21'].append(None)
            log.append(None)
            print("Lambda21: NULL")
    except Exception as ex:
        print(f"Error Lambda21: {ex}")
        data['LAMBDA21'].append(None)
        log.append(None)

    # Adicionar o timestamp aos dados
    log.append(current_time)

    # Gravar os dados no arquivo
    writer.writerow(log)

    # Limitar o número de pontos no gráfico
    max_points = 100  # Mais pontos para melhor visualização
    data['LAMBDA11'] = data['LAMBDA11'][-max_points:]
    data['LAMBDA21'] = data['LAMBDA21'][-max_points:]

# Função para atualizar o gráfico
def update_plot(frame):
    update_data(frame)
    ax.clear()  # Limpar gráfico

    # Gráfico ÚNICO para LAMBDA com escala automática
    x_axis = range(len(data["LAMBDA11"]))
    
    # Plotar as duas curvas lambda
    ax.plot(x_axis, data["LAMBDA11"], label="Banco 1", color="green", linewidth=3, marker='o', markersize=2)
    ax.plot(x_axis, data["LAMBDA21"], label="Banco 2", color="red", linewidth=3, marker='s', markersize=2)
    
    # Configurar eixos e labels
    ax.set_ylabel("Lambda (λ)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Amostras", fontsize=12)
    
    # Escala FIXA para lambda (0.5 a 1.5)
    ax.set_ylim(0.5, 1.5)
    
    # Grid para melhor visualização
    ax.grid(True, alpha=0.3)
    
    # Linhas de referência LAMBDA
    ax.axhline(y=0.9, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="λ=0.9")
    ax.axhline(y=1.15, color="blue", linestyle="--", linewidth=1.5, alpha=0.7, label="λ=1.15")
    
    # Exibir valores atuais grandes e bem visíveis com valor em mA
    if data["LAMBDA11"] and data["LAMBDA11"][-1] is not None:
        # Calcular corrente original baseado na calibração precisa
        if data["LAMBDA11"][-1] <= 1.0:
            current11_display = (data["LAMBDA11"][-1] - 1.0) / 0.25  # Rico
        else:
            current11_display = (data["LAMBDA11"][-1] - 1.0) / 0.5   # Pobre
        ax.text(
            0.02, 0.95, f"λ11: {data['LAMBDA11'][-1]:.3f} ({current11_display:.3f}mA)",
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            color="green", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
        )
    
    if data["LAMBDA21"] and data["LAMBDA21"][-1] is not None:
        # Calcular corrente original baseado na calibração precisa
        if data["LAMBDA21"][-1] <= 1.0:
            current21_display = (data["LAMBDA21"][-1] - 1.0) / 0.25  # Rico
        else:
            current21_display = (data["LAMBDA21"][-1] - 1.0) / 0.5   # Pobre
        ax.text(
            0.02, 0.85, f"λ21: {data['LAMBDA21'][-1]:.3f} ({current21_display:.3f}mA)",
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            color="red", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
        )

    # Configurações finais do gráfico
    ax.legend(loc="upper right", fontsize=12)
    plt.title(f"Monitor Corrente Lambda - {car} (Valores Nativos OBD2)", fontsize=16, fontweight='bold')
    plt.tight_layout()

# Configurar o gráfico - apenas um eixo para lambda
fig, ax = plt.subplots(figsize=(16, 10))  # Gráfico grande para melhor visualização

# Ajustar espaçamento
plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)

ani = FuncAnimation(fig, update_plot, interval=50, cache_frame_data=False)  # 50ms para boa responsividade

# Mostrar o gráfico
try:
    plt.show()
except KeyboardInterrupt:
    print("Stopped")
finally:
    log_file.close()  # Fechar o arquivo ao encerrar
    conn.close()

print("Monitor de Lambda encerrado.")