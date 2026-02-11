#!/usr/bin/env python3
import obd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv


# car = 'Fit'
# commands = [
#     "RPM",
#     "INTAKE_PRESSURE",
#     "TIMING_ADVANCE",
#     "COOLANT_TEMP",
#     "INTAKE_TEMP",
#     "SPEED",
#     "SHORT_FUEL_TRIM_1",
#     "O2_S1_WR_CURRENT",
#     "RELATIVE_THROTTLE_POS",
# ]

car = 'Omega2005'
commands = [
    "O2_S1_WR_CURRENT",  # Lambda11 - Banco 1 (valor nativo)
    "O2_S5_WR_CURRENT",  # Lambda21 - Banco 2 (valor nativo)
]

log_file_path = f"logs/{car}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
# log_file_path = f"../logs/{car}_nha.csv"

# Abrir o arquivo para gravação
log_file = open(log_file_path, mode="w", newline="")
writer = csv.writer(log_file)

# Escrever o cabeçalho
header = commands + ["timestamp"]
writer.writerow(header)


# commands = [
#     'RPM',
#     'COOLANT_TEMP',
#     # 'MAF',
#     # 'THROTTLE_POS',
#     'INTAKE_TEMP',
#     'TIMING_ADVANCE',
#     'ENGINE_LOAD',
#     'ELM_VOLTAGE',
#     # 'SPEED',
#     # 'O2_S1_WR_CURRENT',
#     # 'O2_S5_WR_CURRENT',
#     # 'O2_B1S2',
#     # 'O2_B2S2',
#     'SHORT_FUEL_TRIM_1',
#     # 'SHORT_FUEL_TRIM_2',
#     'LONG_FUEL_TRIM_1',
#     # 'LONG_FUEL_TRIM_2'
# ]

# obd.logger.setLevel(obd.logging.DEBUG)
# conn = obd.OBD('socket://192.168.0.10:35000')
# obd.logger.removeHandler(obd.console_handler)
# conn = obd.Async('/dev/ttyUSB0')
conn = obd.OBD("/dev/ttyUSB0")

# Verificar conexão
if not conn.is_connected():
    print("Not connected to OBD2")
    exit()
else:
    print("Connected to OBD2!")
    print(f"Protocol: {conn.protocol_name()}")
    
# Verificar quais comandos são suportados
print("\nSupported commands:")
for cmd in commands:
    if obd.commands[cmd] in conn.supported_commands:
        print(f"  ✓ {cmd}")
    else:
        print(f"  ✗ {cmd} - NOT SUPPORTED")

print(f"\nTotal supported commands: {len(conn.supported_commands)}")
print("\nStarting data collection...")

# print('Supported Commands:')
# for i in conn.supported_commands:
#     print(i)

# Função para converter corrente lambda nativa (mA) para lambda
def current_to_lambda(current_ma):
    """
    Os comandos O2_S1_WR_CURRENT e O2_S5_WR_CURRENT já fornecem
    valores nativos de corrente lambda da central.
    
    Baseado no teste: 
    - Sensor 1: 0.0 mA
    - Sensor 5: 0.0078125 mA
    
    Estes valores são diretamente proporcionais ao lambda.
    """
    if current_ma is None:
        return None
    
    # Os valores já vêm da central como corrente lambda
    # Conversão direta proporcional (calibração pode precisar ajuste)
    
    # Para valores muito pequenos (próximos de 0), assumir lambda ~1.0
    if abs(current_ma) < 0.001:  # Valores menores que 1mA
        return 1.0
    
    # Conversão baseada na resposta da central
    # Valores observados: -0.090mA a +0.109mA
    # Escala corrigida: ±0.1mA ≈ ±0.3λ (faixa típica 0.7-1.3)
    lambda_offset = current_ma * 3  # Escala corrigida baseada nos dados reais
    return 1.0 + lambda_offset

# Dados para o gráfico
data = {cmd: [] for cmd in commands}
data['LAMBDA11'] = []  # Lambda calculado da sonda 1
data['LAMBDA21'] = []  # Lambda calculado da sonda 2
# timestamps = []


# Função para atualizar os dados
def update_data(frame):
    global data
    log = []
    current_time = datetime.now().timestamp()

    # current_time = datetime.now().timestamp()
    # timestamps.append(current_time)

    for cmd in commands:
        try:
            result = conn.query(obd.commands[cmd])
            if result.value is not None:
                # Tratamento especial para temperaturas (unidades com offset)
                if cmd in ['COOLANT_TEMP', 'INTAKE_TEMP', 'AMBIENT_AIR_TEMP']:
                    # Para temperaturas, converter para Celsius e pegar apenas o valor numérico
                    if hasattr(result.value, 'to'):
                        value = float(result.value.to('celsius').magnitude)
                    else:
                        value = float(result.value.magnitude)
                else:
                    # Para outros comandos, usar magnitude normalmente
                    value = result.value.magnitude if hasattr(result.value, 'magnitude') else result.value
                
                # Calcular lambda para sensores de O2
                if cmd == 'O2_S1_WR_CURRENT':
                    lambda11 = current_to_lambda(value)
                    data['LAMBDA11'].append(lambda11)
                    print(f"Lambda11: {lambda11:.3f} (current: {value:.3f}mA)")
                elif cmd == 'O2_S5_WR_CURRENT':
                    lambda21 = current_to_lambda(value)
                    data['LAMBDA21'].append(lambda21)
                    print(f"Lambda21: {lambda21:.3f} (current: {value:.3f}mA)")
                else:
                    print(f"{cmd}: {value}")  # Debug: mostrar valor lido
            else:
                value = None
                if cmd == 'O2_S1_WR_CURRENT':
                    data['LAMBDA11'].append(None)
                    print("Lambda11: NULL")
                elif cmd == 'O2_S5_WR_CURRENT':
                    data['LAMBDA21'].append(None)
                    print("Lambda21: NULL")
                else:
                    print(f"{cmd}: NULL result")  # Debug: mostrar quando não há valor
            data[cmd].append(value)
            log.append(value)
        except Exception as ex:
            print(f"Error in {cmd} command: {ex}")
            data[cmd].append(None)
            log.append(None)

    # Adicionar o timestamp aos dados
    log.append(current_time)

    # Gravar os dados no arquivo
    writer.writerow(log)

    # Limitar o número de pontos no gráfico
    max_points = 50  # Aumentar pontos para melhor visualização
    # if len(timestamps) > max_points:
    #     timestamps = timestamps[-max_points:]
    for cmd in commands:
        data[cmd] = data[cmd][-max_points:]
    # Limitar também os dados de lambda
    data['LAMBDA11'] = data['LAMBDA11'][-max_points:]
    data['LAMBDA21'] = data['LAMBDA21'][-max_points:]

# Função para atualizar o gráfico
def update_plot(frame):
    update_data(frame)
    ax1.clear()  # LAMBDA (eixo principal)
    ax2.clear()  # RPM
    ax3.clear()  # TIMING_ADVANCE  
    ax4.clear()  # COOLANT_TEMP

    # Gráfico PRINCIPAL para LAMBDA (eixo y principal)
    x_axis = range(len(data["LAMBDA11"]))
    ax1.plot(x_axis, data["LAMBDA11"], label="Lambda11 (Banco 1)", color="green", linewidth=2)
    ax1.plot(x_axis, data["LAMBDA21"], label="Lambda21 (Banco 2)", color="red", linewidth=2)
    ax1.set_ylabel("Lambda (λ)", color="green", fontsize=12, fontweight='bold')
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.set_ylim(0.6, 1.4)  # Faixa típica de lambda para visualização
    ax1.grid(True, alpha=0.3)
    
    # Linhas de referência lambda
    ax1.axhline(y=0.85, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Rico (0.85λ)")
    ax1.axhline(y=1.00, color="black", linestyle="-", linewidth=1.5, alpha=0.8, label="Estequiométrico (1.0λ)")
    ax1.axhline(y=1.15, color="blue", linestyle="--", linewidth=1, alpha=0.7, label="Pobre (1.15λ)")

    # Exibir valores atuais de lambda
    if data["LAMBDA11"] and data["LAMBDA11"][-1] is not None:
        ax1.text(
            len(data["LAMBDA11"]) - 1, data["LAMBDA11"][-1] + 0.05,
            f"λ11: {data['LAMBDA11'][-1]:.3f}",
            color="green", fontsize=12, ha="right", fontweight='bold'
        )
    if data["LAMBDA21"] and data["LAMBDA21"][-1] is not None:
        ax1.text(
            len(data["LAMBDA21"]) - 1, data["LAMBDA21"][-1] - 0.05,
            f"λ21: {data['LAMBDA21'][-1]:.3f}",
            color="red", fontsize=12, ha="right", fontweight='bold'
        )

    # Gráfico para RPM (eixo secundário)
    ax2.plot(range(len(data["RPM"])), data["RPM"], label="RPM", color="blue", alpha=0.7)
    ax2.set_ylabel("RPM", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_ylim(0, 6500)  # Definir escala fixa para o eixo y do RPM

    # Gráfico para TIMING_ADVANCE (menor prioridade)
    ax3.plot(range(len(data["TIMING_ADVANCE"])), data["TIMING_ADVANCE"], label="TIMING_ADVANCE", color="orange", alpha=0.6)
    ax3.set_ylabel("Timing (°)", color="orange", fontsize=10)
    ax3.tick_params(axis="y", labelcolor="orange")
    ax3.set_ylim(-30, 40)

    # Gráfico para COOLANT_TEMP (menor prioridade)
    ax4.plot(range(len(data["COOLANT_TEMP"])), data["COOLANT_TEMP"], label="COOLANT_TEMP", color="purple", alpha=0.6)
    ax4.set_ylabel("Temp (°C)", color="purple", fontsize=10)
    ax4.tick_params(axis="y", labelcolor="purple")
    ax4.set_ylim(80, 120)  # Faixa típica de temperatura

    # Configurações gerais
    ax1.set_xlabel("Amostras", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="lower left", fontsize=8)
    ax3.legend(loc="lower center", fontsize=8)
    ax4.legend(loc="lower right", fontsize=8)
    plt.title(f"Lambda Monitor - {car}", fontsize=14, fontweight='bold')
    # plt.tight_layout()  # Removido para melhorar o desempenho

# Configurar o gráfico
fig, ax1 = plt.subplots(figsize=(15, 9))  # Eixo principal para RPM
ax2 = ax1.twinx()  # Segundo eixo para TIMING_ADVANCE
ax3 = ax1.twinx()  # Terceiro eixo para COOLANT_TEMP
ax4 = ax1.twinx()  # Quarto eixo para O2_S5_WR_CURRENT

# Ajustar os eixos para não sobrepor
ax3.spines["right"].set_position(("outward", 60))  # Deslocar o terceiro eixo
ax4.spines["right"].set_position(("outward", 120))  # Deslocar o quarto eixo

# Ajustar manualmente os espaços
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.2)

ani = FuncAnimation(fig, update_plot, interval=20, cache_frame_data=False)  # 20ms = 50Hz para máxima responsividade

# Mostrar o gráfico
try:
    plt.show()
except KeyboardInterrupt:
    print("Stopped")
finally:
    log_file.close()  # Fechar o arquivo ao encerrar


# if conn.is_connected():
#     # for i in dir(obd.commands):
#     #     print(i)

#     # for c in commands:
#     #     c = c.strip('\n')
#     #     result = conn.query(obd.commands[c])
#     #     if result.is_null():
#     #         print(f'{c} inválido')
#     #         continue
#     #     try:
#     #         print(f'{datetime.now().timestamp()}\t{c}: {str(result.value.magnitude)}')
#     #     except Exception as ex:
#     #         if 'magnitude' in str(ex):
#     #                 print(f'{datetime.now().timestamp()}\t{c}: {str(result.value)}')
#     #         else:
#     #             print(f"Error in {c} command: {ex}")

#     # exit(9)

#     # file = open(f"../logs/{car}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log", 'w')
#     print('\nCollecting...')
#     try:
#         head = [c for c in commands]
#         head.append('timestamp')
#         # file.write(','.join(head)+'\n')
#         # for c in commands:
#         #     conn.watch(obd.commands[c])
#         # conn.start()
#         time.sleep(1)
#         while True:
#             log = []
#             status = conn.query(obd.commands.FUEL_STATUS).value[0]
#             # if not 'Closed' in status:
#             #     print(status)
#             #     time.sleep(0.5)
#             #     continue
#             for c in commands:
#                 try:
#                     log.append(str(conn.query(obd.commands[c]).value.magnitude))
#                 except Exception as ex:
#                     print(f"Error in {c} command: {ex}")
#                     log.append('')
#             log.append(str(datetime.now().timestamp()))
#             print(log)
#             # file.write(','.join(log)+'\n')
#             time.sleep(0.3)
#     except KeyboardInterrupt:
#         print("Stopped")
#     # file.close()
# else:
#     print('Not connected')

conn.close()

# conn.stop()

    # with open('../logs/fit_commands.txt') as f:



# # Scheduler 
# def repeat():
#   threading.Timer(10.0, repeat).start()
#   speedCmd = connection.query(obd.commands.SPEED)
#   speedVal = str(speedCmd.value)
#   fuelCmd = connection.query(obd.commands.FUEL_LEVEL)
#   fuelVal = str(fuelCmd.value)
#   print("Speed: " + speedVal + ", fuel: " + fuelVal)
#   upload(speedVal, fuelVal)
# repeat()


# import obd_io
# import serial
# import platform
# import obd_sensors
# from datetime import datetime
# import time

# from obd_utils import scanSerial


# class OBD_Recorder():
#     def __init__(self, path, log_items):
#         self.port = None
#         self.sensorlist = []
#         localtime = time.localtime(time.time())
#         filename = path+"CivicSI-"+str(localtime[0])+"-"+str(localtime[1])+"-"+str(localtime[2])+"-"+str(localtime[3])+"-"+str(localtime[4])+"-"+str(localtime[5])+".log"
# 	#filename = path+"1st-"+str(localtime[0])+"-"+str(localtime[1])+"-"+str(localtime[2])+"-"+str(localtime[3])+"-"+str(localtime[4])+"-"+str(localtime[5])+".log"
# 	#filename = path+"2nd-"+str(localtime[0])+"-"+str(localtime[1])+"-"+str(localtime[2])+"-"+str(localtime[3])+"-"+str(localtime[4])+"-"+str(localtime[5])+".log"
# 	#filename = path+"3rd-"+str(localtime[0])+"-"+str(localtime[1])+"-"+str(localtime[2])+"-"+str(localtime[3])+"-"+str(localtime[4])+"-"+str(localtime[5])+".log"
# 	#filename = path+"4th-"+str(localtime[0])+"-"+str(localtime[1])+"-"+str(localtime[2])+"-"+str(localtime[3])+"-"+str(localtime[4])+"-"+str(localtime[5])+".log"

#         self.log_file = open(filename, "w", 128)
#         self.log_file.write("Time, RPM, MPH, short term fuel trim, long term fuel trim, Throttle, Gear\n");

#         while 1:
#             localtime = datetime.now()
#             current_time = str(localtime.hour)+":"+str(localtime.minute)+":"+str(localtime.second)+"."+str(localtime.microsecond)
#             log_string = current_time
#             results = {}
#             for index in self.sensorlist:
#                 (name, value, unit) = self.port.sensor(index)
#                 log_string = log_string + ","+str(value)
#                 results[obd_sensors.SENSORS[index].shortname] = value;

#             gear = self.calculate_gear(results["rpm"], results["speed"])
#             log_string = log_string + "," + str(gear)
#             self.log_file.write(log_string+"\n")
            
#     def calculate_gear(self, rpm, speed):
#         if speed == "" or speed == 0:
#             return 0
#         if rpm == "" or rpm == 0:
#             return 0

#         rps = rpm/60
#         mps = (speed*0.44704) #meters per second
        
#         final_drive  = 4.765
        
#         tire_circumference = 1.964 #meters

#         current_gear_ratio = (rps / (mps / tire_circumference)) / final drive
        
#         print current_gear_ratio
	
# 	#gear = min((abs(current_gear_ratio - i), i) for i in self.gear_ratios)[1] 
#         #return gear
            
            
