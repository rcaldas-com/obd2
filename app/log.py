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
    "RPM",
    "TIMING_ADVANCE",
    # "COOLANT_TEMP",
    # "INTAKE_TEMP",
    # "RELATIVE_THROTTLE_POS",
    "O2_S1_WR_CURRENT",
    "O2_S5_WR_CURRENT",
    # "SHORT_FUEL_TRIM_1",
    # "SHORT_FUEL_TRIM_2"
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
    print("Not connected")
    exit()

# print('Supported Commands:')
# for i in conn.supported_commands:
#     print(i)

# Dados para o gráfico
data = {cmd: [] for cmd in commands}
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
            value = result.value.magnitude if result.value else None
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
    max_points = 20
    # if len(timestamps) > max_points:
    #     timestamps = timestamps[-max_points:]
    for cmd in commands:
        data[cmd] = data[cmd][-max_points:]

# Função para atualizar o gráfico
def update_plot(frame):
    update_data(frame)
    # ax1.clear()  # Limpar apenas o eixo principal (RPM)
    ax2.clear()  # Limpar apenas o eixo secundário (O2)
    ax3.clear()  # Limpar o terceiro eixo (TIMING_ADVANCE)

    # # Gráfico principal (eixo y para RPM)
    # ax1.plot(timestamps, data["RPM"], label="RPM", color="blue")
    # ax1.set_ylabel("RPM", color="blue")
    # ax1.tick_params(axis="y", labelcolor="blue")
    # ax1.set_ylim(0, 6500)  # Definir escala fixa para o eixo y do RPM

    # # Exibir o valor atual de RPM no gráfico
    # if data["RPM"] and data["RPM"][-1] is not None:
    #     ax1.text(
    #         timestamps[-1], data["RPM"][-1],
    #         f"{data['RPM'][-1]:.0f} RPM",
    #         color="blue", fontsize=10, ha="right"
    #     )

    # Gráfico para TIMING_ADVANCE (agora no lado esquerdo)
    # ax2.plot(timestamps, data["TIMING_ADVANCE"], label="TIMING_ADVANCE", color="orange")
    ax2.plot(range(len(data["TIMING_ADVANCE"])), data["TIMING_ADVANCE"], label="TIMING_ADVANCE", color="orange")
    ax2.set_ylabel("Timing Advance (°)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.set_ylim(-30, 40)  # Limites fixos de -30 a 40

    # Exibir o valor atual de TIMING_ADVANCE no gráfico
    if data["TIMING_ADVANCE"] and data["TIMING_ADVANCE"][-1] is not None:
        ax2.text(
            # timestamps[-1], data["TIMING_ADVANCE"][-1],
            len(data["TIMING_ADVANCE"]) - 1, data["TIMING_ADVANCE"][-1],  # Substituído timestamps[-1]
            f"{data['TIMING_ADVANCE'][-1]:.1f}°",
            color="orange", fontsize=10, ha="right"
        )

    # Gráfico para O2_S1_WR_CURRENT e O2_S5_WR_CURRENT (agora no lado direito)
    # ax3.plot(timestamps, data["O2_S5_WR_CURRENT"], label="O2_S5_WR_CURRENT", color="red")
    ax3.plot(range(len(data["O2_S1_WR_CURRENT"])), data["O2_S1_WR_CURRENT"], label="O2_S1_WR_CURRENT", color="green")
    ax3.plot(range(len(data["O2_S5_WR_CURRENT"])), data["O2_S5_WR_CURRENT"], label="O2_S5_WR_CURRENT", color="red")
    ax3.set_ylabel("O2 Sensor Current (mA)", color="green")
    ax3.tick_params(axis="y", labelcolor="green")
    ax3.set_ylim(-1, 1)  # Limites fixos de -1 a +1
    ax3.set_yticks([-1, -0.3, 0, 0.3, 1])  # Marcar -1, -0.3, 0, 0.3 e 1

    # Adicionar linhas horizontais nas marcas -0.3, 0 e +0.3
    ax3.axhline(y=-0.3, color="gray", linestyle="--", linewidth=0.8, label="-0.3")
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, label="0")
    ax3.axhline(y=0.3, color="gray", linestyle="--", linewidth=0.8, label="+0.3")

    # Exibir os valores atuais dos sensores de oxigênio no gráfico
    if data["O2_S1_WR_CURRENT"] and data["O2_S1_WR_CURRENT"][-1] is not None:
        ax3.text(
            # timestamps[-1], data["O2_S1_WR_CURRENT"][-1],
            len(data["O2_S1_WR_CURRENT"]) - 1, data["O2_S1_WR_CURRENT"][-1],  # Substituído timestamps[-1]
            f"{data['O2_S1_WR_CURRENT'][-1]:.2f} mA",
            color="green", fontsize=10, ha="right"
        )

    if data["O2_S5_WR_CURRENT"] and data["O2_S5_WR_CURRENT"][-1] is not None:
        ax3.text(
            # timestamps[-1], data["O2_S5_WR_CURRENT"][-1],
            len(data["O2_S5_WR_CURRENT"]) - 1, data["O2_S5_WR_CURRENT"][-1],  # Substituído timestamps[-1]
            f"{data['O2_S5_WR_CURRENT'][-1]:.2f} mA",
            color="red", fontsize=10, ha="right"
        )

    # Configurações gerais
    # ax1.set_xlabel("Timestamp")
    # ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    ax3.legend(loc="lower left")
    plt.title(f"Live Data for {car}")
    # plt.tight_layout()  # Removido para melhorar o desempenho

# Configurar o gráfico
# fig, ax1 = plt.subplots(figsize=(15, 9))  # Aumentar o tamanho da figura
# ax2 = ax1.twinx()  # Criar um segundo eixo y
# ax3 = ax1.twinx()  # Criar um terceiro eixo y
fig, ax2 = plt.subplots(figsize=(15, 9))  # Aumentar o tamanho da figura
ax3 = ax2.twinx()  # Criar um terceiro eixo y


# Ajustar o terceiro eixo para o lado direito
# ax3.spines["right"].set_position(("outward", 60))  # Deslocar o eixo para evitar sobreposição

# Ajustar manualmente os espaços
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

ani = FuncAnimation(fig, update_plot, interval=50, cache_frame_data=False)

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
            
            
