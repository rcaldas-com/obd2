#!/usr/bin/env python3
import obd
from datetime import datetime
import time
from tinydb import TinyDB, Query


car = 'Omega'
commands = [
    'RPM',
    'COOLANT_TEMP',
    'MAF',
    'THROTTLE_POS',
    'INTAKE_TEMP',
    'TIMING_ADVANCE',
    'ENGINE_LOAD',
    'ELM_VOLTAGE',
    'SPEED',
    'O2_S1_WR_CURRENT',
    'O2_S5_WR_CURRENT',
    # 'O2_B1S2',
    'O2_B2S2',
    'SHORT_FUEL_TRIM_1',
    # 'LONG_FUEL_TRIM_1',
    'SHORT_FUEL_TRIM_2',
    # 'LONG_FUEL_TRIM_2'
]

conn = obd.OBD('socket://192.168.0.10:35000')

if conn.is_connected():
    file = open(f"logs/{car}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log", 'w')
    print('\nCollecting...')
    try:
        file.write(','.join(commands)+'\n') # Head of csv
        time.sleep(1)
        while True:
            print(datetime.now().time()) # Just to see resulting interval
            log = []
            for c in commands:
                try:
                    log.append(str(conn.query(obd.commands[c]).value.magnitude))
                except Exception as ex:
                    print(f"Error in {c} command: {ex}")
                    log.append('')
            file.write(','.join(log)+'\n')
            # time.sleep(0.359) # ~ 0.997 / 2
    except KeyboardInterrupt:
        print("Stopped")
    file.close()
else:
    print('Not connected')
conn.close()



# def calculate_gear(self, rpm, speed):
#     if speed == "" or speed == 0:
#         return 0
#     if rpm == "" or rpm == 0:
#         return 0
#     rps = rpm/60
#     mps = (speed*0.44704) #meters per second
#     final_drive  = 4.765
#     tire_circumference = 1.964 #meters
#     current_gear_ratio = (rps / (mps / tire_circumference)) / final_drive
#     print(current_gear_ratio)
#     gear = min((abs(current_gear_ratio - i), i) for i in self.gear_ratios)[1] 
#     return gear

            
