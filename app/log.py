#!/usr/bin/env python3
import obd
from datetime import datetime
import time

# TinyDB !

car = 'Fit'
commands = [
    'RPM',
    'COOLANT_TEMP',
    # 'MAF',
    # 'THROTTLE_POS',
    'INTAKE_TEMP',
    'TIMING_ADVANCE',
    'ENGINE_LOAD',
    'ELM_VOLTAGE',
    # 'SPEED',
    # 'O2_S1_WR_CURRENT',
    # 'O2_S5_WR_CURRENT',
    # 'O2_B1S2',
    # 'O2_B2S2',
    'SHORT_FUEL_TRIM_1',
    # 'SHORT_FUEL_TRIM_2',
    'LONG_FUEL_TRIM_1',
    # 'LONG_FUEL_TRIM_2'
]

conn = obd.OBD('socket://192.168.0.10:35000')

for i in conn.supported_commands:
    print(i)
    print('\n')


# conn = obd.Async('/dev/ttyCAN0')
# obd.logger.setLevel(obd.logging.DEBUG)
# obd.logger.removeHandler(obd.console_handler)

if conn.is_connected():
    file = open(f"../logs/{car}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log", 'w')
    print('\nCollecting...')
    try:
        head = [c for c in commands]
        head.append('timestamp')
        file.write(','.join(head)+'\n')
        # for c in commands:
        #     conn.watch(obd.commands[c])
        # conn.start()
        time.sleep(1)
        while True:
            log = []
            for c in commands:
                try:
                    log.append(str(conn.query(obd.commands[c]).value.magnitude))
                except Exception as ex:
                    print(f"Error in {c} command: {ex}")
                    log.append('')
            log.append(str(datetime.now().timestamp()))
            print(log)
            file.write(','.join(log)+'\n')
            time.sleep(0.359) # ~ 0.997 / 2
    except KeyboardInterrupt:
        print("Stopped")
    file.close()
else:
    print('Not connected')
conn.close()
# conn.stop()




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
            
            
