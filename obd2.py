#!/usr/bin/env python3
import obd
from datetime import datetime
import time

class OBD():
    conn = False
    def connect(self, type='wireless'):
        if type == 'wireless':
            self.conn = obd.OBD('socket://192.168.0.10:35000')
        else:
            print('not implemented')
            self.conn = False
    def check_conn(self):
        if self.conn:
            if self.conn.is_connected():
                return True
            else:
                self.conn.close()
                return False
        else:
            print('false conn')
            return False
    def disconnect(self):
        self.conn.close()

    def loop_cmds(self, commands):
        data = {}
        for c in commands:
            try:
                data[c] = self.conn.query(obd.commands[c]).value.magnitude
            except Exception as ex:
                data[c] = False
                print(f"Error in {c} command: {ex}")
        return data

    def speed_view(self):
        commands = ['RPM', 'SPEED', 'COOLANT_TEMP']
        while True:
            data = self.loop_cmds(commands)
            if data['RPM'] == 0 or data['SPEED'] == 0:
                data['RATIO'] = 0
            else:
                rps = data['RPM']/60
                mps = data['SPEED']*0.277777
                final_drive  = 2.87
                tire_circumference = 2.085
                data['RATIO'] = (rps / (mps / tire_circumference)) / final_drive
                # gear = min((abs(current_gear_ratio - i), i) for i in gear_ratios)[1] 
            for d in data.keys():
                print(f'{d}: {data[d]}')

    def o2_view(self):
        commands = [
            'O2_S1_WR_CURRENT',
            'O2_S5_WR_CURRENT',
            'SHORT_FUEL_TRIM_1',
            'SHORT_FUEL_TRIM_2',
            'INTAKE_TEMP'
        ]
        while True:
            data = self.loop_cmds(commands)
            if data['O2_S1_WR_CURRENT'] < -0.01:
                data['color'] = 'blue'
            elif data['O2_S1_WR_CURRENT'] <= 0.01:
                data['color'] = 'green'
            else:
                data['color'] = 'red'
            for d in data.keys():
                print(f'{d}: {data[d]}')

    def log_set(self, car, commands):
        file = open(f"logs/{car}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log", 'w')
        file.write(','.join(commands)+'\n') # Head of csv
        while True:
            print(datetime.now().time())
            data = self.loop_cmds(commands)
            log = []
            for d in data.values():
                log.append(d)
            file.write(','.join(log)+'\n')


            # try:
            #     speed_view()
            #     # log_set(o2Set)
            # except KeyboardInterrupt:
            #     print("Stopped")
            # file.close()
