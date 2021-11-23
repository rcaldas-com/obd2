from kivymd.app import MDApp
from kivymd.tools.hotreload.app import MDApp
from kivy.lang import Builder
from kivy.clock import Clock
from functools import partial
from kivy.properties import StringProperty, BooleanProperty
from kivymd.uix.floatlayout import MDFloatLayout
import obd
import time
from datetime import datetime

class Connection(MDFloatLayout):
    conn = False
    conntype = 'wireless'
    btn = StringProperty('Conectar')
    status = StringProperty('desconectado')
    def press(self, btn):
        btn.disabled = True
        btn.text = 'Conectando'
        self.status = 'conectando...'
        Clock.schedule_once(partial(self.connect, btn), 0.359)
    def connect(self, btn, *args):
        if self.conntype == 'wireless':
            self.conn = obd.OBD('socket://192.168.0.10:35000')
        elif self.conntype == 'bluetooth':
            print('not implemented')
        if self.conn:
            if self.conn.is_connected():
                btn.txt = 'Desconectar'
                btn.bind(on_release=self.disconnect)
                btn.disabled = False
                self.status = 'conectado'
                return True
            else:
                self.conn.close()
        btn.text = 'Conectar'
        btn.disabled = False
        self.status = 'desconectado'
        return False
    def disconnect(self, *args):
        if self.conn and self.conn.is_connected():
            self.conn.close()
        self.status = 'desconectado'
        self.btn = 'Conectar'
    def set_conntype(self, active, value):
        if active:
            self.conntype = value

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

class OBDII(MDApp):
    # KV_FILES = ['obd2.kv']
    DEBUG = True
    def build_app(self):
        return Builder.load_file('obd2.kv')

OBDII().run()
