# from kivymd.app import MDApp
from kivymd.tools.hotreload.app import MDApp
from kivy.clock import Clock
from functools import partial
from kivy.properties import StringProperty, BooleanProperty

import obd
import time
from datetime import datetime

from kivy.uix.floatlayout import FloatLayout

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

Builder.load_file('obd2.kv')

class Connection(Screen):
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
            ports = obd.scan_serial()
            if len(ports) > 0:
                self.conn = obd.OBD(ports[0])
            else:
                print(f'Nenhuma porta encontrada')
        if self.conn:
            if self.conn.is_connected():
                print(f'Comandos suportados:\n{self.conn.supported_commands}')
                btn.txt = 'Desconectar'
                btn.md_bg_color = [1,0,0,1]
                btn.bind(on_release=self.disconnect)
                btn.disabled = False
                self.status = 'conectado'
                self.ids.status.color = [0,1,0,0.7]
                self.manager.current = 'speeds'
                return True
            else:
                self.conn.close()
        btn.text = 'Conectar'
        btn.disabled = False
        self.status = 'desconectado'
        self.ids.status.color = [1,0,0,0.7]
        return False
    def disconnect(self, *args):
        if self.conn and self.conn.is_connected():
            self.conn.close()
        self.status = 'desconectado'
        self.ids.status.color = [1,0,0,0.7]
        self.btn = 'Conectar'
        btn.md_bg_color = [0,0,1,1]
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

class Speeds(Screen):
    pass
    # Clock.schedule_interval(lambda dt: self.read_data(), 0.10)

class OBDII(App):
    # KV_FILES = ['obd2.kv']
    DEBUG = True
    def change_color(self):
        theme = self.theme_cls.theme_style
        if theme == 'Dark':
            self.theme_cls.theme_style = 'Light'
        else:
            self.theme_cls.theme_style = 'Dark'
    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(Connection(name='connection'))
        self.sm.add_widget(Speeds(name='speeds'))
        return self.sm

        # for i in ['connection', 'info']:
        #     screen = Screen(name=i.capitalize())

        # self.theme_cls.primary_palette = 'DeepPurple'
        # self.theme_cls.primary_hue = '700'
        # return Builder.load_file('obd2.kv')

if __name__ == "__main__":
    OBDII().run()
