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
    btn = StringProperty('Conectar')
    status = StringProperty('desconectado')
    conn = False
    def press(self, btn):
        btn.disabled = True
        btn.text = 'Conectando'
        self.status = 'conectando...'
        Clock.schedule_once(partial(self.connect, 'wireless', btn), 0.359)
    def connect(self, conntype, btn, *args):
        if conntype == 'wireless':
            self.conn = obd.OBD('socket://192.168.0.10:35000')
        elif conntype == 'bluetooth':
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

class OBDII(MDApp):
    # KV_FILES = ['obd2.kv']
    DEBUG = True
    def build_app(self):
        return Builder.load_file('obd2.kv')

OBDII().run()
