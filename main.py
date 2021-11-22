from kivymd.app import MDApp
from kivymd.tools.hotreload.app import MDApp
from kivy.lang import Builder
from kivy.properties import StringProperty, BooleanProperty
from kivymd.uix.floatlayout import MDFloatLayout
from obd2 import OBD
import time

class Connection(MDFloatLayout):
    btn = StringProperty('Conectar')
    status = StringProperty('desconectado')
    connected = BooleanProperty(False)
    obd = OBD()
    def connect(self):
        print('conectando')
        self.btn = 'Conectando...'
        self.status = 'conectando...'
        time.sleep(2)
        self.obd.connect()
        self.show_conn()
    def disconnect(self):
        self.obd.disconnect()
        self.show_conn()
    def show_conn(self):
        if self.obd.check_conn():
            self.status = 'conectado'
            self.btn = 'Desconectar'
        else:
            self.status = 'desconectado'
            self.btn = 'Conectar'

class OBD2(MDApp):
    KV_FILES = ['obd2.kv']
    DEBUG = True
    def build_app(self):
        return Builder.load_file('obd2.kv')

OBD2().run()
