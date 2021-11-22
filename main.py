from kivymd.app import MDApp
from kivymd.tools.hotreload.app import MDApp
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivymd.uix.floatlayout import MDFloatLayout

class Connection(MDFloatLayout):
    status = StringProperty('disconnected')


class OBD2(MDApp):
    KV_FILES = ['obd2.kv']
    DEBUG = True
    def build_app(self):
        return Builder.load_file('obd2.kv')

OBD2().run()
