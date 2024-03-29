import obd
import time
from datetime import datetime

class Connection():
    conn = False
    suported = []
    def get_status(self):
        if self.conn and self.conn.is_connected():
            return {'result': self.conn.port_name()}
        else:
            return {'error': 'Not connected'}
    def connect(self, conntype=None):
        if conntype:
            if conntype == 'wi':
                self.conn = obd.OBD('socket://192.168.0.10:35000')
            elif conntype == 'bt':
                ports = obd.scan_serial()
                if len(ports) > 0:
                    self.conn = obd.OBD(ports[0])
                else:
                    return {'error': 'No bluetooth connection found'}
            else:
                return {'error': 'Unknown type of connection'}
        else:
            self.conn = obd.OBD()
        time.sleep(1)
        if self.conn.is_connected():
            self.suported = [x.name for x in self.conn.supported_commands]
            
            # for i in self.suported:
            #     print(i)

            return {'result': self.conn.port_name()}
        else:
            return {'error': 'Could not connect'}

    def disconnect(self):
        if self.conn: # and self.conn.is_connected():
            self.conn.close()
            # self.suported = []
    def get_cmds(self, commands):
        if not self.conn or not self.conn.is_connected():
            return {'error': 'disconnected'}
        data = {}
        for c in commands:
            if c in self.suported:
                try:
                    data[c] = self.conn.query(obd.commands[c]).value.magnitude if self.conn.query(obd.commands[c]).value else None
                except Exception as e:
                    data[c] = None
                    print(f"\nError in {c} command: {e}\n")
            else:
                print(f'\nCommand {c} not in supported_commands\n')
        return {'result': data}
    # def log_set(self, car, commands):
    #     file = open(f"logs/{car}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log", 'w')
    #     file.write(','.join(commands)+'\n') # Head of csv
    #     while True:
    #         print(datetime.now().time())
    #         data = self.loop_cmds(commands)
    #         log = []
    #         for d in data.values():
    #             log.append(d)
    #         file.write(','.join(log)+'\n')

