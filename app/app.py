#!/usr/bin/env python3
from datetime import datetime
import time

import obd
from flask import Flask, render_template, request

from connection import Connection

app = Flask(__name__)
conn = Connection()

@app.get('/')
def index():
    return render_template('obd2.html')

@app.get('/get_status')
def get_status():
    return conn.get_status()

@app.get('/connect')
def connect():
    return conn.connect(request.args.get('type'))

@app.get('/disconnect')
def disconnect():
    return conn.disconnect()


@app.get('/speed')
def get_speeds():
    commands = ['RPM', 'SPEED', 'COOLANT_TEMP']
    data = conn.get_cmds(commands)
    if data.get('result'):
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
        return {'result': data}
    else:
        return data

@app.get('/o2')
def get_o2():
    commands = [
        'O2_S1_WR_CURRENT',
        'O2_S5_WR_CURRENT',
        'SHORT_FUEL_TRIM_1',
        'SHORT_FUEL_TRIM_2',
        'INTAKE_TEMP'
    ]
    data = conn.get_cmds(commands)
    if data['O2_S1_WR_CURRENT'] < -0.01:
        data['color'] = 'blue'
    elif data['O2_S1_WR_CURRENT'] <= 0.01:
        data['color'] = 'green'
    else:
        data['color'] = 'red'
    for d in data.keys():
        print(f'{d}: {data[d]}')
    return {'result': data}



### LOG
# carname

# def log_set(commands):
#     file = open(f"logs/{car}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log", 'w')
#     file.write(','.join(commands)+'\n') # Head of csv
#     while True:
#         print(datetime.now().time())
#         data = loop_cmds(commands)
#         log = []
#         for d in data.values():
#             log.append(d)
#         file.write(','.join(log)+'\n')
