#!/usr/bin/env python3
import csv
import time
from datetime import datetime

import obd


CAR = "Omega2005"
PORT = "/dev/ttyUSB0"
LOG_DIR = "logs"
SAMPLE_HZ = 10.0
FAST_MODE = True
PER_PID_TIMESTAMP = True

# Comandos principais para mapear ponto original
COMMANDS = [
    "RPM",
    "TIMING_ADVANCE",
    "MAF",
    "ENGINE_LOAD",
    "THROTTLE_POS",
    "RELATIVE_THROTTLE_POS",
    "INTAKE_PRESSURE",
    "INTAKE_TEMP",
    "COOLANT_TEMP",
    "SPEED",
    "BAROMETRIC_PRESSURE",
]

FAST_COMMANDS = [
    "RPM",
    "TIMING_ADVANCE",
    "MAF",
    "ENGINE_LOAD",
    "THROTTLE_POS",
]


def to_float(result, temp_to_c=False):
    if result is None or result.value is None:
        return None

    value = result.value

    if temp_to_c and hasattr(value, "to"):
        try:
            return float(value.to("celsius").magnitude)
        except Exception:
            pass

    if hasattr(value, "magnitude"):
        return float(value.magnitude)

    try:
        return float(value)
    except Exception:
        return None


def maf_g_per_rev(maf_g_s, rpm):
    if maf_g_s is None or rpm is None or rpm <= 0:
        return None
    # 4 tempos: 2 voltas por ciclo
    return (maf_g_s * 60.0) / (rpm * 2.0)


def main():
    log_file_path = f"{LOG_DIR}/{CAR}_ignition_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"

    conn = obd.OBD(PORT)
    if not conn.is_connected():
        print("Not connected to OBD2")
        return

    print("Connected to OBD2!")
    print(f"Protocol: {conn.protocol_name()}")

    supported = set(conn.supported_commands)
    base_cmds = FAST_COMMANDS if FAST_MODE else COMMANDS
    active_cmds = []
    for cmd in base_cmds:
        if obd.commands[cmd] in supported:
            active_cmds.append(cmd)
        else:
            print(f"⚠ {cmd} not supported - skipping")

    header = []
    for cmd in active_cmds:
        header.append(cmd)
        if PER_PID_TIMESTAMP:
            header.append(f"{cmd}_ts")
    header.extend([
        "MAF_G_PER_REV",
        "LOAD_EQUIV",
        "timestamp",
    ])

    with open(log_file_path, mode="w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(header)

        print("Logging started... Press Ctrl+C to stop.")
        sample_period = 1.0 / SAMPLE_HZ

        while True:
            row = []
            values = {}

            for cmd in active_cmds:
                try:
                    result = conn.query(obd.commands[cmd])
                    if cmd in ["COOLANT_TEMP", "INTAKE_TEMP"]:
                        values[cmd] = to_float(result, temp_to_c=True)
                    else:
                        values[cmd] = to_float(result)
                    ts = time.time()
                except Exception:
                    values[cmd] = None
                    ts = None

                row.append(values[cmd])
                if PER_PID_TIMESTAMP:
                    row.append(ts)

            rpm = values.get("RPM")
            maf = values.get("MAF")
            load = values.get("ENGINE_LOAD")

            maf_per_rev = maf_g_per_rev(maf, rpm)
            load_equiv = load if load is not None else None

            row.extend([
                maf_per_rev,
                load_equiv,
                datetime.now().timestamp(),
            ])

            writer.writerow(row)
            log_file.flush()
            time.sleep(sample_period)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped")