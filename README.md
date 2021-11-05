Log and Analyse a vehicle reading OBD2 data from wifi or bluetooth.
---

***It creates a csv .log file with the starting date and result of each obd2 command present in 'commands' list.***

---

### How to use

You need a obd2 scanner like ELM-327, I used a Wifi one.

- Set car name and list of obd2 commands to log
- Verify connection
  - It's set to connect a wifi device in default '192.168.0.10:35000' address
  - If bluetooth, configure properly, see [Python-OBD Docs](https://python-obd.readthedocs.io/en/latest/Connections/)
- To start logging run ```python3 obd2.py```
- To stop logging press ```CTRL+C```
- A new file with start date will be in 'logs' directory

---

### What's next

- Become a portable app with Kivy
- Calculate differential ratio from velocity to estimate actual gear
- Record in time-series format
- Log only O2 sensors to analyse with more precision
- ...