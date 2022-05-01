Log and Analyse OBD2 data from wifi or bluetooth with Python Kivy.
---

***Changing project to use Flask.***

---

### How to use

You need a obd2 scanner like ELM-327, I used a Wifi one.

You need Docker or install requirements in your PC or venv

With Docker run:
`docker-compose up`
With local install:
`cd app`
`flask run`

And open `http://127.0.0.1:5000` in your browser


- It's set to connect a wifi device in default '192.168.0.10:35000' address
- If bluetooth, configure properly, see [Python-OBD Docs](https://python-obd.readthedocs.io/en/latest/Connections/)


---

### What's next

- Change docker to bind bluetooth device
- Calculate differential ratio from velocity to estimate actual gear
- Log only O2 sensors to analyse
- Record in time-series format
- ...