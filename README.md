Log and Analyse Wifi or Bluetooth OBD2 Data with Python and Javascript.
---

***A dashboard for monitoring vehicle information to identify problems, calculate consumption, view/clear fault codes, etc***

---

### How to use  

You need a obd2 adapter like ELM-327, I used a Wifi one.  

You need Docker or install the python requirements in your env or venv  

With Docker:  
- Install docker if not already installed: [Docker Install](https://docs.docker.com/engine/install/)  
- Get docker-compose binary for your system: [Docker Compose Install](https://docs.docker.com/compose/install/)  
- Run: `docker-compose up`  

Local run:  
- `cd app`  
- `FLASK_ENV=development flask run`  


Open [`http://localhost:5000`](http://127.0.0.1:5000) in your browser    
-----


### How it works  
- Using flask as frontend and api
- Using python-obd to comunicate with adapter
- Connects a wifi device in default '192.168.0.10:35000' address  
- If bluetooth, configure properly, see [Python-OBD Docs](https://python-obd.readthedocs.io/en/latest/Connections/)  


### How to code  

- HTML, CSS and JavaScript files are in `app/templates`  

- Static files in `app/static`  

- Flask routes in `app/app.py`  

- OBD2 Connection in `app/connection.py`  

- You can run in background to keep terminal free with `docker-compose up -d`  
See logs with `docker-compose logs`. And to keep showing: `docker-compose logs -f`  
  
- Stop and Remove all Containers about: `docker-compose down`  


---  

### What's next  

- Create dashboard for different views
- Calculate differential ratio from velocity to estimate actual gear
- Change docker to bind bluetooth device
- Use Web Sockets / ASync
- Record log in time-series format
- ...
