document.addEventListener("DOMContentLoaded", (event) => {
    get_status()
})
const $ = ( id ) => document.getElementById( id )
const spinner_w = '<div class="d-flex justify-content-center"><div class="text-center spinner-border spinner-border-sm text-ligth" role="status"></div></div>'

const section_list = document.querySelectorAll("section")
function show_section(name) {
    section_list.forEach(s => {
        if (s.id == name) {
          s.style.display = 'block'
        } else { s.style.display = 'none' }
      })
}

const status_text = $('status')
function get_status() {
    connect_btn.innerHTML = spinner_w
    fetch('/get_status').then(data => data.json())
    .then(data => {
        if (data['result']) {
            connect_btn.innerHTML = 'Disconnect'
            connect_btn.setAttribute('onclick', 'disconnect()')
            status_text.innerHTML = `Connected in ${data['result']}`
            speed()
        } else {
            connect_btn.innerHTML = 'Connect'
            connect_btn.setAttribute('onclick', 'connect()')
            status_text.innerHTML = 'Disconnected'
            show_section('connection')
            // data['error'] ? alert(data['error']) : console.error('Unknown data:', data)
        }
    }).catch(error => { alert(`Error getting status | ${error}`) })
}

const connection_div = $('connection')
const connect_btn = $('connect')
function connect() {
    conntype = document.querySelector('input[name="conntype"]:checked').value
    connect_btn.innerHTML = spinner_w
    fetch(`/connect?type=${conntype}`).then(data => data.json())
    .then(data => {
        if (data['result']) {
            connect_btn.innerHTML = 'Disconnect'
            connect_btn.setAttribute('onclick', 'disconnect()')
            status_text.innerHTML = `Connected in ${data['result']}`
            speed()
        } else {
            connect_btn.innerHTML = 'Connect'
            connect_btn.setAttribute('onclick', 'connect()')
            status_text.innerHTML = 'Disconnected'
            data['error'] ? alert(data['error']) : console.error('Unknown data:', data)
        }
    }).catch(error => {
        alert(`Error connecting | ${error}`) 
        get_status()
    })
}
function disconnect() {
    fetch('/disconnect').then(data => data.json())
    .then(data => {
        if (data['result']) {
            connect_btn.innerHTML = 'Connect'
            connect_btn.setAttribute('onclick', 'connect()')
            status_text.innerHTML = 'Disconnected'
        } else {
            data['error'] ? alert(data['error']) : console.error('Unknown data:', data)
        }
    }).catch(error => {
        alert(`Error disconnecting | ${error}`) 
        get_status()
    })
}

const speed_div = $('speed')
const speed_data = speed_div.querySelector('#speed_data')
let speed_on = false
async function speed() {
    show_section('speed')
    speed_on = true
    while (speed_on) {
      await new Promise(resolve => setTimeout(resolve, 5000))
      fetch('/speed').then(data => data.json())
      .then(data => {
        if (data['result']) {
            speed_data.innerHTML = Object.keys(data['result']).map(k => `
                <p><strong>${k}</strong>: ${data['result'][k]}<p><br>`)
        } else if (data['error']) {
            switch (data['error']) {
                case 'disconnected':
                    speed_on = false
                    connect_btn.innerHTML = 'Connect'
                    connect_btn.setAttribute('onclick', 'connect()')
                    status_text.innerHTML = 'Disconnected'
                    show_section('connection')
                default:
                    console.log(data['error'])
            }
        } else {console.error('Unknown data:', data)}
        }).catch(error => {
            speed_on = false
            alert(`Error getting speed dash | ${error}`) 
        })
    }
}
