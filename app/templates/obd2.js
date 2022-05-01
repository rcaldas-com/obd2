document.addEventListener("DOMContentLoaded", (event) => {
    get_status()
})
const $ = ( id ) => document.getElementById( id )
const status_div = $('status')
const connection_div = $('connection')
const connect_btn = $('connect')
const spinner_w = '<div class="d-flex justify-content-center"><div class="text-center spinner-border spinner-border-sm text-ligth" role="status"></div></div>'

function get_status() {
    connect_btn.innerHTML = spinner_w
    fetch('/get_status').then(data => data.json())
    .then(data => {
        if (data['result']) {
            connect_btn.innerHTML = 'Disconnect'
            connect_btn.setAttribute('onclick', 'disconnect()')
            status_div.innerHTML = `Connected in ${data['result']}`
        } else {
            connect_btn.innerHTML = 'Connect'
            connect_btn.setAttribute('onclick', 'connect()')
            status_div.innerHTML = 'Disconnected'
            // data['error'] ? alert(data['error']) : console.error('Unknown data:', data)
        }
    }).catch(error => { alert(`Error getting status | ${error}`) })
}

function connect() {
    conntype = document.querySelector('input[name="conntype"]:checked').value
    connect_btn.innerHTML = spinner_w
    fetch(`/connect?type=${conntype}`).then(data => data.json())
    .then(data => {
        if (data['result']) {
            connect_btn.innerHTML = 'Disconnect'
            connect_btn.setAttribute('onclick', 'disconnect()')
            status_div.innerHTML = `Connected in ${data['result']}`
        } else {
            connect_btn.innerHTML = 'Connect'
            connect_btn.setAttribute('onclick', 'connect()')
            status_div.innerHTML = 'Disconnected'
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
            status_div.innerHTML = 'Disconnected'
        } else {
            data['error'] ? alert(data['error']) : console.error('Unknown data:', data)
        }
    }).catch(error => {
        alert(`Error disconnecting | ${error}`) 
        get_status()
    })
}

