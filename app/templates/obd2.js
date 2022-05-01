document.addEventListener("DOMContentLoaded", (event) => {
    check_connection()
})
const $ = ( id ) => document.getElementById( id )
const status_div = $('status')
const connection_div = $('connection')
const spinner_w = '<div class="d-flex justify-content-center"><div class="text-center spinner-border spinner-border-sm text-ligth" role="status"></div></div>'

function check_connection() {
    fetch('/api').then(data => data.json()).then(data => {
        console.log(data)

        status_div.innerHTML = data['result']

    })
}

