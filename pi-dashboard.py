import meraki
import requests
import time
import os
from flask import Flask, request, render_template, jsonify, redirect, url_for, session
import psutil
import subprocess

from dotenv import load_dotenv

# Load all environment variables from .env file
load_dotenv()

# Initialize Meraki SDK
dashboard = meraki.DashboardAPI(os.environ['MERAKI_API_TOKEN'], output_log=False, print_console=False)
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key")

device_stats = {}

# Helpers for IoT device control
def triggerRoutineOff():
    url = "https://api-v2.voicemonkey.io/trigger"
    params = {'token': os.environ['ROUTINE_API_TOKEN'], 'device': 'device-off'}
    print("EVERYTHING OFF")
    requests.get(url, params=params)
    return

def triggerRoutineWhite():
    url = "https://api-v2.voicemonkey.io/trigger"
    params = {'token': os.environ['ROUTINE_API_TOKEN'], 'device': 'device-on'}
    print("LIGHTS ON")
    requests.get(url, params=params)
    return

def triggerRoutineMovie():
    url = "https://api-v2.voicemonkey.io/trigger"
    params = {'token': os.environ['ROUTINE_API_TOKEN'], 'device': 'movie-mode'}
    print("MOVIE ON")
    requests.get(url, params=params)
    return

def triggerRoutineLightsOff():
    url = "https://api-v2.voicemonkey.io/trigger"
    params = {'token': os.environ['ROUTINE_API_TOKEN'], 'device': 'lights-off'}
    print("LIGHTS OFF")
    requests.get(url, params=params)
    return

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/receive-stats', methods=['POST'])
def receive_stats():
    global device_stats
    data = request.json
    print("Received stats")
    if data and 'device' in data and 'stats' in data:
        print(data)
        device_stats[data['device']] = data['stats']
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

@app.route('/stats')
def stats():
    local_stats = {
        'CPU Usage': f"{psutil.cpu_percent()}%",
        'Compute Temp': get_compute_temp(),
        'RAM': get_ram_info(),
        'Storage': get_storage_info(),
    }
    all_stats = {'Device 1': local_stats}
    all_stats.update(device_stats)
    return jsonify(all_stats)

def get_compute_temp():
    try:
        compute_temp_output = subprocess.check_output(['vcgencmd', 'measure_temp'], encoding='utf-8')
        compute_temp_c = float(compute_temp_output.split('=')[1].split("'")[0])
        compute_temp_f = compute_temp_c * 9 / 5 + 32
        return f"{compute_temp_c:.2f}°C / {compute_temp_f:.2f}°F"
    except Exception:
        return "N/A"

def get_ram_info():
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024 ** 3)
    ram_total = ram.total / (1024 ** 3)
    return f"{ram_used:.2f} GB / {ram_total:.2f} GB"

def get_storage_info():
    storage = psutil.disk_usage('/')
    storage_used = storage.used / (1024 ** 3)
    storage_total = storage.total / (1024 ** 3)
    return f"{storage_used:.2f} GB / {storage_total:.2f} GB"

@app.route('/login', methods=['GET', 'POST'])
def login():
    passwords = os.environ.get('BUTTON_PAGE_PASSWORDS', '').split(',')
    passwords = [p.strip() for p in passwords]
    
    if request.method == 'POST':
        if request.form['password'] in passwords:
            session['authenticated'] = True
            return redirect(url_for('button_control'))
        return render_template('login.html', error="Invalid password")
    return render_template('login.html', error=None)

@app.route('/button-control', methods=['GET', 'POST'])
def button_control():
    if 'authenticated' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'off_button' in request.form:
            triggerRoutineOff()
        elif 'white_button' in request.form:
            triggerRoutineWhite()
        elif 'onlylightsoff_button' in request.form:
            triggerRoutineLightsOff()
    return render_template('button_control.html')

@app.route("/meraki_webhook", methods=["POST"])
def handle_meraki_event():
    if request.method == 'POST':
        print(request.json)
        print("Received POST event")

        button_name = request.json['deviceName']
        button_press_type = ''
        try:
            button_press_type = request.json['alertData']['trigger']['button']['pressType']
        except KeyError:
            pass
        
        if button_press_type == 'short':
            print("Button Name: " + button_name + "\n")
            print('SHORT BUTTON PRESS')
            triggerRoutineOff()
            return 'Meraki Button - Short Press Received'
        elif button_press_type == 'long':
            print("Button Name: " + button_name + "\n")
            print('LONG BUTTON PRESS')
            triggerRoutineWhite()
            return 'Meraki Button - Long Press Received'
        else:
            print('TEST WEBHOOK')
            return 'Meraki Button - Webhook Test'
    return

if __name__ == "__main__":
    # Use environment variables for host and port
    app.run(host=os.environ.get('HOST', '0.0.0.0'), 
            port=int(os.environ.get('PORT', 5050)))
