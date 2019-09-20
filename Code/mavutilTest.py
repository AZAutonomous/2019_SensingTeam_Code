from pymavlink import mavutil
import serial
import threading
import os
import sys

def getPos():
    global latitude
    global longitude
    global altitude
    global heading
    threading.Timer(0.2, getPos).start()

    try:
        posMsg = pixhawk.recv_match(type="GLOBAL_POSITION_INT", blocking=False, timeout=10.0)
        latitude = posMsg.lat
        longitude = posMsg.lon
        altitude = posMsg.alt
        heading = posMsg.hdg
    except:
        pass

def getAttitude():
    global pitch
    global roll
    global yaw
    threading.Timer(0.2, getAttitude).start()

    try:
        attMsg = pixhawk.recv_match(type="ATTITUDE", blocking=False, timeout=10.0)
        pitch = attMsg.pitch
        roll = attMsg.roll
        yaw = attMsg.yaw
    except:
        pass

def printTelem():
    threading.Timer(1, printTelem).start()
    os.system('clear')
    print("Latitude: ", latitude)
    print("Longitude: ", longitude)
    print("Altitude: ", altitude)
    print("Heading: ", heading)
    print("Roll: ", roll)
    print("Pitch: ", pitch)
    print("Yaw: ", yaw)


latitude = 0.0
longitude = 0.0
altitude = 0.0
heading = 0.0
roll = 0.0
pitch = 0.0
yaw = 0.0

if (len(sys.argv) != 2):
    print("ERROR: Usage: mavutilTest.py <serial_port>")
    exit()

mavutil.set_dialect("ardupilotmega")
SERIAL_PORT = str(sys.argv[1]).strip()

try:
    pixhawk = mavutil.mavlink_connection(SERIAL_PORT, autoreconnect=True)
except:
    print("ERROR: unable to connect to \"", SERIAL_PORT, "\"")
    exit()

getPos()
getAttitude()
printTelem()