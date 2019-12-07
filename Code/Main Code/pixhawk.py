import threading

# before the sensors are integrated,
# we have globals to simulate each kinematic datum
pixhawk = None # this is the pointer to the sensor data will go

pitch = 0.0
roll = 0.0
yaw = 0.0
latitude = 0.0
longitude = 0.0
altitude = 0.0
heading = 0.0


def getPos():
    """
    This function populates the globals for position data
    """
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
    """
    This function populates the 3-D attitude globals
    """
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