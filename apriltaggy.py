import pid
import numpy as np
import cv2
import sys
import signal
from pymavlink import mavutil
from dt_apriltags import Detector
#from pymavlink import mavutil
import depth_control as dc
import matplotlib.pyplot as plt

cameraMatrix = np.array([ 1060.71, 0, 960, 0, 1060.71, 540, 0, 0, 1]).reshape((3,3))
camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )

cap = cv2.VideoCapture('AprilTagTest.mkv')
ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

mav = mavutil.mavlink_connection("udpin:0.0.0.0:14550")

# catch CTRL+C
def signal_handler(sig, frame):
    print("CTRL+C pressed. Disarming")
    mav.arducopter_disarm()
    mav.motors_disarmed_wait()
    print("Disarmed")
    sys.exit(0)

# catch CTRL+C
signal.signal(signal.SIGINT, signal_handler)

# wait for the heartbeat message to find the system ID
mav.wait_heartbeat()

# arm the vehicle
print("Arming")
mav.arducopter_arm()
mav.motors_armed_wait()
print("Armed")

# set mode to MANUAL
print("Setting mode to MANUAL")
mav.mav.set_mode_send(
    mav.target_system,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    19,  # Manual mode
)
print("Mode set to MANUAL")

count=0
frequency=200
while ret:
    if count%frequency==count:
        at_detector = Detector(families='tag36h11',
                            nthreads=1,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)
        tags = at_detector.detect(gray, True, camera_params, tag_size  = 0.1)
        color_img = frame
        
        # Instead of doing for tag in tags, I just used the first tag spotted
        center_x = sum(coord[0] for coord in tags[0].corners) / 4
        center_y = sum(coord[1] for coord in tags[0].corners) / 4
        for idx in range(len(tags[0].corners)):
            cv2.line(color_img, tuple(tags[0].corners[idx - 1, :].astype(int)), tuple(tags[0].corners[idx, :].astype(int)), (0, 255, 0), 3)

        cv2.putText(color_img, str(tags[0].tag_id),
                    org=(tags[0].corners[0, 0].astype(int) + 10, tags[0].corners[0, 1].astype(int) + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=5,
                    color=(0, 0, 255),
                    thickness =  10)
        
        width, length = np.shape(color_img)[0:2]

        #depth
        depth_pid = pid.PID(32, 0.5, -1, 20)

        #lateral movement
        lateral_pid = pid.PID(32, 0.5, -1, 20)
        
        while True:

            desired_depth = center_y
            current_depth = width/2

            print("Depth: ", current_depth)

            # calculate error
            error = desired_depth - current_depth
            print("Error: ", error)

            output = depth_pid.update(error)
            print("Output: ", output)

            # set vertical power
            dc.set_vertical_power(mav, -output) 

            #-------------------------------------------------------------

            desired_lat  = center_x
            fixed_lat  = length/2

            # calculate error
            error = fixed_lat - desired_lat

            print("Error: ", error)


            output = lateral_pid.update(error)
            print("Output: ", output)

            # set lateral power
            dc.set_lateral_power(mav, -output)

            
     
        plt.imshow(color_img)
        count+=1

        






    else:
        print(count)
        ret, frame = cap.read()
