import pid
import numpy as np
import cv2
from apriltag import Detector
from pymavlink import mavutil
import depth_control as dc
import matplotlib.pyplot as plt

cameraMatrix = np.array([ 1060.71, 0, 960, 0, 1060.71, 540, 0, 0, 1]).reshape((3,3))
camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )

cap = cv2.VideoCapture('AprilTagTest.mkv')
ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        mav = mavutil.mavlink_connection("udpin:0.0.0.0:14550")

        for tag in tags:
            center_x = sum(coord[0] for coord in tag.corners) / 4
            center_y = sum(coord[1] for coord in tag.corners) / 4
            for idx in range(len(tag.corners)):
                cv2.line(color_img, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 3)

            cv2.putText(color_img, str(tag.tag_id),
                        org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=5,
                        color=(0, 0, 255),
                        thickness =  10)
            
            (width, length) = np.shape(color_img)

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
