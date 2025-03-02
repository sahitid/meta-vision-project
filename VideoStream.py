
# Import the necessary packages
from threading import Thread
import cv2


class VideoStream:
    """Camera object"""
    def __init__(self, resolution=(640,480),framerate=10,PiOrUSB=1,src=0):

        # Create a variable to indicate if it's a USB camera or PiCamera.
        # PiOrUSB = 1 will use PiCamera. PiOrUSB = 2 will use USB camera.
        self.PiOrUSB = PiOrUSB
        if self.PiOrUSB == 2: # USB camera
            # Initialize the USB camera and the camera image stream
            self.stream = cv2.VideoCapture(src)
            ret = self.stream.set(3,resolution[0])
            ret = self.stream.set(4,resolution[1])
            #ret = self.stream.set(5,framerate) #Doesn't seem to do anything so it's commented out

            # Read first frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

	# Create a variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread to read frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        if self.PiOrUSB == 2: # USB camera

            # Keep looping indefinitely until the thread is stopped
            while True:
                # If the camera is stopped, stop the thread
                if self.stopped:
                    # Close camera resources
                    self.stream.release()
                    return

                # Otherwise, grab the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
		# Return the most recent frame
        return self.frame

    def stop(self):
		# Indicate that the camera and thread should be stopped
        self.stopped = True
