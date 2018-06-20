import PIL
import cv2, sys, os
import numpy as np

format = 'XVID' #This format should work for Windows
colour = False #no colours. (In fact colours do not work at the moment.)
FPS = -1 # if FPS < 0, get FPS from file
maxFrames = -1

#get args
if(len(sys.argv)>=3):
   inputPath = sys.argv[1]
   outputPath = sys.argv[2]
   if(len(sys.argv)>=4):
      FPS = int(sys.argv[3])
   if(len(sys.argv)>=5):
      maxFrames = int(sys.argv[4])
else:
   print('usage: AVIToLog [input.avi] [output.avi] [FPS] [Max frames]')
   print('Use -1 for FPS and Max frames to read from input file')



inputPath = sys.argv[1]
outputPath = sys.argv[2]

print('In: ' + inputPath )
print('Out:'  +  outputPath)
print('Press Q to quit and save video')

frames = []

inputVideo = cv2.VideoCapture(inputPath)




frameNumber = 0
while(inputVideo.isOpened()):
   try:
      ret, frame = inputVideo.read()
      if(FPS < 0):
         FPS = inputVideo.get(cv2.CAP_PROP_FPS) #read fps
      frames.append(frame) # add frame to list
      frameNumber = frameNumber + 1
      if(maxFrames > 0 and frameNumber >= maxFrames): #check if max frames is reached
         break

      cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) #show video while loading
   except:
      break
   if( cv2.waitKey(1) & (0xFF == ord('q') or 0xFF == ord('Q'))):
      break
      

inputVideo.release()
cv2.destroyAllWindows()


frames = np.array(frames)

while(frames[-1] is None): #last frame is often None. Remove last Nones from list.
   frames = frames[:-1]


nf= np.zeros((len(frames), len(frames[0]), len(frames[0][0]),len(frames[0][0][0])))
for i,frame in enumerate(frames):
   nf[i] = frame.astype(np.uint16)

frames = nf

print('Lowest pixel: ' + str(np.amin(frames)))
print('Highest pixel: ' + str(np.amax(frames)))


if (outputPath[-3] is not "."):#put .avi to end of output.
   outputPath = outputPath + ".avi"

frames = frames + 1 #add 1 to prevent log(0)


frames = np.log((frames))
maxPixelValue = np.amax(frames) # no normalize again.


frames = (1.0*frames/maxPixelValue)*255.0 #normalizing
frames = frames.astype(np.uint8) #back to uint8

print("Saving to AVI: " + outputPath)

fourcc = cv2.VideoWriter_fourcc(*format)
videoFile = cv2.VideoWriter(outputPath,fourcc, FPS,(len(frames[0][0]),len(frames[0])),True) #this is now saved as colour, but should be changed to gray.
        
for i in range(np.shape(frames)[0]):
   videoFile.write(np.uint8(frames[i,:,:]))
       
videoFile.release()

print('done')





 