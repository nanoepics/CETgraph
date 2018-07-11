import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import datetime
import pypylon
import h5py
import cv2

exposureTime = 20000 # exposure time in useconds
laserCurrent = 90 # laser current in mA
maxFrames = 1000
gain = 0
maxFramesString = ""
FPSString = ""
gainString = ""
signalString = ""
potentialString = ""
signalType = ""

frequencyString = ""
electricFrequency = 0

subtractBackground = True



pythonLocation = "E:\\Peter\\Anaconda\\python.exe"
backgroundSubtractionScript = '"E:\\Peter\\python scripts\\subtractBackground.py"'
trackScript = '"E:\\Peter\\python scripts\\trackParticles.py"' 
plotScript = '"E:\\Peter\\python scripts\\plotHistogramSilent.py"' 

today = datetime.datetime.now().strftime("%y-%m-%d")

FOV = [-1,-1]


available_cameras = pypylon.factory.find_devices()
print(available_cameras)

if (len(available_cameras) == 1):
   camb = pypylon.factory.create_device(available_cameras[0])
elif( len(available_cameras) > 1):
   for i in range(len(available_cameras)):
      if str(available_cameras[i])[33:41] == '21958373': #camera id
         camb = pypylon.factory.create_device(available_cameras[i])

else:
    print("no camera found")




particleDiameterString = input("Enter label for measured sample (e.g. 30nm gold): ")
laserCurrentString = input("Enter laser current (mA) (Null = 89,9 mA) : ")
if(laserCurrentString == ""):
   laserCurrentString = "89.9"
print("Laser at " + laserCurrentString + " mA")
laserCurrent = float(laserCurrentString)

potentialString = input("Enter electric potential (blank is 0.0): ")
if(potentialString == ""):
   electricPotential = 0.0
else:
   electricPotential = float(potentialString)

signalType = input("Enter signal type (blank = block, b = block, s = sin. Enter 'b' for DC ): ").lower()

if(signalType == "s" or signalType == "sin"):
   signalType = 'sin'
elif(signalType == "" or signalType == "b" or signalType == "block" ):
   signalType = 'block'
else:
   print(signalType + " not recognized as a signal type. use 'b' or 's' " )

frequencyString = input("Enter electric frequency in Hz (0 or -1 for DC): ")
if(frequencyString == ""):
   electricFrequency = 0
else:
   electricFrequency = float(frequencyString)

   


FPSString = input("Enter FPS (or leave blank for 55 Hz): ")
maxFramesString = input("Enter maximum number of frames: ")
XFOVString = input("Enter width of imgage in pixels (leave blank to skip, start form top left corner): ")
YFOVString = input("Enter height of image in pixels: ")

objectiveString = input("Objective? (Null = green, g = green, r = red, b = brown, y = yellow, 63x = light blue 63x, 40x = light blue 40x)" ).lower()

if(objectiveString == "r" or objectiveString == 'red'):
   pixelSize = 0.75374991 #red objective um/pixel
elif(objectiveString == "y"):
   pixelSize = 0.37252272 #yellow objective um/pixel
elif(objectiveString == "b"):
   pixelSize = 1.44780657
   print('Are you sure you want to use the brown objective (2.5x/0.075)?')
elif(objectiveString == '63' or objectiveString == '63x'):
   pixelSize = 0.05859364
elif(objectiveString == '40' or objectiveString == '40x' ):
   pixelSize = 0.09374971
else:
   pixelSize = 0.225664 #green objective um/pixel
print("The pixel size is set to " + str(pixelSize) + " um/pixel")

if(XFOVString == ""):
	XFOVString = "1280"
if(YFOVString == ""):
	YFOVString = "960"
FOV[0] = int(XFOVString)
FOV[1] = int(YFOVString)

if(FOV[0] > 1280 or FOV[0] < 0):
	FOV[0] = 1280
if(FOV[1] > 960 or FOV[1] < 0):
	FOV[1] = 960

maxFrames = int(maxFramesString)
print("Ouput data will be: " + str(maxFrames) + " frames with size: " + str(FOV[0]) + "x" + str(FOV[1]))
subtractBGString = input("Subtract background? (Y/N, blank is yes): ").lower()

if(subtractBGString == "n" or subtractBGString == "no" or subtractBGString == "nee"):
   subtractBackground = False

trackString= input("Track particles? (Y/N, blank is no): ").lower()

if(trackString == "y" or trackString== "yes" or trackString== "ja" or trackString == 'j'):
   trackParticles = True
else:
   trackParticles = False


gainString = input("Enter Gain level (type -1 for max, 0 or blank for 0): ")

if(gainString == "" or gainString == None):
	gain = 0.0
	print("Gain set to 0")
else:
	gain = float(gainString)
	if(gain < 0):
		gain = 18.027804
	if(gain > 18.027804):
		print("Is de gain niet wat aan de hoge kant? (gain = " + str(gain) + ")\n")

"""
print(FPSString)
if(FPSString == "" or FPSString == None):
	exposureTimeString = input("Enter exposure time (us):"  )
	exposureTime = int(exposureTimeString)
	FPS = 1/(exposureTime*1000000)
else:
	FPS = int(FPSString)
	exposureTime = 1000000/FPS

"""

if(FPSString == "" or FPSString == None):
   FPSString = "55"

FPS = int(FPSString)
exposureTime = 1000000/FPS



print("FPS = " + str(FPS) + " ; Exposure time = " + str(exposureTime) + " us ; Gain: " + str(gain) + " ; Max frames: " + str(maxFrames)  )	
	






runs = 0
today = datetime.datetime.now().strftime("%d-%m-%y")
print(today)
filename = 'CameraData'






if not os.path.isdir("./data/"):
    os.mkdir("./data")
    
if not os.path.isdir("./data/" + today):
    os.mkdir("./data/" + today)


files = os.listdir('./data/' + today)

for file in files:
    if(file[3:].isdigit() and file[0:3] == 'run' and int(file[3:]) > runs):
        runs = int(file[3:])
runs  = runs + 1

currentPath = './data/' + today + '/run' + str(runs)
    
os.mkdir(currentPath)


metadataText = ''

def writeMetadata(text):
    try:
        f = open(currentPath + '/metadata.txt', "a")
        f.write(text)
        f.close()
    except:
        print("Cannot make metadata file.")

try:
    f = open(currentPath + '/metadata.txt', "w")
    f.write(metadataText)
    f.close()
except:
    print("Kan geen metadatabestand aanmaken.")

print("start run " + str(runs))




writeMetadata("Run " + str(runs) +  ' ' + str(datetime.datetime.now()) + "\n")

"""
cam_initialize and cam_config by Jeroen. cam_initialize sets all data except exposure time.
cam_config returns a list of data for a metadata file.
"""

def cam_initialize(basler_id):
    global gain
    basler_id.open()
    #basler_id.properties['PixelFormat']='Mono12'
    basler_id.properties['Gain'] = gain
    basler_id.properties['AcquisitionFrameRate']=500
    basler_id.properties['DefectPixelCorrectionMode']='Off'
    basler_id.properties['ContrastEnhancement']=0.0
    basler_id.properties['GainAuto']='Off'
    basler_id.properties['ExposureAuto']='Off'
    basler_id.properties['BlackLevel']=0.0



    return None


def cam_config(basler_id):
   counter=0
   settings=[]
   for key in basler_id.properties.keys():
      settings.append([])
      try:
         value = basler_id.properties[key]
      except IOError:
         value = '<NOT READABLE>'
      settings[counter].append(key)
      settings[counter].append(value)
      counter=counter+1
	
   return settings



print(camb)

print("initialize camera")
camb.open()
camb.properties['PixelFormat'] = 'Mono12'
print(camb.properties['PixelFormat'])



cam_initialize(camb)
camb.properties['ExposureTime'] = exposureTime
writeMetadata("\nLaser current: "+ str(laserCurrent)+"\n")
writeMetadata("Laser current (user): " + laserCurrentString + "\n")
writeMetadata("Electric properties:\n ")
writeMetadata("['ElectricPotential', "+ str(electricPotential) +"]\n")
writeMetadata("['SignalType', '"+ signalType +"']\n")
writeMetadata("['ElectricFrequency', "+ str(electricFrequency) +"]\n")
writeMetadata("Particle: " + particleDiameterString+"\n" )
writeMetadata("['PixelSize', "+ str(pixelSize) +"]\n")
writeMetadata("\nCamera settings:\n")

if(camb.properties["Gain"] != 0):
	print("Warning: gain != 0   (gain: " + str(camb.properties["Gain"]) + ")")

count = 0

print("start imaging")
rawData = []
for image in camb.grab_images():
   rawData.append(np.array(image)[:FOV[1],:FOV[0]])
   maxFrames = maxFrames - 1
   if(maxFrames <= 0):
      break
   count = count + 1
   if(count > FPS):
      count = 0
      frame  = np.uint16(np.array(image)[:FOV[1],:FOV[0]]*65536/4096)
      cv2.imshow('frame', frame) #show video while loading
      if( cv2.waitKey(1) & (0xFF == ord('q') or 0xFF == ord('Q'))):
         break
   #print(image.shape)
   #plt.imshow(image)
   #plt.show()

print("imaging done")
print("max pixel value (12 bit): " + str(np.amax(np.array(rawData[0]))))


hf = h5py.File(currentPath + '/data.h5', 'w')
hf.create_dataset('Basler data', data = rawData)
hf.close()







print("Data taken: " +  str(np.shape(rawData)[0]) + ' ' + str(np.shape(rawData)[1]) + 'x'  +str(np.shape(rawData)[2]))

for setting in cam_config(camb): 
   writeMetadata(str(setting))
   writeMetadata('\n')






FPS = camb.properties['ResultingFrameRate']


camb.close()


aviData = np.array(rawData)
aviData = 1.0*aviData/np.amax(np.array(aviData))*255
aviData = np.uint8(aviData)
print(np.amax(aviData))


fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoFile = cv2.VideoWriter(currentPath +'/' + filename + '.avi',fourcc,FPS,(len(aviData[0][0]),len(aviData[0])),False)
for i in range(np.shape(aviData)[0]):
	videoFile.write(np.uint8(aviData[i]))
videoFile.release()

if(subtractBackground):
   print("Subtract background:")
   command = pythonLocation  + " " + backgroundSubtractionScript + " " + currentPath
   os.system(command)
   print("background subtracted")


if(trackParticles):
   print("Track particles:")
   command = pythonLocation  + " " + trackScript + " " + currentPath
   os.system(command)
   print("Particles tracked")

"""
runs = 0
files = os.listdir(currentPath + "\\tracking\\" + today)
for file in files:
   if(file[3:].isdigit() and file[0:3] == 'run' and int(file[3:]) > runs):
      runs = int(file[3:])
"""

print('done')