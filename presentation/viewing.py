import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import os
import datetime
import re




useRuns = range(34,40)
today = datetime.datetime.now().strftime("%d-%m-%y")
today = "14-03-18"
signalNoise = []
data = []
folders = []

def getParticleData(file):
    try:
        output = []
        #print("Reading " + file)
        f = open(file, "r")
        input = f.read().split("\n")
        f.close()
        del input[-1]
        for element in input:
            temp = element.split("    ")
            output.append( float(temp[1]))
        return output
    except:
        print("Cannot read file.")



for run in useRuns:
    contentOfFolder = os.listdir("./simulation/runs/" + today + "/run"+str(run))
    for folder in contentOfFolder:    
        folders.append("./simulation/runs/" + today + "/run"+str(run) +"/"+ folder)


def writeMetadata(text, folder = "./",file = "metadata.txt"):
    try:
        f = open(folder + '/'+ file, "a")
        f.write(text)
        f.close()
    except:
        print("Cannot make metadata file.")


for folder in folders:
    found = False

    if(os.path.isfile(folder)):
        break
    print(folder)
    
    signal = int(re.search('DeltaSignal_(.+?)_noise', folder ).group(1))
    noise = int(re.search('noise_(.+?$)', folder ).group(1))
    if (len(signalNoise) < 1):
        data.append(getParticleData(folder + '/mass.dat'))
        signalNoise.append([signal,noise])
    else:
        for i in range(len(signalNoise)):
            if ([signal, noise] == signalNoise[i]):
                data[i] = data[i] + getParticleData(folder + '/mass.dat')
                found = True
                break
        if(not found):
            data.append(getParticleData(folder + '/mass.dat'))
            signalNoise.append([signal,noise])
            found = False


for massList in data:
    massList.sort()
    
def gaussianDistribution(mu, sigma, x):
    sigma2 = sigma*sigma
    return np.exp(-(x-mu)*(x-mu)/(2*sigma2))/np.sqrt(2*np.pi*sigma2)

    
for i in range(len(signalNoise)):

    """
    This assumes an equal distribution of both particle species. I have to find a way to separate
    two particle species based on their mass.
    """
    massList = data[i]
    
    plotXMin = round(int(np.amin(massList))-500,3)
    plotXMax = round(int(np.amax(massList))+500,3)
    print(plotXMin)

    
    mu1 = np.mean(massList[:int(len(massList)/2)])
    mu2 = np.mean(massList[int(len(massList)/2):])
    mu1SquaredList = []
    mu2SquaredList = []
    for mu in massList[:int(len(massList)/2)]:
        mu1SquaredList.append(mu*mu)
    for mu in massList[int(len(massList)/2):]:
        mu2SquaredList.append(mu*mu)
        
    mu1Squared = np.mean(mu1SquaredList)
    mu2Squared = np.mean(mu2SquaredList)
    sigma1 = np.sqrt(mu1Squared - mu1*mu1)
    sigma2 = np.sqrt(mu2Squared - mu2*mu2)
  
    
    fig, ax = plt.subplots()
    #plt.hist(massList)
    n1, bins1, patches1 = ax.hist(massList[:int(len(massList)/2)], 10)
    n2, bins2, patches2 = ax.hist(massList[int(len(massList)/2):], 10)
    fit  = np.amax(n1)*gaussianDistribution(mu1,sigma1,range(plotXMin,plotXMax))/(gaussianDistribution(mu1,sigma1,mu1))+np.amax(n2)*gaussianDistribution(mu2,sigma2,range(plotXMin,plotXMax))/(gaussianDistribution(mu2,sigma2,mu2))
    overlap = range(int(np.amin([mu1,mu2])),int(np.amax([mu1,mu2])))

    
    fit2  = np.amax(n1)*gaussianDistribution(mu1,sigma1,overlap)/(gaussianDistribution(mu1,sigma1,mu1))+np.amax(n2)*gaussianDistribution(mu2,sigma2,overlap)/(gaussianDistribution(mu2,sigma2,mu2))    
    minimum=np.amin(fit2)
    print(minimum)
    max1 = np.amax(n1)*gaussianDistribution(mu1,sigma1,mu1)/(gaussianDistribution(mu1,sigma1,mu1))+np.amax(n2)*gaussianDistribution(mu2,sigma2,mu1)/(gaussianDistribution(mu2,sigma2,mu2))
    max2 = np.amax(n1)*gaussianDistribution(mu1,sigma1,mu2)/(gaussianDistribution(mu1,sigma1,mu1))+np.amax(n2)*gaussianDistribution(mu2,sigma2,mu2)/(gaussianDistribution(mu2,sigma2,mu2))
    print(max1)
    print(max2)
    
    if(minimum > 0.5*np.amin([max1,max2]) ):
        print("Particles cannot be distinguished")
        writeMetadata(str(signalNoise[i][0]) + "," + str(signalNoise[i][1]) + "," + "0,", file= "output.dat")
    else:
        print("Particles can be distinguished")
        writeMetadata(str(signalNoise[i][0]) + "," + str(signalNoise[i][1]) + "," + "1,", file= "output.dat")
    for run in useRuns:
       writeMetadata(str(run) + ",", file="output.dat")
    writeMetadata("\n", file="output.dat")
    plt.plot(range(plotXMin,plotXMax),fit)
    #plt.plot(range(10000,13000),np.amax(n2)*gaussianDistribution(mu2,sigma2,range(10000,13000))/(gaussianDistribution(mu2,sigma2,mu2)))
    #plt.plot(range(10000,13000),np.amax(n2)*gaussianDistribution(mu2,sigma2,range(10000,13000))/(gaussianDistribution(mu2,sigma2,mu2)))
    
    plt.xlabel('Mass')
    plt.ylabel('Count')
    plt.savefig('./histogramS' + str(signalNoise[i][0]) + 'N' + str(signalNoise[i][1]) + '.pdf')
    plt.show()

    
