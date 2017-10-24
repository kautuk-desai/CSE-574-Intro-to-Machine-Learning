import numpy as np
import matplotlib.pyplot as plt

def plots():
    with open('optimization_output_synthetic.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    #str.replace(content)
    content = [x.strip() for x in content]
    #print(content)
    content_split = [contenteach.split() for contenteach in content]
    #print(content_split)
    M = np.array([content_split[4] for content_split in content_split]).astype(np.int)
    #M
    M = M[::10]
    _lambda = np.array([content_split[8] for content_split in content_split]).astype(np.float)
    plotdata = np.array([content_split[10] for content_split in content_split]).astype(np.float)
    plotdata = plotdata[::10]
    errMax = np.max(plotdata)
    errMin = np.min(plotdata)

    #plotdata
    plotdataboth = np.array([M*10,plotdata])
    plotdataMean = np.mean(plotdata)
    plotdataVar = np.var(plotdata)
    plotdata = (plotdata - plotdataMean)/(errMax - errMin)
    #plotdata = plotdata/np.sum(plotdata)
    fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    plt.plot(M, plotdata)
    plt.xlabel('M = 1 to 100 (lambda = 0)')
    plt.ylabel('normalized (over mean) Validation E_rms')
    plt.show()