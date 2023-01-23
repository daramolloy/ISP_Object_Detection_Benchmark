import glob
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import StrMethodFormatter

# resName = "DanganGL"
# path = "G:/ISP_Paper/Results/GridSearch/580Dangan-AllModels-AllConfigs-COCO/Results/*.csv"
# resPath = "G:/ISP_Paper/Results/GridSearch/580Dangan-AllModels-AllConfigs-COCO/Plots/"
resName = "Retrain"
path = "G:/ISP_Paper/Results/GridSearch/ModelRetraining/Results/*.csv"
resPath = "G:/ISP_Paper/Results/GridSearch/ModelRetraining/Plots/"
# resName = "Dangan-Retrained"
# path = "G:/ISP_Paper/Results/GridSearch/Important/Final/Retraining2/Results/*.csv"
# resPath = "G:/ISP_Paper/Results/GridSearch/Important/Final/Retraining2/Results/"


#metric = 3
for metric in tqdm(range(3,91)):
    # metric = {"AP":2,"AP50":3,"AP5095-MEAN": 4,"AP5095-STDDEV": 5,"AP5095-STDERROR": 6 ,"AP50-MEAN": 7,"AP50-STDDEV": 8,
    #           "AP50-STDERROR": 9,"AR5095_100-MEAN": 10,"AR5095_100-STDDEV": 11,"AR5095_100-STDERROR": 12,
    #           "AP5095small-MEAN": 13,"AP5095small-STDDEV": 14,"AP5095small-STDERROR": 15 }
    errorBars = False
    subset_flag = False

    results = []

    for res in glob.glob(path):
        if "fasterrcnn_resnet50_fpn_coco_results.csv" not in res and "yolov5m_coco_results.csv" not in res and subset_flag: #and "ssd300_vgg16_coco_results.csv" not in res
            continue
        with open(res,"r") as f:
            contents = f.readlines()
        metricName = contents[0][:-1].split(',')[metric]
        if "MEAN" not in metricName and errorBars:
            continue
        currRes = []
        for line in contents:
            if "AP" not in line and len(line) > 5:
                if "bnf" in line:
                    currLineRes = line[:-1].split(',')
                    currLineRes[2] = f"IS{line.split(',')[2].split(' ')[0]}\nSS{line.split(',')[2].split(' ')[1]}\nKS{line.split(',')[2].split(' ')[2]}"
                elif "eeh" in line:
                    currLineRes = line[:-1].split(',')
                    currLineRes[2] = f"EG{line.split(',')[2].split(' ')[0]}\nFT{line.split(',')[2].split(' ')[1]}\nDT{line.split(',')[2].split(' ')[2]}\nKS{line.split(',')[2].split(' ')[3]}"
                else:
                    currLineRes = line[:-1].split(',')
                currRes.append(currLineRes)
        # results.append([os.path.basename(res)[:-17],currRes])
        results.append([os.path.basename(res)[:-4],currRes])

    if len(results) != 0:
        configName = ["gamma", "saturation","clip_limit","bnf","eeh","contrast","hue","cfa"]
        hvParamName = [0.42,256,0.02,"IS0.5\nSS0.4\nKS5","EG384\nFT12\nDT64\nKS5",256,0,"malvar"]
        hvBaseline = {}

        NUM_COLORS = len(glob.glob(path))
        rows = 3
        cols = 3
        inc = 0
        font = {'size':13}
        fontBig = {'size':15}
        colours = [tuple(np.array([255,0,0])/255),tuple(np.array([0,255,0])/255),tuple(np.array([255,128,0])/255),
                   tuple(np.array([255,0,150])/255),tuple(np.array([228, 226, 27])/255),
                   tuple(np.array([120,120,120])/255),tuple(np.array([0,131,12])/255),
                   tuple(np.array([0,255,26])/255),tuple(np.array([91,255,160])/255),
                   tuple(np.array([0,155,255])/255),
                   tuple(np.array([0,64,129])/255),
                   tuple(np.array([135,43,255])/255),tuple(np.array([0,0,0])/255),
                   tuple(np.array([233,0,250])/255),tuple(np.array([150,35,85])/255),
                   tuple(np.array([0,0,255])/255),tuple(np.array([255,255,255])/255)]
        if subset_flag:
            colours = [tuple(np.array([255,0,150])/255),#tuple(np.array([0,131,12])/255),
                       tuple(np.array([0,155,255])/255),
                       tuple(np.array([0,64,129])/255),
                       tuple(np.array([135,43,255])/255),tuple(np.array([0,0,0])/255),
                       tuple(np.array([233,0,250])/255),tuple(np.array([150,35,85])/255),
                       tuple(np.array([0,0,255])/255),tuple(np.array([255,255,255])/255)]
        # colours=[tuple(np.array([255,0,0])/255),tuple(np.array([30,255,30])/255)] # For Retrained results

        # colours=[]
        # for tot in range(len(results)):
        #     random_colour=np.random.choice(range(1000),size=3)/1000
        #     colours.append(tuple(random_colour))

        plt.figure(figsize=(18, 10))
        for name in configName:
            xString = False
            if "bnf" in name or "eeh" in name or "cfa" in name:
                xString = True
            inc+=1

            # if "gamma1111" in name:
            #     plt.subplot(rows,cols,inc,xlim=(0,2))
            # elif "clip1111" in name :
            #     plt.subplot(rows,cols,inc,xlim=(-0.1,1))
            # elif "contrast1111" in name :
            #     plt.subplot(rows,cols,inc,xlim=(0,2048))
            # else:
            plt.subplot(rows, cols, inc)
            #plt.subplot(rows, cols, inc)

            for res in results:
                x = []
                y = []
                yerr  =[]
                model = res[0]
                for config in res[1]:
                    if name in config[1]:
                        if xString:
                            x.append(config[2])
                        else:
                            x.append(float(config[2]))
                        y.append((float(config[metric])*100))
                        if errorBars:
                            yerr.append((float(config[metric+2])*1.96))
                        if str(hvParamName[configName.index(name)]) == str(config[2]):
                            hvBaseline[model+name] = (float(config[metric])*100)
                if not errorBars:
                    plt.plot(x, y,color=colours[results.index(res)], label=model)

                else:
                    plt.errorbar(x,y,yerr=yerr,color=colours[results.index(res)], label=model)
            #plt.title(F"{name.replace('_',' ').upper()}")

            if xString and "cfa" not in name:
                plt.xticks(fontsize=9)
            else:
                plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            if "clip" in name:
                xLabel = "CEH CLIP LIMIT"
            else:
                xLabel = name
            plt.xlabel(xLabel.upper(),fontdict=fontBig)

            plt.ylabel(metricName,fontdict=font)
            plt.grid(linestyle = 'dashed')
            plt.axvline(x=hvParamName[configName.index(name)], color='black',linestyle='dashed')
        #plt.suptitle(resName+"-Absolute")
        labels = [row[0][:-13] for row in results]
        plt.figlegend(labels, loc="lower right", borderaxespad=0.5,prop={'size': 12})
        plt.subplots_adjust(left=0.06,right=0.957,bottom=0.055,top=0.99,hspace=0.42,wspace=0.35)
        figName = f"{resPath}{resName}-Absolute-{metricName}.png"
        plt.savefig(figName,dpi=450)
        #plt.show()


        plt.figure(figsize=(18, 10))
        inc=0
        for name in configName:
            xString = False
            if "bnf" in name or "eeh" in name or "cfa" in name:
                xString = True
            inc+=1
            # if "gamma11111" in name:
            #     plt.subplot(rows,cols,inc,xlim=(0,1.5),ylim=(-18,8))
            # elif "clip11111" in name :
            #     plt.subplot(rows,cols,inc,xlim=(-0.1,1))
            # elif "contrast111111" in name :
            #     plt.subplot(rows,cols,inc,xlim=(0,2048),ylim=(-50,2))
            # else:
            plt.subplot(rows, cols, inc)
            for res in results:
                x = []
                y = []
                model = res[0]
                for config in res[1]:
                    if name in config[1]:
                        if xString:
                            x.append(config[2])
                        else:
                            x.append(float(config[2]))
                        performance = (float(config[metric])*100)
                        if hvBaseline[model+name] == 0: # Added
                            y.append(0)
                            continue
                        relativePerformance = (performance - hvBaseline[model+name])
                        # y.append(relativePerformance) ## AP diff
                        y.append((relativePerformance/hvBaseline[model+name])*100) ## % AP Diff
                plt.plot(x, y,color=colours[results.index(res)], label=model)

            #plt.title(F"{name.replace('_',' ').upper()}") ## Title for each Sub Figure
            if xString and "cfa" not in name:
                plt.xticks(fontsize=9)
            else:
                plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            if "clip" in name:
                xLabel = "CEH CLIP LIMIT"
            else:
                xLabel = name
            plt.xlabel(xLabel.upper(),fontdict=fontBig)
            plt.ylabel(f"{metricName} % Difference",fontdict=font)
            plt.grid(linestyle='dashed')
            plt.axvline(x=hvParamName[configName.index(name)], color='black',linestyle='dashed')
        #plt.suptitle(resName+"-Relative")  ## Overall Title
        labels = [row[0][:-13] for row in results]
        plt.figlegend(labels, loc="lower right", borderaxespad=0.5,prop={'size': 12})
        plt.subplots_adjust(left=0.06,right=0.957,bottom=0.055,top=0.99,hspace=0.42,wspace=0.35)
        figName = f"{resPath}{resName}-Relative-{metricName}.png"
        plt.savefig(figName,dpi=450)
        #plt.show()

