## Dara Molloy (19/01/2023)

from inference import Inference
from benchmark import Benchmark
from Utils.Utils import filterAnno
from collections import OrderedDict
import numpy as np
import shutil
import os
import time
import glob
import json
from fastopenISP.asyncpipeline import Pipeline
from fastopenISP.utils.yacs import Config
from functools import reduce
import operator
import subprocess
import gc
import rawpy
import pandas as pd
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


## File Locations
project_name = "ISP-Evaluation"
configPath = 'fastopenISP/configs/'
raw_paths_canon= glob.glob("Data/RAW/*.CR2")
raw_paths_lumix= glob.glob("Data/RAW/*.RW2")
raw_paths_nikon= glob.glob("Data/RAW/*.nef")
rootFolder = "Data/"

## System Variables
num_processes = 6
gpu = 0

### ISP Variations, apologies for the formatting....
search_space = [[['gac', 'gamma'], [0.1, 0.15, 0.2, 0.3, 0.42, 0.4545, 0.6, 0.8, 1.0, 1.2, 2.0]],
                [['ceh', 'clip_limit'], [0, 0.02, 0.05, 0.075, 0.1, 0.2, 0.5, 1]],
                [['hsc', 'saturation_gain'], [0, 32, 64, 128, 192, 256, 320, 384, 512, 768, 1024]],
                [['hsc', 'hue_offset'], [0, 60, 120, 180, 240, 300]],
                [['cfa', 'mode'], ["malvar", "bilinear", "nn", "edge"]],
                [['bnf', 'multi'], [[["intensity_sigma", 0.35], ["spatial_sigma", 0.3], ["BNF_kernel_size", 5]],
                                    [["intensity_sigma", 0.5], ["spatial_sigma", 0.4], ["BNF_kernel_size", 5]],
                                    [["intensity_sigma", 1.2], ["spatial_sigma", 1], ["BNF_kernel_size", 5]],
                                    [["intensity_sigma", 4], ["spatial_sigma", 4], ["BNF_kernel_size", 5]],
                                    [["intensity_sigma", 6], ["spatial_sigma", 6], ["BNF_kernel_size", 7]],
                                    [["intensity_sigma", 8], ["spatial_sigma", 8], ["BNF_kernel_size", 9]],
                                    [["intensity_sigma", 16], ["spatial_sigma", 16], ["BNF_kernel_size", 13]],
                                    [["intensity_sigma", 36], ["spatial_sigma", 36], ["BNF_kernel_size", 21]],
                                    [["intensity_sigma", 72], ["spatial_sigma", 72], ["BNF_kernel_size", 25]]]],
                [['eeh', 'multi'],
                 [[["edge_gain", 0], ["flat_threshold", 16], ["delta_threshold", 32], ["kernel_size", 5], ["sigma", 3]],
                  [["edge_gain", 256], ["flat_threshold", 16], ["delta_threshold", 32], ["kernel_size", 5],
                   ["sigma", 3]],
                  [["edge_gain", 384], ["flat_threshold", 12], ["delta_threshold", 64], ["kernel_size", 5],
                   ["sigma", 3]],
                  [["edge_gain", 512], ["flat_threshold", 10], ["delta_threshold", 64], ["kernel_size", 7],
                   ["sigma", 3]],
                  [["edge_gain", 768], ["flat_threshold", 8], ["delta_threshold", 64], ["kernel_size", 7],
                   ["sigma", 3]],
                  [["edge_gain", 1024], ["flat_threshold", 8], ["delta_threshold", 64], ["kernel_size", 9],
                   ["sigma", 3]],
                  [["edge_gain", 1408], ["flat_threshold", 6], ["delta_threshold", 128], ["kernel_size", 13],
                   ["sigma", 3]],
                  [["edge_gain", 1792], ["flat_threshold", 4], ["delta_threshold", 128], ["kernel_size", 17],
                   ["sigma", 3]],
                  [["edge_gain", 2048], ["flat_threshold", 2], ["delta_threshold", 128], ["kernel_size", 21],
                   ["sigma", 3]]]],
                [['bcc', 'contrast_gain'], [64, 256, 512, 1024, 2048, 4096]]]

## Sample Search Space
# search_space = [[['bcc', 'contrast_gain'], [0, 256, 1024, 4096,80000]]]




## Can be used to delete ISP variation data
def cleanup_dataset(saveFolder):
    shutil.rmtree(saveFolder + "images/")

### Returns white balance values for each raw file, enabling fastopenISP AWB.py
def getWhiteBalanceGains(rawFile, gainMult):
    (redGain,greenGain,blueGain, offset) = rawFile.daylight_whitebalance
    redGain = int(np.multiply(redGain,gainMult).astype('u2'))
    greenGain = int(np.multiply(greenGain, gainMult).astype('u2'))
    blueGain = int(np.multiply(blueGain, gainMult).astype('u2'))
    return [redGain, greenGain, blueGain]

## Function that is passed to fastopenISP to retrieve the bayer information from the raw file
def load_bayer(raw_path):
    data = OrderedDict()
    raw = rawpy.imread(raw_path)
    bayer = np.asarray(raw.raw_image_visible)
    if ".CR2" in raw_path.upper():
        [redGain, greenGain, blueGain] = getWhiteBalanceGains(raw, gainMult=1024)
    elif ".NEF" in raw_path.upper():
        [redGain, greenGain, blueGain] = getWhiteBalanceGains(raw, gainMult=4)
    else:
        [redGain, greenGain, blueGain] = getWhiteBalanceGains(raw, gainMult=1024)
    data['RGB'] = [redGain, greenGain, blueGain]
    data['bayer'] = bayer
    return data

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

## Takes a COCO JSON annotation file and generates a new one filtered based on the image_paths
def generateJSON(fullAnnoPath, currAnnoPath, image_paths):
    currentImages = []
    new_anno = {}

    for path in image_paths:
        currentImages.append(os.path.basename(path)[:-4])

    with open(fullAnnoPath, 'r') as f:
        fullAnno = json.load(f)

    categories = fullAnno['categories']
    images = fullAnno['images']
    annotations = fullAnno['annotations']

    curr_img_ids = []
    for img in images:
        if img['file_name'][:-4] in currentImages:
            curr_img_ids.append(img['id'])

    new_images = []
    new_annotations = []
    for id in curr_img_ids:
        for img in images:
            if id == int(img['id']):
                new_images.append(img)
        for anno in annotations:
            if id == int(anno['image_id']):
                new_annotations.append(anno)
    new_anno['images'] = new_images
    new_anno['annotations'] = new_annotations
    new_anno['categories'] = categories

    with open(currAnnoPath,'w') as f:
        json.dump(new_anno,f)




if __name__ == '__main__':
    inc = 0
    for i in range(len(search_space)):  ## For each chosen ISP block
        for param in search_space[i][1]: ## For each ISP block parameter; run the inference and benchmarking

            ### APPLY ISP ###

            runName = int(time.time()*100)
            saveFolder = f"{rootFolder}Output/ISPVariations/{runName}-{inc}/"
            os.makedirs(saveFolder)
            os.makedirs(saveFolder + "Images/")

            if len(raw_paths_nikon) != 0:
                ## Run ISP on Nikon Images
                cfg = Config(configPath + 'PASCALRAW.yaml')
                dictNames = []
                params = []
                if search_space[i][0][1] == 'multi':
                    for combo in param:
                        dictNames.append([search_space[i][0][0], combo[0]])
                        params.append(combo[1])
                else:
                    dictNames.append(search_space[i][0])
                    params.append(param)
                with cfg.unfreeze():
                    for j in range(len(dictNames)):
                        setInDict(cfg, dictNames[j], params[j])
                        print(f"{dictNames[j]}: {params[j]}")
                pipeline = Pipeline(cfg)
                pipeline.batch_run(raw_paths_nikon, saveFolder + "Images/", load_bayer, suffixes='',
                                   num_processes=num_processes)
                print("Waiting for ISP to Finish.....")
                time.sleep(30)
                del pipeline


            if len(raw_paths_canon) != 0:
                ## Run ISP on Canon Images
                cfg = Config(configPath+'canon.yaml')
                dictNames = []
                params = []
                if search_space[i][0][1] == 'multi':
                    for combo in param:
                        dictNames.append([search_space[i][0][0],combo[0]])
                        params.append(combo[1])
                else:
                    dictNames.append(search_space[i][0])
                    params.append(param)
                with cfg.unfreeze():
                    for j in range(len(dictNames)):
                        setInDict(cfg,dictNames[j],params[j])
                        print(f"{dictNames[j]}: {params[j]}")
                pipeline = Pipeline(cfg)
                pipeline.batch_run(raw_paths_canon, saveFolder + "Images/", load_bayer, suffixes='', num_processes=num_processes)
                print("Waiting for ISP to Finish.....")
                time.sleep(30)
                del pipeline


            if len(raw_paths_lumix) != 0:
                ## Run on Lumix Images
                cfg = Config(configPath+'lumix.yaml')
                dictNames = []
                params = []
                if search_space[i][0][1] == 'multi':
                    for combo in param:
                        dictNames.append([search_space[i][0][0],combo[0]])
                        params.append(combo[1])
                else:
                    dictNames.append(search_space[i][0])
                    params.append(param)
                with cfg.unfreeze():
                    for j in range(len(dictNames)):
                        setInDict(cfg,dictNames[j],params[j])
                        print(f"{dictNames[j]}: {params[j]}")
                pipeline = Pipeline(cfg)
                pipeline.batch_run(raw_paths_lumix, saveFolder + "Images/", load_bayer, suffixes='', num_processes=num_processes)
                print("Waiting for Lumix ISP to Finish.....")
                time.sleep(30)
                del pipeline


            ## Wait until GPU free if running multiple in parallel
            gpu_taken = True
            while (gpu_taken):
                memStr = str(subprocess.check_output(f"nvidia-smi --query-gpu=memory.used --id={gpu} --format=csv"))
                mem = int(str(memStr).split("\\r\\")[1][1:][:-3])
                if mem < 4000:
                    gpu_taken = False
                else:
                    print("Waiting for GPU......")
                    time.sleep(30)

            ### INFERENCE ###

            ## Image Files
            imgFiles = glob.glob(saveFolder + "Images/*.png")
            ## Annotation Path
            annoPath = rootFolder + "Annotations/full_anno.json"
            predDir = saveFolder + "Predictions/"
            os.makedirs(predDir)

            ## The label IDs and corresponding text labels that we want to predict, see COCO list if changes are needed
            labelDict = {
                0: "Background",
                1: "Person",
                2: "Bicycle",
                3: "Car"
            }
            ## Infer based on above info
            print("Runnning Inference")
            inference = Inference(predDir, labelDict, 0.01, 30, imageSize=(640, 338), annoPath= annoPath)
            _ = inference.batch_run(imgFiles=imgFiles, modelList=None, saveJSON=True, visualise=True)
            del inference

            ### BENCHMARK ###

            ## Glob JSONs saved in benchmark
            predJSONs = glob.glob(predDir + "*.json")
            ## Path to annotation file containing only the images that are glob
            imgFiles = raw_paths_nikon+raw_paths_canon+raw_paths_lumix
            annoPath = filterAnno(imgFiles,annoPath)
            ## Making Results Directory
            resDir = saveFolder + "Results/"
            os.makedirs(resDir)
            ## Run the benchmark for each prediction JSON against the newest annoPath annotation JSON and output a results file
            print("Runnning Benchmark")
            benchmark = Benchmark()
            results = benchmark.batch_run(predictions=predJSONs, annoPath=annoPath, outDir=resDir, saveCSV=True,
                                          batchName=[inc,search_space[i],param])
            del benchmark

            with open(f"{saveFolder}config-{inc}.txt", "w") as f:
                f.write(str(cfg))

            ## Uncomment to remove all ISP variation images, needed when dataset too large
            #shutil.rmtree(saveFolder + "Images/")

            inc+=1

            collected = gc.collect()
            print(f"{collected} objects collected")

    ## Concatenate Result CSVs Together
    resultCSVs = glob.glob(f"{rootFolder}/Output/**/Results.csv",recursive=True)
    allRes = pd.DataFrame()
    for resFile in resultCSVs:
        dfTemp = pd.read_csv(resFile)
        allRes = allRes.append(dfTemp)
    allResDir = f"{rootFolder}/Output/Results/"
    if not os.path.exists(allResDir):
        os.makedirs(allResDir)
    allRes.to_csv(allResDir+"OverallResults.csv")





