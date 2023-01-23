import json
import glob
import os.path
import numpy as np
import cv2
import shutil

def filterPred(predDict):
    masked_off = np.array([[0, 2160], [0, 1950], [2130, 440], [2290, 440], [3800, 2160]])
    filteredPred = []
    for pred in predDict:
        centreX = pred['bbox'][0] + (pred['bbox'][2] / 2)
        centreY = pred['bbox'][1] + (pred['bbox'][3] / 2)
        inside_path = cv2.pointPolygonTest(masked_off, (int(centreX), int(centreY)), False)
        if inside_path == 1:
            filteredPred.append(pred)

    return filteredPred


## Filter current annotation file to contain only images in the predictions
## (i.e. useful for when you want to benchmark a subset of the data, this allows you to not create a new anno file each time)
def filterAnno(imgGlob,fullAnnoPath):

    filteredAnno = {}
    with open(fullAnnoPath, "r") as f:
        allAnno = json.load(f)

    newImages = []
    newIDs = []
    for img in allAnno['images']:
        for globImg in imgGlob:
            if os.path.basename(globImg).split('.')[0] == img['file_name'].split('.')[0] and img['id'] not in newIDs:
                newImages.append(img)
                newIDs.append(img['id'])

    newAnno = []
    for nImg in newImages:
        for ann in allAnno['annotations']:
            if nImg['id'] == ann['image_id']:
                newAnno.append(ann)

    filteredAnno['annotations'] = newAnno
    filteredAnno['images'] = newImages
    filteredAnno['categories'] = allAnno['categories']

    outPath = os.path.dirname(fullAnnoPath) + "/filtered_annotations.json"
    with open(outPath,"w") as f:
        json.dump(filteredAnno, f)

    return outPath

def sortTempOutputs(batchName,outDir,annoPath,predJSONs):
    newOutPath = f"{outDir}{batchName}-Outputs/"
    if not os.path.exists(newOutPath):
        os.mkdir(newOutPath)
    if "filtered" in annoPath:
        shutil.move(annoPath,newOutPath)
    for pred in predJSONs:
        shutil.move(pred,newOutPath)