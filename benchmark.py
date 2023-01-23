import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE
import tidecv.datasets as datasets
import pandas as pd
from tqdm import tqdm
import csv
import json
import math
import statistics

class Benchmark:
    def __init__(self):
        header = ["Config","ISP Setting","Parameters","Model", "AP5095", "AP50", "AP75", "APsmall", "APmedium","APlarge","AR5095_1","AR5095_10","AR5095_100",
                  "AR5095_100_small","AR5095_100_medium","AR5095_100_large"]

        for className in ["person", "bicycle", "car"]:
            header.append(f"AP5095_{className}")
            header.append(f"AP50_{className}")
            header.append(f"AP75_{className}")
            header.append(f"APsmall_{className}")
            header.append(f"APmedium_{className}")
            header.append(f"APlarge_{className}")
            header.append(f"AR5095_1_{className}")
            header.append(f"AR5095_10_{className}")
            header.append(f"AR5095_100_{className}")
            header.append(f"AR5095_100_small_{className}")
            header.append(f"AR5095_100_medium_{className}")
            header.append(f"AR5095_100_large_{className}")
        header.extend(["TIDE-AP5095", "AP50", "AP55", "AP60", "AP65", "AP70", "AP75", "AP80", "AP85", "AP90", "AP95",
                       "CLS", "LOC", "BOTH", "DUPE","BKG", "MISS", "FalsePos", "FalseNeg"])

        for className in ["person", "bicycle", "car"]:
            header.append(f"CLS_{className}")
            header.append(f"LOC_{className}")
            header.append(f"BOTH_{className}")
            header.append(f"DUPE_{className}")
            header.append(f"BKG_{className}")
            header.append(f"MISS_{className}")
        header.append("MeanConf")
        header.append("STDDEVConf")
        header.append("STDERRConf")

        self.header = header


    def batch_run(self,predictions=None,annoPath=None,saveCSV=None,outDir=None,batchName=None):
        data = []

        cocoAnno = COCO(annoPath)
        tide_gt = datasets.COCO(annoPath)

        for pred in predictions:
            currData = [batchName[0],' '.join(batchName[1][0]),batchName[2],os.path.basename(pred)[:-10]]

            ## Standard COCO Metrics
            coco_pred = cocoAnno.loadRes(pred)
            eval = COCOeval(cocoAnno,coco_pred,'bbox')
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            for metric in eval.stats:
                currData.append(metric)

            ## Class COCO Metrics
            for i in range(3):
                eval = COCOeval(cocoAnno, coco_pred, 'bbox')
                eval.params.catIds = [i + 1]
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                for metric in eval.stats:
                    currData.append(metric)

            ## TIDE Metrics
            tide_pred = datasets.COCOResult(pred)
            tide = TIDE()
            tide.evaluate_range(tide_gt, tide_pred, mode=TIDE.BOX)
            tide.summarize()

            ## Pull Out Errors
            errors = tide.get_all_errors()
            class_errors = tide.get_main_per_class_errors()
            class_errors = list(class_errors.values())[0]
            main_errors = list(list(errors.values())[0].values())[0]
            special_errors = list(list(errors.values())[1].values())[0]

            ## Pull Out APs
            tide_aps = list(tide.run_thresholds.values())[0]
            tide_threshs = []
            for i in range(10):
                tide_threshs.append(tide_aps[i].ap)
            tide_ap5095 = sum(tide_threshs) / len(tide_threshs)
            currData.append(tide_ap5095)
            for thresh in tide_threshs:
                currData.append(thresh)

            ## Add in Errors
            currData.append(main_errors['Cls'])
            currData.append(main_errors['Loc'])
            currData.append(main_errors['Both'])
            currData.append(main_errors['Dupe'])
            currData.append(main_errors['Bkg'])
            currData.append(main_errors['Miss'])
            currData.append(special_errors['FalsePos'])
            currData.append(special_errors['FalseNeg'])

            for i in range(3):
                for err in class_errors.keys():
                    try:
                        currData.append(class_errors[err][i + 1])
                    except:
                        currData.append(-1)

            with open(pred, "r") as f:
                raw_pred = json.load(f)

            scores = []
            for pred in raw_pred:
                scores.append(pred['score'])

            mean_score = statistics.mean(scores)
            if len(scores) == 1:
                stddev_score = -1
                stderr_score = -1
            else:
                stddev_score = statistics.stdev(scores)
                stderr_score = stddev_score / (math.sqrt(len(scores)))
            currData.append(mean_score)
            currData.append(stddev_score)
            currData.append(stderr_score)

            data.append(currData)

        ## Save out Dataframe
        if saveCSV:
            if not os.path.exists(outDir + "Results.csv"):
                with open(outDir + "Results.csv", "w", newline='') as f:
                    write = csv.writer(f)
                    write.writerows([self.header])
            with open(outDir + "Results.csv", "a", newline='') as f:
                write = csv.writer(f)
                write.writerows(data)
        return data

