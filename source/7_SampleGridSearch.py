"""Sample script for grid search, note this differs for each study system/ annotation type!"""

import cv2
import os
import sys

import pickle
import pandas as pd

sys.path.append("Repositories/sort/")
from sort import Sort
import numpy as np
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob

from itertools import product
from joblib import Parallel, delayed
import sklearn


def GridSearch(AllVideoNames,AllBORISFiles,BehavHyperParam,iter):
    print(iter)

    HyperParam = BehavHyperParam["Eat"]
    SortTracker = Sort(max_age=HyperParam["max_age"],min_hits=HyperParam["min_hits"],iou_threshold=HyperParam["iou_threshold"])
    GT_Data = []
    Pred_Data = []


    for vidPath in tqdm(AllVideoNames):
        print(vidPath)

        DetectionDict = pickle.load(open(vidPath,"rb"))
        file = [filePath for filePath in AllBORISFiles if os.path.basename(vidPath).split(".")[0] in filePath][0]


        ### Process YOLO predictions first##
        TrackingOutList = []
        for frame,frameDict in DetectionDict.items():
            detectedClass = []
            [detectedClass.append(frameDict["bbox"][x] + [frameDict["conf"][x]])  for x in range(len(frameDict["Class"])) if frameDict["Class"][x] == "Eat" and frameDict["conf"][x] > BehavHyperParam[frameDict["Class"][x]]["YOLO_Threshold"]]

            
            ##Update SORT trackers:
            if len(detectedClass)>0:
                TrackingOut = SortTracker.update(np.array([detectedClass]).reshape(len(detectedClass),5))
            else:
                TrackingOut = SortTracker.update(np.empty((0,5)))
            if len(TrackingOut) == 0:
                TrackingOut = np.zeros((1,5))
            TrackingOutList.append(TrackingOut.tolist())

        ###Get start end time of behaviours
        ### index 4 is the track id
        UnqEvents = list(set([bbox[4] for framelist in TrackingOutList for bbox in framelist]))
        BadTracks = []
        for track in UnqEvents:
            if track ==0:
                continue
            TrackIndexes = [frame for frame in range(len(TrackingOutList)) for bbox in TrackingOutList[frame] if bbox[4] == track]
            if len(TrackIndexes) < HyperParam["min_duration"]:
                BadTracks.append(track)


        PredList = []
        for i in range(len(TrackingOutList)):
            tracks= [bbox[4] for bbox in TrackingOutList[i] if sum(bbox)>0 and bbox[4] not in BadTracks]

            PredList.append(len(tracks)) #append how many tracks
        
        
        #####Get ground truth####
        df = pd.read_csv(file)

        ###round time col to frame number
        df["RoundTime"] = df["Time"].apply(lambda x: round(x*25)) ##get frame number instead

        RoundTimeCounts = df.groupby("RoundTime").apply(lambda x: len(x["Subject"].unique()))

        GTList = [0]*len(PredList)
        for i in RoundTimeCounts.index:
            if i >= len(GTList):
                break
            GTList[i] = RoundTimeCounts.loc[i]

        ####Adjust window
        windowThresh = 2*25
        EqualChunks = [PredList[i:i + windowThresh] for i in range(0, len(PredList), windowThresh)]
        PredList = [max(x) for x in EqualChunks] #maximum number detected within this chunk

        EqualChunks = [GTList[i:i + windowThresh] for i in range(0, len(GTList), windowThresh)]
        GTList = [max(x) for x in EqualChunks] #maximum number detected within this chunk


        ### if any pecking detected within window, then matches
        for i in range(len(GTList)):
            if GTList[i]==PredList[i] == 0:
                GT_Data.append("Not Pecking")
                Pred_Data.append("Not Pecking")
                continue
            elif GTList[i] > 0 and PredList[i] > 0:
                GT_Data.append("Pecking")
                Pred_Data.append("Pecking")

            elif GTList[i] > 0 and PredList[i] == 0:
                GT_Data.append("Pecking")
                Pred_Data.append("Not Pecking")

            elif GTList[i] == 0 and PredList[i] > 0:
                GT_Data.append("Not Pecking")
                Pred_Data.append("Pecking")


    ##Metrics
    GT_Data = [str(y) for y in GT_Data]
    Pred_Data = [str(y) for y in Pred_Data]

    labels= ["Pecking","Not Pecking"]

    Precision,Recall,fbeta_score,support = sklearn.metrics.precision_recall_fscore_support(GT_Data, Pred_Data, labels=Labels, average=None, sample_weight=None, zero_division='warn')
    BehavIndex = labels.index("Pecking")

    BehavOut = {"precision":Precision[BehavIndex],"recall":Recall[BehavIndex],"f1-score":fbeta_score[BehavIndex],"support":support[BehavIndex]}

    return iter,BehavOut



def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())


if __name__ == "__main__":
    InputVideoDir = "./Data/JaySampleData/YOLO/"
    BORISDir = "./Data/JaySampleData/BORISAnnotations/"
    OutDir = "./Data/JaySampleData/GridSearchResults/"

    if not os.path.exists(OutDir):
        os.mkdir(OutDir)


    AllVideoNames = glob("%s/*.p"%InputVideoDir)
    AllBORISFiles = glob("%s/*.csv"%BORISDir)


    ###Define range of values to explore
    HyperParams = {"max_age": list(range(1,26,5)),
                    "min_hits": list(range(1,6,2)), 
                    "iou_threshold": list(np.arange(0.1,0.5,0.1)), 
                    "min_duration": list(range(1,20,2)),
                    "YOLO_Threshold":list(np.arange(0.1,0.9,0.1))}
    GridSearchDF = expand_grid(HyperParams)


    ###Loop:
    OutDict = {}
    for i in tqdm(range(len(GridSearchDF))):
        BehavHyperParam = {"Eat":GridSearchDF.iloc[0].to_dict()}

        Out= GridSearch(AllVideoNames,AllBORISFiles,BehavHyperParam,i)
        OutDict[Out[0]] = Out[1]
    ##

    ###Parallel:
    # Out = Parallel(n_jobs=10)(delayed(GridSearch)(AllVideoNames,AllBORISFiles,{"Eat":GridSearchDF.iloc[x].to_dict()},x) for x in range(len(GridSearchDF)))
    # OutDict = {x[0]:x[1] for x in Out}
    ##

    OutDF = pd.DataFrame.from_dict(OutDict, orient='index')

    FinalDF = pd.concat([GridSearchDF, OutDF], axis=1)

    FinalDF.to_csv(os.path.join(OutDir,"GridSearch_Eat.csv"))


    # ###look at results, find best params
    BestParamDict = {}
    for behav in ["Eat"]:
        df = pd.read_csv(os.path.join(OutDir, "GridSearch_%s.csv"%behav))

        MaxIndex = df["f1-score"].idxmax()
        # MaxIndex = df["recall"].idxmax()

        BestParams = df.iloc[MaxIndex].to_dict()
        BestParamDict[behav] = BestParams

    print("Best Params:")
    print(BestParamDict)




