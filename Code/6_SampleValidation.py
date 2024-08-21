"""Sample script for validating the model, note this differs for each study system/ annotation type!"""

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
from sklearn.metrics import classification_report,cohen_kappa_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json



from glob import glob

def EventValidation(AllVideoNames,AllBORISFiles,BehavHyperParam):

    HyperParam = BehavHyperParam["Eat"]
    SortTracker = Sort(max_age=HyperParam["max_age"],min_hits=HyperParam["min_hits"],iou_threshold=HyperParam["iou_threshold"])
    GT_Data = []
    Pred_Data = []


    for vidPath in tqdm(AllVideoNames):
        print(vidPath)

        DetectionDict = pickle.load(open(vidPath,"rb"))
        file = [filePath for filePath in AllBORISFiles if os.path.basename(vidPath).split(".")[0] in filePath][0]


        ### Part 1: Process YOLO predictions first##
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
        
        
        #####Part 2: Get ground truth####
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

    SumCorrect = [1 for i in range(len(GT_Data)) if GT_Data[i] == Pred_Data[i]]
    LabelWithData = [1 for i in range(len(GT_Data)) if GT_Data[i] != "nan"]
    print(len(LabelWithData))

    Accuracy = sum(SumCorrect)/len(GT_Data)
    print("Overall Accuracy:%s"%Accuracy)

    labels= ["Pecking","Not Pecking"]

    cm = confusion_matrix(GT_Data, Pred_Data, labels=labels,normalize="true")
    

    labels = ["Eating","Not Eating"]
    colour = sns.cubehelix_palette(as_cmap=True)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, xticklabels=labels, 
                yticklabels=labels,cbar=False, cmap=colour,
                annot_kws={"fontsize":16})


    plt.xlabel("Predicted",fontsize=20)
    plt.ylabel("Manual", fontsize=20)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.tight_layout()
    plt.show()    

    OutReport = classification_report(GT_Data, Pred_Data, output_dict=False)

    cohen_kappa = cohen_kappa_score(GT_Data, Pred_Data)

    # MCC = matthews_corrcoef(GT_Data, Pred_Data)


    return OutReport,cohen_kappa


if __name__ == "__main__":
    InputVideoDir = "./Data/JaySampleData/YOLO/"
    BORISDir = "./Data/JaySampleData/BORISAnnotations/"
    ParamFile = "./Data/JaySampleData/Jay_Sample_HyperParameters.json"


    AllVideoNames = glob("%s/*.p"%InputVideoDir)
    AllBORISFiles = glob("%s/*.csv"%BORISDir)

    ###Hyper parameters:
    BehavHyperParam = json.load(open(ParamFile,"r"))

    OutReport,cohen_kappa = EventValidation(AllVideoNames,AllBORISFiles,BehavHyperParam)

    print(OutReport)
    print("Cohen Kappa: %s"%cohen_kappa)

    




