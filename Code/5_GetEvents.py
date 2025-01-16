"""Given YOLO detections, associate detections together to create behavioural events"""

import sys

import pickle
import pandas as pd

sys.path.append("Repositories/sort/")
from sort import Sort
import numpy as np
from tqdm import tqdm
import numpy as np
import os
from glob import glob

import json

import argparse


def CSVtoDict(CSVFile):
    """Convert csv output back to dictionary"""
    DF = pd.read_csv(CSVFile)
    
    MaxFrame = DF["Frame"].max()
    
    OutDict = {}
    
    for i in range(MaxFrame):
        FrameDict = {"Class":[],"conf":[],"bbox":[]}
        
        if i not in DF["Frame"].values:
            OutDict[i] = FrameDict
            continue
    
        SubDF = DF[DF["Frame"]==i]

        for x in range(len(SubDF)):
            FrameDict["Class"].append(SubDF["Behaviour"].values[x])
            FrameDict["conf"].append(SubDF["Confidence"].values[x])
            FrameDict["bbox"].append([SubDF["BBox_xmin"].values[x],SubDF["BBox_ymin"].values[x],SubDF["BBox_xmax"].values[x],SubDF["BBox_ymax"].values[x]])

        OutDict[i] = FrameDict
    
    return OutDict
    


def GetEvents(Detections,BehavHyperParam):

    counter = 0
    OutDict = {}


    for behav in BehavHyperParam.keys():


        HyperParam = BehavHyperParam[behav]
        SortTracker = Sort(max_age=HyperParam["max_age"],min_hits=HyperParam["min_hits"],iou_threshold=HyperParam["iou_threshold"])

        if Detections.endswith(".csv"):
            DetectionDict = CSVtoDict(Detections)
        else:
            DetectionDict = pickle.load(open(Detections,"rb"))

        TrackingOutList = []
        ### Process YOLO predictions ###
        for frame,frameDict in tqdm(DetectionDict.items(), desc="Processing %s"%behav):
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
        for track in UnqEvents:
            if track ==0:
                continue
            TrackIndexes = [frame for frame in range(len(TrackingOutList)) for bbox in TrackingOutList[frame] if bbox[4] == track]
            if len(TrackIndexes) < HyperParam["min_duration"]:
                continue

            StartFrame, EndFrame = TrackIndexes[0],TrackIndexes[-1]

            OutDict[counter] = {"Behaviour": behav, "StartFrame":StartFrame, "EndFrame":EndFrame}
            counter += 1
        
    OutDF = pd.DataFrame.from_dict(OutDict,orient="index")
    print("Done!")

    return OutDF

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--Detection",
                        type=str,
                        help="Path to the detections pickle file")
    parser.add_argument("--Param",
                        type=str,
                        help="Path to the Hyperparameter json file")

    arg = parser.parse_args()

    return arg


if __name__ == "__main__":

    args = ParseArgs()

    ## Custom Define arguments:
    Detections = "./Data/JaySampleData/Jay_Sample_YOLO.pkl"
    
    ParamFile = "./Data/JaySampleData/Jay_Sample_HyperParam.json"

    #######

    Detections = args.Detection if args.Detection else Detections
    ParamFile = args.Param if args.Param else ParamFile


    BehavHyperParam = json.load(open(ParamFile,"r"))
    
    ##Sample BehavHyperParam Dictionary:
    # BehavHyperParam = {"Eat": {
    #         "max_age": 21.0,
    #         "min_hits": 1.0,
    #         "iou_threshold": 0.2,
    #         "min_duration": 1.0,
    #         "YOLO_Threshold": 0.1
    #     }
    # }
    


    OutDF = GetEvents(Detections,BehavHyperParam)
    OutputDir = os.path.dirname(Detections) ##Can specify output directory here, default same dir as detection pickle
    VidName = os.path.basename(Detections).split("_YOLO")[0]
 
    OutDF.to_csv(os.path.join(OutputDir,"%s_OutputEvents.csv"%(VidName)), index=False)
    

    

