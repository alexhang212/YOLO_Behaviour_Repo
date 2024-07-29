"""Run Inference in a video"""


import os
from tqdm import tqdm
from natsort import natsorted
import cv2
from ultralytics import YOLO
import pandas as pd
import argparse
import pickle


def YOLOEventInference(vid,model,OutputDir,outputType = "csv"):
    """
    For given video, run YOLO and save all results and confidence into a dictionary
    
    vid: Input video path
    model: YOLO model
    OutputDir: Output directory to save results
    outputType: csv or pickle for which format to save data
    """

    EventDict = {}

    cap = cv2.VideoCapture(vid)
    counter=0

    cap.set(cv2.CAP_PROP_POS_FRAMES,counter)

    TotalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(TotalFrame), desc = "Running YOLO...."):

        ret, frame = cap.read()
        # print(counter)

        if ret == True:
            results = model(frame, verbose = False)
            ClassDict = results[0].names

            bbox = results[0].boxes.xyxy.cpu().numpy().tolist()
            conf =  results[0].boxes.conf.cpu().numpy().tolist()
            cls =  results[0].boxes.cls.cpu().numpy().tolist()
            clsList = [ClassDict[idx] for idx in cls]
            EventDict[counter] = {"Class":clsList, "conf":conf, "bbox":bbox}

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        counter += 1
    cap.release()

    if outputType == "csv":
        subcounter = 0
        DFDict = {}
        for frame in EventDict.keys():
            for x in range(len(EventDict[frame]["Class"])):
                DFDict[subcounter] = {"Frame":frame,"Behaviour":EventDict[frame]["Class"][x],"Confidence":EventDict[frame]["conf"][x],
                                      "BBox_xmin":EventDict[frame]["bbox"][x][0],
                                       "BBox_ymin":EventDict[frame]["bbox"][x][1],
                                       "BBox_xmax":EventDict[frame]["bbox"][x][2],
                                       "BBox_ymax":EventDict[frame]["bbox"][x][3]}
                subcounter += 1

        OutDict = pd.DataFrame.from_dict(DFDict,orient = "index")
        VideoName = os.path.basename(vid).split("." + vid.split(".")[-1])[0]
        OutDict.to_csv(os.path.join(OutputDir,"%s_YOLO.csv"%VideoName))

    elif outputType == "pickle":
        VideoName = os.path.basename(vid).split("." + vid.split(".")[-1])[0]
        with open(os.path.join(OutputDir,"%s_YOLO.pkl"%VideoName), 'wb') as f:
            pickle.dump(EventDict,f)
    return EventDict



def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--Video",
                        type=str,
                        help="Input Video Path")
    parser.add_argument("--Weight",
                        type=str,
                        help="Path to YOLO Weights file")
    parser.add_argument("--Output",
                        type=str,
                        help="Output type: csv, or pickle")

    arg = parser.parse_args()

    return arg


if __name__ == "__main__":
    args = ParseArgs()

    ## Custom Define arguments:
    VidPath = "Data/JaySampleData/Jay_Sample.mp4"
    WeightPath = "./Data/Weights/JayBest.pt"
    OutputType = "pickle"
    #######

    VidPath = args.Video if args.Video else VidPath
    WeightPath = args.Weight if args.Weight else WeightPath
    OutputType = args.Output if args.Output else OutputType


    ##
    model = YOLO(WeightPath)
    OutputDir = os.path.dirname(VidPath) ##Can specify output directory here, default same dir as video
    print(OutputType)
    YOLOEventInference(VidPath,model,OutputDir,outputType = OutputType)

