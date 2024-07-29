"""Sample script for human in the loop, first prepare videos then manually review"""

import cv2
import os
import sys

import pickle
import pandas as pd

sys.path.append("Repositories/sort/")
from sort import Sort

import numpy as np
from natsort import natsorted
from tqdm import tqdm
import numpy as np

def PrepareVideos(InputVideo,DetectionDict,SubVideoDir,BehavHyperParam):
    """First get events start stop"""
    Classes = BehavHyperParam.keys()

    SortTrackerDict = {Behav:Sort(max_age=BehavHyperParam[Behav]["max_age"],
                                  min_hits=BehavHyperParam[Behav]["min_hits"],
                                  iou_threshold=BehavHyperParam[Behav]["iou_threshold"]) for Behav in Classes}

    TrackingOutDict = {Behav:[] for Behav in Classes}
    for frame,frameDict in DetectionDict.items():
        detectedClass = {Behav:[] for Behav in SortTrackerDict.keys()}
        [detectedClass[frameDict["Class"][x]].append(frameDict["bbox"][x] + [frameDict["conf"][x]])  for x in range(len(frameDict["Class"])) if frameDict["conf"][x] > BehavHyperParam[frameDict["Class"][x]]["YOLO_Threshold"]]

        
        ##Update SORT trackers:
        for Behav in SortTrackerDict.keys():

            if len(detectedClass[Behav])>0:
                TrackingOut = SortTrackerDict[Behav].update(np.array([detectedClass[Behav]]).reshape(len(detectedClass[Behav]),5))
            else:
                TrackingOut = SortTrackerDict[Behav].update(np.empty((0,5)))
            if len(TrackingOut) == 0:
                TrackingOut = np.zeros((1,5))
            TrackingOutDict[Behav].append(TrackingOut.tolist())

    ###Get start end time of behaviours
    ### index 4 is the track id
    StartStopFrameDict = {Behav:{} for Behav in Classes}

    for Behav in Classes:
        UnqEvents = list(set([bbox[4] for framelist in TrackingOutDict[Behav] for bbox in framelist]))
        BadTracks = []
        for track in UnqEvents:
            if track ==0:
                continue
            TrackIndexes = [frame for frame in range(len(TrackingOutDict[Behav])) for bbox in TrackingOutDict[Behav][frame] if bbox[4] == track]
            if len(TrackIndexes) < BehavHyperParam[Behav]["min_duration"]:
                BadTracks.append(track)
            else:
                StartStopFrameDict[Behav][track] = (min(TrackIndexes),max(TrackIndexes))


    ###Write Videos
    cap = cv2.VideoCapture(InputVideo)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    imsize = (int(cap.get(3)),int(cap.get(4)))

    for Behav in Classes:

        for track, (start,stop) in tqdm(StartStopFrameDict[Behav].items(), desc = "Writing Videos...."):
            out = cv2.VideoWriter(os.path.join(SubVideoDir,"%s_Track_%s.mp4"%(Behav,track)), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize = imsize)
            BBoxData = {frame:box for frame in range(start,stop) for box in TrackingOutDict[Behav][frame] if box[4] == track}

            cap = cv2.VideoCapture(InputVideo)
            cap.set(cv2.CAP_PROP_POS_FRAMES,start)

            for i in range(start,stop+1):
                ret, frame = cap.read()

                if i in BBoxData:

                    frame = cv2.rectangle(frame, (int(BBoxData[i][0]),int(BBoxData[i][1])),(int(BBoxData[i][2]),int(BBoxData[i][3])), (255, 0, 0), 2)
                    frame = cv2.putText(frame, Behav, (int(BBoxData[i][0])-10,int(BBoxData[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                out.write(frame)

            out.release()
            cap.release()

    ###OutputDF
    OutDFDict = {}
    counter=0
    for Behav in Classes:
        for track, (start,stop) in StartStopFrameDict[Behav].items():
            OutDFDict[counter] = {"Behaviour":Behav,"Track":track,"StartFrame":start,"EndFrame":stop}
            counter += 1
    
    OutDF = pd.DataFrame.from_dict(OutDFDict,orient = "index")
    OutDF.to_csv(os.path.join(SubVideoDir,"TracksInfo.csv"))


def ManualReview(SubVideoDir):

    TracksDF = pd.read_csv(os.path.join(SubVideoDir,"TracksInfo.csv"))
    AllSubVideos = natsorted([os.path.join(SubVideoDir,vid) for vid in os.listdir(SubVideoDir) if ".mp4" in vid])
    cv2.namedWindow("window",cv2.WINDOW_NORMAL)
    ManualReviewList = []


    for idx in range(len(AllSubVideos)):
        while True:
            cap = cv2.VideoCapture(AllSubVideos[idx])
            EndVideo = 0

            while True:
                ret, frame = cap.read()
                frame = cv2.putText(frame, "Approve (y) or Disapprove (n)", (20,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                frame = cv2.putText(frame, "Track Num: %s"%idx, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                if ret == True:
                    cv2.imshow("window",frame)
                    if cv2.waitKey(1) & 0xFF == ord('y'):
                        ManualReviewList.append(True)
                        EndVideo = 1
                        break
                    elif cv2.waitKey(1) & 0xFF == ord('n'):
                        ManualReviewList.append(False)
                        EndVideo = 1
                        break
                else:
                    break
            if EndVideo == 1:
                break

    TracksDF["ManualReview"] = ManualReviewList

    TracksDF.to_csv(os.path.join(SubVideoDir,"TracksInfo.csv"))



        

if __name__ == "__main__":
    InputVideo = "./Data/JaySampleData/Jay_Sample.mp4"
    InferencePickle = "./Data/JaySampleData/Jay_Sample_YOLO.pkl"

    VideoName = os.path.basename(InputVideo).split(".")[0]

    ###Output directory
    SubVideoDir = "./Data/JaySampleData/%s_VideoChunks/"%VideoName
    if not os.path.exists(SubVideoDir):
        os.mkdir(SubVideoDir)


    DetectionDict = pickle.load(open(InferencePickle,"rb"))

    BehavHyperParam = {'Eat': {'max_age': 21.0, 'min_hits': 1.0, 'iou_threshold': 0.2, 'min_duration': 1.0, 'YOLO_Threshold': 0.1}}
    

    # PrepareVideos(InputVideo,DetectionDict,SubVideoDir,BehavHyperParam)

    ManualReview(SubVideoDir)

