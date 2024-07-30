"""Given a video input, sample random frames from it"""

import os
import sys
import cv2
from tqdm import tqdm
import argparse

import random
random.seed(534202)

def SampleImages(InputVideo,OutDir, NumFrames):
    """Given input video, sample random frames
    
    """
    vidBaseName =  os.path.basename(InputVideo).split(".")[0]
    cap = cv2.VideoCapture(InputVideo)
    TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FrameNums = list(range(TotalFrames))

    RandomFrames = random.sample(FrameNums,NumFrames)

    for i in tqdm(range(NumFrames)):
        ret,frame = cap.read()

        if ret == False:
            print("End of video.")
            break

        if i in RandomFrames:
            SaveFile = os.path.join(OutDir,"%s_F%s.jpg"%(vidBaseName,i))
            cv2.imwrite(filename=SaveFile,img=frame)

    cap.release()


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--Input",
                        type=str,
                        help="Path to input video")
    parser.add_argument("--Output",
                        type=str,
                        help="Path to output directory")
    parser.add_argument("--Frames",
                        type=int,
                        help="TotalNumber of frames to sample")

    arg = parser.parse_args()

    return arg


if __name__ == "__main__":
    InputVideo = "./Data/JaySampleData/Jay_Sample.mp4"
    OutDir = "./Data/JaySampleData/RandomFrames"
    NumFrames = 100

    if not os.path.exists(OutDir):
        os.makedirs(OutDir)
    SampleImages(InputVideo,OutDir)