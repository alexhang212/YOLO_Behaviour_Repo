"""Inference and Visualize Results"""
import cv2 
from ultralytics import YOLO
import torch
import argparse



def RunInference(model, InputVideo,StartFrame=0,TotalFrames=-1):
    if InputVideo == "webcam":
        cap = cv2.VideoCapture(0)
        imsize = (int(cap.get(3)),int(cap.get(4)))
        counter = 0

    else:
        cap = cv2.VideoCapture(InputVideo)
        cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
        imsize = (int(cap.get(3)),int(cap.get(4)))
        counter=StartFrame
        cap.set(cv2.CAP_PROP_POS_FRAMES,counter)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if TotalFrames == -1:
        TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter("YOLO_Sample.mp4", cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize = imsize)

    for i in range(TotalFrames):

        ret, frame = cap.read()

        if ret == True:
            InferFrame = frame.copy()
            results = model(InferFrame)
            frame = results[0].plot()

            cv2.imshow('Frame',frame)
            out.write(frame)
            counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    out.release
    

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--Video",
                        type=str,
                        help="Input Video Path or 'webcam' for webcam")
    parser.add_argument("--Weight",
                        type=str,
                        help="Path to YOLO Weights file")
    parser.add_argument("--Start",
                        type=int,
                        help="Start Frame")
    parser.add_argument("--Frames",
                        type=int,
                        help="Total number of frames, -1 for whole video")

    arg = parser.parse_args()

    return arg


if __name__ == "__main__":
    args = ParseArgs()

    ## Custom Define arguments:
    VidPath = "./Data/JaySampleData/Jay_Sample.mp4"
    WeightPath = "./Data/Weights/JayBest.pt"
    StartFrame = 0
    TotalFrames = -1

    ########
    VidPath = args.Video if args.Video else VidPath
    WeightPath = args.Weight if args.Weight else WeightPath
    StartFrame = args.Start if args.Start else StartFrame
    TotalFrames = args.Frames if args.Frames else TotalFrames

    model = YOLO(WeightPath)
    
    RunInference(model, VidPath,StartFrame=StartFrame, TotalFrames = TotalFrames)

    