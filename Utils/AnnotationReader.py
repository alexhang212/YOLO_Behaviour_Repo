"""Object to Read JSON output from label-studio"""

import sys
import os
sys.path.append("./")
import json
import cv2
import numpy as np

def getColor(keyPoint):
    if keyPoint.endswith("beak"):
        return (255, 0 , 0 )
    elif keyPoint.endswith("nose"):
        return (63,133,205)
    elif keyPoint.endswith("leftEye"):
        return (0,255,0)
    elif keyPoint.endswith("rightEye"):
        return (0,0,255)
    elif keyPoint.endswith("leftShoulder"):
        return (255,255,0)
    elif keyPoint.endswith("rightShoulder"):
        return (255, 0 , 255)
    elif keyPoint.endswith("topKeel"):
        return (128,0,128)
    elif keyPoint.endswith("bottomKeel"):
        return (203,192,255)
    elif keyPoint.endswith("tail"):
        return (0, 255, 255)
    else:
        # return (0,165,255)
        return (0,255,0)

def IsPointInBox(bbox, point):
    """Check if a point is within bounding box, i.e within image frame"""
    #get dimension of screen from config
    Valid = False
    if bbox[0] <= point[0] <= bbox[0]+bbox[2] and bbox[1] <= point[1] <= bbox[1]+bbox[3]:
        Valid = True
    else:
        return Valid
    return Valid

class LS_JSONReader:
    
    def __init__(self, DatasetPath,JSONPath):
        """
        Initialize JSON reader object
        JSONPath: path to json file
        DatasetPath: Path to dataset root directory to read images
        Type: 2D or 3D, based om which type was read     
        """
        with open(JSONPath) as f:
            self.data = json.load(f)

        self.DatasetPath = DatasetPath

    def ExtractBBox(self,index):
        """Extract bbox, xyxy format"""
        if "label" not in self.data[index]:####no annotation, return empty dict
            return {}

        BBoxData = self.data[index]["label"]

        BBoxDict = {}
        for x in range(len(BBoxData)):
            XRatio = BBoxData[x]["original_width"]/100
            YRatio = BBoxData[x]["original_height"]/100

            bbox = [BBoxData[x]["x"]*XRatio,BBoxData[x]["y"]*YRatio,
                    BBoxData[x]["x"]*XRatio+BBoxData[x]["width"]*XRatio,BBoxData[x]["y"]*YRatio+BBoxData[x]["height"]*YRatio]
            BBoxDict.update({BBoxData[x]["rectanglelabels"][0]:bbox})
        
        return BBoxDict
    
    def GetAllClasses(self):
        """Get all unique classes in the dataset"""
        AllClasses = []
        for i in range(len(self.data)):
            if "label" not in self.data[i]:
                continue
            BBoxData = self.data[i]["label"]

            for x in range(len(BBoxData)):
                AllClasses.append(BBoxData[x]["rectanglelabels"][0])
        return list(set(AllClasses))

    def GetImagePath(self,index):
        return self.data[index]["image"][1:] #remove first slash
    
    def CheckAnnotations(self, index, show=True):
        ImgPath = self.GetImagePath(index)
        BBox = self.ExtractBBox(index)
        
        RealImgPath = os.path.join(self.DatasetPath,ImgPath) #removes first slash in ImgPath

        img = cv2.imread(RealImgPath)
        
        ##Draw keypoints:
        for label,BBoxData in BBox.items():

            cv2.rectangle(img,(round(BBoxData[0]),round(BBoxData[1])),(round(BBoxData[2]),round(BBoxData[3])),(255,0,0),3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, label,(round(BBoxData[0]),round(BBoxData[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
        if show:
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img



if __name__ == "__main__":
    DatasetPath = "./Data/"
    JSONPath = "./Data/JayAnnotations.json"


    Reader = LS_JSONReader(DatasetPath,JSONPath)


    for i in range(len(Reader.data)):
        print(i)
        Reader.CheckAnnotations(i)