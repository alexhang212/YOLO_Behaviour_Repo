"""From JSON dataset, convert to YOLO format"""

import numpy as np

import sys
sys.path.append("./")
sys.path.append("Utils")
from Utils import AnnotationReader
import os
import shutil
from tqdm import tqdm
import cv2
import argparse


def ConvertBBox(bbox,imsize):
    """convert from top left/ bot right format to mid point width height format"""
    MidPoint = [(bbox[2]+bbox[0])/2,(bbox[3]+bbox[1])/2 ]
    Width = bbox[2]-bbox[0]
    Height = bbox[3]-bbox[1]

    return [MidPoint[0]/imsize[0],MidPoint[1]/imsize[1],Width/imsize[0],Height/imsize[1] ]


def ConvertJSON_YOLO(Dataset,OutPath, IndexList,Classes):
    """Main function to convert JSON format for yolo"""
    
    ImgDir = os.path.join(OutPath,"images")
    LabelDir = os.path.join(OutPath, "labels")
    
    if not os.path.exists(OutPath):
        os.mkdir(OutPath)

    if not os.path.exists(ImgDir):
        os.mkdir(ImgDir)
        os.mkdir(LabelDir)


    for i in tqdm(IndexList, desc= "Converting to YOLO format...."):
        ImgPath = Dataset.GetImagePath(i)
        # import ipdb;ipdb.set_trace()
        BaseName = os.path.basename(ImgPath).split(".jpg")[0]
        
        shutil.copy(os.path.join(Dataset.DatasetPath,ImgPath),ImgDir)

        img = cv2.imread(os.path.join(Dataset.DatasetPath,ImgPath))
        imsize = (img.shape[1],img.shape[0])
        
        ##Write label:
        BBoxData = Dataset.ExtractBBox(i)
        
        with open(os.path.join(LabelDir, "%s.txt"%BaseName), 'w') as f:
            if len(BBoxData) == 0: #no annotaiton, negative example
                continue

            for k,v in BBoxData.items():

                if k in Classes:
                    Class = Classes.index(k)
                else:
                    continue
                
                BBoxList = ConvertBBox(v,imsize)
                f.write("%s %s %s %s %s\n"%(Class,BBoxList[0],BBoxList[1],BBoxList[2],BBoxList[3]))
                

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--Dataset",
                        type=str,
                        help="Path to dataset folder, where images are stored")
    parser.add_argument("--JSON",
                        type=str,
                        help="Path to annotation JSON file")
    parser.add_argument("--Output",
                        type=str,
                        help="Output Directory to save YOLO dataset")

    arg = parser.parse_args()

    return arg


if __name__ == "__main__":
    args = ParseArgs()

    ### Manual define paths:
    DatasetPath = "./Data/LabelStudio"
    JSONPath = "./Data/LabelStudio/JayAnnotations.json"
    OutDir = "./Data/YOLO_Datasets/Jay2"

    ####################################


    ####Get arg parse
    DatasetPath = args.Dataset if args.Dataset else DatasetPath
    JSONPath = args.JSON if args.JSON else JSONPath
    OutDir = args.Output if args.Output else OutDir


    if not os.path.exists(OutDir):
        os.mkdir(OutDir)


    Dataset = AnnotationReader.LS_JSONReader(DatasetPath,JSONPath)
    Classes = Dataset.GetAllClasses()


    TotalData = len(Dataset.data)
    TotalIndex = list(range(TotalData))
    import random
    random.seed(30624700)
    random.shuffle(TotalIndex)

    TrainSet = round(TotalData *0.7)
    ValSet = round(TotalData *0.2)
    TestSet = round(TotalData*0.1)

    TrainIndex = TotalIndex[0:TrainSet]
    ValIndex = TotalIndex[TrainSet:TrainSet+ValSet]
    TestIndex = TotalIndex[TrainSet+ValSet:TotalData+1]
    
    OutPath = os.path.join(OutDir,"train")
    ConvertJSON_YOLO(Dataset,OutPath, IndexList = TrainIndex,Classes=Classes)
    OutPath = os.path.join(OutDir,"val")
    ConvertJSON_YOLO(Dataset,OutPath, IndexList = ValIndex,Classes=Classes)
    OutPath = os.path.join(OutDir,"test")
    ConvertJSON_YOLO(Dataset,OutPath, IndexList = TestIndex,Classes=Classes)
