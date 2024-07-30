"""Train YOLO"""

from ultralytics import YOLO
import argparse


def TrainYOLO(model,ConfigPath,Epochs):
    model.train(data=ConfigPath, batch=-1 ,epochs=Epochs)  # train the model
    model.val()

            

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--Model",
                        type=str,
                        help="YOLO Model type varies with size, options: n (nano), s (small), m (medium), l (large), x (extra large)")
    parser.add_argument("--Config",
                        type=str,
                        help="Path to YOLO Config file")
    parser.add_argument("--Epochs",
                        type=int,
                        help="Total number of epochs to train")

    arg = parser.parse_args()

    return arg


if __name__ == "__main__":
    args = ParseArgs()

    ### Manual define arguments:
    modelType = "n"
    ConfigPath = "./Data/YOLO_Datasets/Jays_YOLO.yaml"
    Epochs = 200
    #####

    modelType = args.Model if args.Model else modelType
    ConfigPath = args.Config if args.Config else ConfigPath
    Epochs = args.Epochs if args.Epochs else Epochs

    model = YOLO("./yolov8%s.pt"%modelType) ##Can also update to newer models if new ones came out!
    TrainYOLO(model,ConfigPath,Epochs)