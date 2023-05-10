from __future__ import absolute_import

import argparse
import torch
import torchvision
from lp_utils.lp_realtime import keypointOnCam
from lp_config.lp_common_config import config
from lp_testing.lp_test import test
from lp_training.lp_trainer import train
from lp_model.lp_litepose import LitePose
from lp_coco_utils.lp_getDataset import getDatasetProcessed
from lp_inference.lp_inference import inference, assocEmbedding
from lp_testing.lp_evaluate import evaluateModel
import cv2
from lp_utils.lp_image_processing import drawHeatmap, drawKeypoints, drawSkeleton

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--train',
                        help='train the network on Coco dataset',
                        action='store_true')

    parser.add_argument('--resnet-live', 
                        help="use keypoint Resnet50 pretrained model as live detector",
                        action="store_true")

    parser.add_argument('--test', 
                        help="just for testing purposes",
                        action="store_true")

    parser.add_argument('--inference', 
                        help="perform inference and shows the results for the provided model")
    
    parser.add_argument('--score', 
                        help="Comptue OKS score for the provided model")

    args = parser.parse_args()

    return args

def handleResnetLive():
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, 
                                        weights= torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()
    model.to(config["device"])
    keypointOnCam(model, "~/Videos/r1.mp4v")

def handleTrain():
    train(config["batch_size"])

def handleTest():
    test()

def handleScore(args):
    model = LitePose().to(config["device"])
    model.load_state_dict(torch.load(args.score, map_location=config["device"]))
    res = evaluateModel(model)
    print(f"Object Keypoint Similarity (OKS) score: {res}")

def handleInference(args):
    model = LitePose().to(config["device"])
    model.load_state_dict(torch.load(args.inference, map_location=config["device"]))

    ds = getDatasetProcessed("validation")

    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=64
    )

    row = next(iter(data_loader))
    images = row[0].to(config["device"])

    groundTruthHeatmap = row[1]
    output, keypoints = inference(model, images)
    jointsHeatmap = output[1][2][:config["num_joints"]]

    img, finalHeatmap, superImposed = drawHeatmap(images[2], jointsHeatmap)
    img, groundTruthFinalHeatmap, groundTruthSuperImposed = drawHeatmap(images[2], groundTruthHeatmap[1][2])
    
    # heatmap
    cv2.imshow("Image", img)
    cv2.imshow("Final heatmap", finalHeatmap)
    cv2.imshow("Super imposed", superImposed)
    cv2.imshow("Ground Truth heatmap", groundTruthFinalHeatmap)
    cv2.imshow("Ground Truth Superimposed", groundTruthSuperImposed)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # keypoints
    idx = 7
    img = drawKeypoints(images[idx], keypoints[idx])
    img = cv2.resize(img, (600, 600))
    cv2.namedWindow("Image Keypoints", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Keypoints", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Skeleton
    idx = 7
    embedding = assocEmbedding(keypoints)
    img = drawSkeleton(images[idx], embedding[idx])
    img = cv2.resize(img, (600, 600))
    cv2.namedWindow("Image Skeleton", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Skeleton", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    if(args.train):
        handleTrain()
    elif(args.resnet_live):
        handleResnetLive()
    elif(args.test):
        handleTest()
    elif(args.inference):
        handleInference(args)
    elif(args.score):
        handleScore(args)
    
if __name__ == '__main__':
    main()