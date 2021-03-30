import os

if __name__ == "__main__":
    os.makedirs("FinalRes", exist_ok=True)
    os.makedirs("FinalRes/GNNPDAnimSeq", exist_ok=True)
    os.makedirs("FinalRes/ReconstructPDAnimSeq", exist_ok=True)
    os.makedirs("FinalRes/ReconstructPNAnimSeq", exist_ok=True)
    os.makedirs("FinalRes/TrainReconstructPDAnimSeq", exist_ok=True)
    os.makedirs("PDAnimSeq", exist_ok=True)
    os.makedirs("PNAnimSeq", exist_ok=True)
    os.makedirs("RunNNRes", exist_ok=True)
    os.makedirs("TestingData", exist_ok=True)
    os.makedirs("TmpRenderedImgs", exist_ok=True)
    os.makedirs("TrainingData", exist_ok=True)
    os.makedirs("StartFrame", exist_ok=True)
