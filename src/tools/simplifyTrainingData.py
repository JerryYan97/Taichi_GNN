import sys, os, time

# datapath = "TrainingData"
datapath = "./tmpFullData"

if __name__ == "__main__":
    # Read files
    Results_Files = []
    for _, _, files in os.walk(datapath):
        Results_Files.extend(files)
    Results_Files.sort()
    print("Results_Files:", Results_Files)

    option = int(input("Culling max frames (0) / Culling frame percentage (1):"))
    if option == 0:
        maxFrame = int(input("Please specify the max number of frames that you want:"))
    if option == 1:
        cullingMultiply = int(input("Please specify the multiply that you want to keep:"))

    for f in range(len(Results_Files)):
        file_index_str = Results_Files[f][Results_Files[f].rfind('_')+1:Results_Files[f].find('csv')-1]
        f_int = int(file_index_str)
        if option == 0:
            if f_int > maxFrame:
                os.remove(datapath + "/" + Results_Files[f])
        if option == 1:
            if f_int % cullingMultiply != 0:
                os.remove(datapath + "/" + Results_Files[f])
