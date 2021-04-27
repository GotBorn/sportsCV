import csv
import shutil
import os

test = {
    "badminton": 104,
    "baseball": 83,
    "formula1": 80,
    "fencing": 74,
    "motogp": 76,
    "ice_hockey": 83,
    "wrestling": 64,
    "boxing": 81,
    "volleyball": 84,
    "cricket": 75,
    "basketball": 56,
    "wwe": 76,
    "swimming": 79,
    "weight_lifting": 67,
    "gymnastics": 79,
    "tennis": 83,
    "kabaddi": 53,
    "football": 91,
    "table_tennis": 79,
    "hockey": 60,
    "shooting": 62,
    "chess": 56
}
validation = {
    "badminton": 104,
    "baseball": 83,
    "formula1": 80,
    "fencing": 74,
    "motogp": 76,
    "ice_hockey": 83,
    "wrestling": 64,
    "boxing": 81,
    "volleyball": 84,
    "cricket": 75,
    "basketball": 56,
    "wwe": 76,
    "swimming": 79,
    "weight_lifting": 67,
    "gymnastics": 79,
    "tennis": 83,
    "kabaddi": 53,
    "football": 91,
    "table_tennis": 79,
    "hockey": 60,
    "shooting": 62,
    "chess": 56
}

outputDir = "D:\\train\\"
testDir = "D:\\test\\"
validationDir = "D:\\valid\\"
totalTestMoved = 0
totalValMoved = 0

with open("train_labels.csv", encoding='utf-8') as inputFile:
    inputReader = csv.reader(inputFile, delimiter=",")
    for row in inputReader:
        if row[0] == "image":  # скипаем первую строчку
            continue
        currentFileName = row[0].replace("./train/", "")
        print(currentFileName)
        currentCategory = row[1]
        print(currentCategory)
        if validation[currentCategory] > 0:
            shutil.copy(outputDir + currentFileName, validationDir)
            os.remove(outputDir + currentFileName)
            validation[currentCategory] = validation[currentCategory] - 1
            totalValMoved = totalValMoved + 1
        elif test[currentCategory] > 0:
            shutil.copy(outputDir + currentFileName, testDir)
            os.remove(outputDir + currentFileName)
            test[currentCategory] = test[currentCategory] - 1
            totalTestMoved = totalTestMoved + 1
    print("totalValMoved ", totalValMoved)
    print("totalTestMoved ", totalTestMoved)
