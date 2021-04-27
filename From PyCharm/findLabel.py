import csv


def FindLabel(fileName):
    with open("train_labels.csv", encoding='utf-8') as inputFile:
        inputReader = csv.reader(inputFile, delimiter=",")
        for row in inputReader:
            if row[0] == "image":  # скипаем первую строчку
                continue
            csvFileName = row[0].replace("./train/", "")
            if csvFileName == fileName:
                return row[1]
    return 0

