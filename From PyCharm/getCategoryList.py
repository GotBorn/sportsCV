import csv


def GetCategoryList():
    labelList = []

    with open("train_labels.csv", encoding='utf-8') as inputFile:
        inputReader = csv.reader(inputFile, delimiter=",")
        for row in inputReader:
            if row[0] == "image":  # скипаем первую строчку
                continue
            if labelList.count(row[1]) == 0:
                labelList.append(row[1])

    return labelList


testList = GetCategoryList()

print(f'{len(testList)} уникальных элементов')
print(testList)
