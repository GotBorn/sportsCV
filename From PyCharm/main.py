import os

from testLoad import infer

# C:\Users\alexc\Desktop\pythonProjectCV\test
# C:\Users\alexc\Desktop\trueValidation.pt

print("Введите полный путь до папки с весами всех 4-х моделей:")
modelFolderDataPath = input()
print("Введите полный путь до датасета (Например D:\\dataset):")
datasetPath = input()

infer(modelFolderDataPath, datasetPath)

print("Результат находится в output.csv")