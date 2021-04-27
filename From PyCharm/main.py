from testLoad import infer

# C:\Users\alexc\Desktop\pythonProjectCV\test
# C:\Users\alexc\Desktop\pythonProjectCV\trueValidation.pt

print("Введите полный путь до датасета (Например D:\\dataset):")
datasetPath = input()
print("Введите полный путь до весов модели:")
modelDataPath = input()

infer(modelDataPath, datasetPath)
