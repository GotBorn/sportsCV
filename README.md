# sportsCV
Image classification task

Тестовая точность: 84,7% (91,1% если при тесте убрать все аугментации, кроме нормализации и ресайза)

В папке содержатся файлы для запуска непосредственно с компьютера, путь нужно указывать полный

* Для пропорционального разбиения на тестовую, валидационную и тренирующию выборки был использован скрипт createDataset.py
* Функция findLabel позволяет находить лейбл файла по его имени в csv файле, используется во время тренировки
* Функция getCategoryList используется для получения списка всех возможных лейблов в датасете, используется во время тренировки
* Для повышения точности использован ансамбль из 4-х моделей: Resnext, GoogleNet, DenseNet, resnet152
* Анализ датасета находится в файле рапределение.xlsx
* Некоторые общие заметки написаны в файле статистика.xlsx

Ссылка на папку с весами для модели:
https://drive.google.com/drive/folders/10a7mj12lmyNy0nZaEtPMdiJXKLlUlvD5?usp=sharing

При тестировании вводить путь как, например: "/content/drive/MyDrive/models"
