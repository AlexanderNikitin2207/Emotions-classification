# Emotions-classification
Телеграм-бот, распознающий эмоции по изображению. Бот умеет распознавать 8 эмоций: anger, surprise, contempt, happy, neutral, fear, sad, disgust. 
## src
* start.py - телеграм-бот
* emotions_classification.ipynb - ноутбук с обучением модели
## Стек используемых технологий
1. [Python](https://www.python.org/)
2. [Wandb](https://wandb.ai/site)
3. [🤗 Transformers](https://huggingface.co/docs/transformers/index)
4. [Telebot](https://github.com/eternnoir/pyTelegramBotAPI?ysclid=lhpd9hofgk485400689)
## Датасет
Датасет для обучения представляет собой часть набора данных [AffectNet](https://paperswithcode.com/dataset/affectnet) и состоит из 29 тысяч изображений. Ссылка на датасет: https://huggingface.co/datasets/Mauregato/affectnet_short/viewer/Mauregato--affectnet_short/train
## Модель
В качестве нейронной модели был выбран предобученный Vision Transformer, взятый отсюда: https://huggingface.co/google/vit-base-patch16-224. Модель была дообучена на используемом датасете, были подобраны гиперпараметры с помощью wandb. Ссылка на модель: https://huggingface.co/Mauregato/vit-base-patch16-224-finetuned-on-all-affectnet_short
### Метрики
На тестовом множестве, состоящем из 5800 изображений, модель показала следующие результаты:
* Accuracy: 0.7259
* Precision: 0.7293
* Recall: 0.7259
* F1: 0.7255
