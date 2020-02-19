# Bengali.AI Handwritten Grapheme Classification
- [대회 주소](https://www.kaggle.com/c/bengaliai-cv19)


### 대회의 흐름
```data load``` -> ```resize & image processing``` -> ```reshape``` -> ```model load```-> ```predict```

### 사용해본 모델 [이미지 분류 모델 모음](https://github.com/qubvel/classification_models.git)
#### se-resnext50


### 겪은 문제점과 해결과정
- kaggle에서 제공된 데이터 파일 형식인 ```parquet```은 load시 오랜시간이 걸림. 이는 ```feather```라는 파일 형식으로 데이터를 구성하여 load 하였더니 한 파일당 1분 걸리던 것이 2초만에 load되었음(총4개의 train data 즉, 4분걸리던것이 10초만에 load됨) [<관련커널>](https://www.kaggle.com/corochann/bangali-ai-super-fast-data-loading-with-feather)
```
import pandas as pd
df = pd.read_parquet('train.parquet')
df.to_feather('train_feather.feather', index=False)
df_feather = pd.read_feather('train_feather.feather')
```

- 이미지 데이터 특성상 데이터의 크기가 너무 커서 전처리(resize, image processing, reshape)를 할 때 문제가 되었음. 이는 python 의 garbage collection을 통해 해결하였음.
```
import gc
del a
gc.collect()
```
- 커널 대회이다 보니 제출을 하면 숨겨진 test data를 커널 흐름에 따라 전처리되고 추론이 되는데 이 과정에서 "Submission Scoring Error", "Notebook Exceeded Allowed Compute" 결과를 보였지만 이는 모두 out of memory 문제였음. 이 또한 위와 비슷하게 이미지 전처리 과정에서 N개의 이미지가 전처리될 때 마다 ```gc.collect()```를 해주니 효과가 있었음.
```
if cnt%N == 0:
    gc.collect()
```

- 제출 후 test data 추론과정에서 gpu의 out of memory 문제가 있었음. 한번에 test dataset을 모델에 입력하다보니 문제가 되었을 것이라 판단하고 test data 를 추론 시 mini batch 크기 만큼 모델에 입력하여 추론을 진행하였더니 해결이 되었음.
```
batch_size=256
for batch in range(0, X_test.shape[0], batch_size):
    preds = model.predict(X_test[batch:batch+batch_size])
# preds를 정리 후
gc.collect()
```

### 02.14 ~ 02.18
- baseline 와 beginner 위주의 커널들을 보면서 데이터를 처리하는 흐름을 읽음
- baseline 커널을 수정하여 제출까지 완료 (score : 0.9290)
- 
