# Bengali.AI Handwritten Grapheme Classification
- [대회 주소](https://www.kaggle.com/c/bengaliai-cv19)

# ★
- Team Merge를 통해 팀 참여를 하려고 했지만 팀원과의 소통이 부족하여 Merge를 못해 Solo로 참가하였습니다.

### 대회의 흐름
```data load``` -> ```resize & image processing``` -> ```reshape``` -> ```model load```-> ```predict```
### 참고하기 좋은 자료
- [분류모델](https://github.com/qubvel/classification_models.git)
- [대회관련정보](https://bengali.ai/wp-content/uploads/CV19-COCO-Grapheme.pdf)
- [Best Single Model](https://www.kaggle.com/c/bengaliai-cv19/discussion/123198)

### 사용해본 모델 [이미지 분류 모델 모음](https://github.com/qubvel/classification_models.git)
- se-resnext50


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
- 커널 대회이다 보니 제출을 하면 숨겨진 test data를 커널 흐름에 따라 전처리되고 추론이 되는데 이 과정에서 "Submission Scoring Error", "Notebook Exceeded Allowed Compute" 결과를 보였지만 이는 모두 out of memory 문제였음. 이 또한 위와 비슷하게 이미지 전처리 과정에서 ```N```개의 이미지가 전처리될 때 마다 ```gc.collect()```를 해주니 효과가 있었음.
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
![](https://github.com/JeongMyeong/Ka.KR-Bengali.AI/blob/master/fig/se-resnext50-24epoch-loss%20graph.png)

### 02.18 ~~
- baseline을 수정하여 validation set에 대해 ```grapheme_root``` 96.5% ```vowel_diacritic``` 99.4% ```consonant_diacritic```  99.5% 정확도를 달성.
- 제출을 통해 리더보드에서 socre 96.31% 를 달성.
- mixup, cutoff 기법이 효과가 있던것으로 생각되지만 자세한 처리방법은 모르고 공개된자료를 통해 적용을 시키기만함.
- grapheme_root의 분류갯수가 168개로 아주 많다. grapheme_root 의 정확도에따라 LB Socore 도 상승하는 것으로 생각된다.
