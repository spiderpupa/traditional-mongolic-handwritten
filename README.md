# CNN 이용 필기체 고전몽골어 인식

## 목차
1. [개요](##1-1.-개요)
2. [데이터셋 처리 후 이미지 파일로 저장](https://github.com/spiderpupa/traditional-mongolic-handwritten/blob/main/README.md#2-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EC%B2%98%EB%A6%AC-%ED%9B%84-%EC%9D%B4%EB%AF%B8%EC%A7%80-%ED%8C%8C%EC%9D%BC%EB%A1%9C-%EC%A0%80%EC%9E%A5)
3. [데이터 라벨링](https://github.com/spiderpupa/traditional-mongolic-handwritten/blob/main/README.md#3-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%9D%BC%EB%B2%A8%EB%A7%81)
4. [데이터 전처리](https://github.com/spiderpupa/traditional-mongolic-handwritten/blob/main/README.md#4-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC)
5. [학습 및 평가](https://github.com/spiderpupa/traditional-mongolic-handwritten/blob/main/README.md#5-%ED%95%99%EC%8A%B5-%EB%B0%8F-%ED%8F%89%EA%B0%80)
6. [결론](https://github.com/spiderpupa/traditional-mongolic-handwritten/blob/main/README.md#6-%EA%B2%B0%EB%A1%A0)

## 1. 개요

* **CNN에 관하여**
   * CNN은 Convolutional Neural Network의 약자로, 이미지를 행렬 형태로 읽어 필터의 각 요소가 데이터 처리에 적합하도록 자동으로 학습 과정을 통해 이미지를 분류하는 기법 
   * 대용량 파일을 전처리하는 법을 익히고, 적절한 형태로 가공하여 정상적으로 작동하도록 딥러닝 모델을 구현하는 것이 목적
* **고전몽골어**
    * 고전몽골어, 혹은 몽골 비칙(монгол бичиг)은 선형이며 좌종서 형태의 문자 체계로 위구르 문자를 바탕으로 만들어짐
    * 종서 형태임에도 오른쪽에서 왼쪽으로 써나가지 않는 이유는 위구르인들이 중국의 표기 체계를 흉내내기 위해 그들의 문자를 반시계 방향으로 90도 회전했기 때문
    * 몇몇 모음(o/u, ö/ü, 끝위치의 a/e) 및 자음들(t/d, k/ɡ, ž/y)을 구분할 수 없는 경우가 있으며 어두, 어중, 어미 등 위치에 따라 그 형태가 바뀌지만 보통 모음 조화와 모음의 순서가 필요하여 구분이 가능 

    ##### <고전몽골어 문자표>
    ![스크린샷(10)](https://user-images.githubusercontent.com/101073973/204441403-a9382958-93e1-483a-b41d-9c212a5de501.png)

   * 개별 알파벳마다 3가지의 패턴을 가지고 있으며, 단순히 형태 상으로는 서로 다른 알파벳이라도 구별할 수 없는 경우가 많아 이를 쉽게 전자 텍스트로 바꾸기 어려움
* 데이터셋
    - 학습에 필요한 데이터셋은 kaggle에 공유된 파일(https://www.kaggle.com/datasets/fandaoerji/mhw-dataset)을 사용
    - 10만 개의 데이터를 가진 trainset, 각각 5,000개와 14,085개의 데이터가 들어있는 testset 2개 및 이에 해당하는 라벨로 구성
    - datasets from FANDAOERJI(https://www.kaggle.com/datasets/fandaoerji/mhw-dataset), last downloaded time is 2022.10.26 KST 14:08
    - 원본 데이터셋은 .mat 확장자로 되어있어 전용 프로그램이 필요, 알파벳 단위가 아닌 단어 단위로 저장, 데이터의 사이즈가 단어의 길이에 따라 제각각
    - 따라서 python을 통해 데이터셋 내의 배열을 읽고, 가장 긴 이미지에 맞춰 패딩을 추가 후 각각 이미지 파일 형태로 저장하기로 함
    - 데이터셋 저장 폴더: https://drive.google.com/drive/folders/1TKAFFS5GjWm2-0GtrlbY83G3TLbPQ-35?usp=sharing
    - DataGenerator 클래스 사용하여 메모리 초과 방지

## 2. 데이터셋 처리 후 이미지 파일로 저장

#### 데이터셋을 불러와 확인, 형태 확인 후 필요한 데이터셋 위치를 확인한다.
```
data=sio.loadmat('/content/drive/MyDrive/mhw/data/Trainset.mat')
print("\nTrainset.mat")
print('dtype: ', type(data))
for i in data.keys():
    print('label name: "{name}", item quantity: {length}'.format(name=i, length=len(data[i])))
```
> <결과><br>
>![스크린샷(11)](https://user-images.githubusercontent.com/101073973/204452058-d74170c1-c720-49a1-b04c-6f0898b32355.png)

#### 데이터셋이 들어있는 열은 train_data 열이므로 여기에서 데이터를 추출, 길이 비교를 통해 가장 긴 이미지를 찾는다. (각 이미지의 폭은 48로 고정되어있어 따로 찾지 않았음)
```
def imageFile(n):
    return(data["train_data"][n][0])
    
lengthlist=[]
for i in range(len(data["train_data"])):
    lengthlist.append(imageFile(i).shape[0])

print('가장 긴 이미지 파일의 길이:', max(lengthlist),'px')
```
> 가장 긴 이미지 파일의 길이: 299 px<br>

#### 가장 큰 이미지의 길이는 299, 그보다 짧은 이미지들은 길이가 299가 되도록 패딩을 추가한다.
* 1 x 48 형태의 ndarray 하나를 생성
```
a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0 ]])
```
* 이미지 경로를 설정하고 배열 추가에 용이하도록 이미지를 n x 48 형태로 재배열
```
def outputPath(n):
    return('/content/drive/MyDrive/mhw/data/images/trainset_images/img%d.png' %(n+1))
    
image=(imageFile(i))
dataReshaped=(image.reshape(1,image.shape[0]*48))
```
* 패딩은 위아래로 균일하게, 만약 그러지 못할 경우 윗단이 1px 짧고 아랫단이 1px 긴 형태로 만듦
```
    margin=(299-image.shape[0])/2
     
    if margin%1==0:
        upper_padding=int(margin)
        lower_padding=int(margin)
    else:
        upper_padding=int(margin-0.5)
        lower_padding=int(margin+0.5)
```
* 저장용 배열을 만들고, 반복문을 통해 원하는 길이만큼 배열을 추가
```
    img=np.array([])
    upper_margin=np.array([])
    lower_margin=np.array([])

    for j in range(upper_padding):     
        upper_margin = np.append(a,upper_margin)
    for k in range(lower_padding):
        lower_margin = np.append(a,lower_margin)

    img=np.append(upper_margin,dataReshaped)
    img=np.append(img,lower_margin)
```
* 299 x 48 형태로 재배열 후 png 형태로 저장
```
    img=img.reshape(299,48)
    matplotlib.image.imsave(outputPath(i), img, cmap='gray')
```
* 반복문을 통해 이미지의 개수만큼 이를 반복
> <샘플><br>
![다운로드](https://user-images.githubusercontent.com/101073973/204458144-2ac936e0-b51f-4d7d-9541-569742c5f5e4.png)
<br>

* 메모리 관리 위해 del() 함수를 이용, 사용을 마친 데이터를 제거한다.
```
del(data)
del(lengthlist)
del(a)
del(image)
del(dataReshaped)
del(margin)
del(upper_padding)
del(lower_padding)
del(upper_margin)
del(lower_margin)
del(img)
```

## 3. 데이터 라벨링
### 데이터 생성기 클래스에 맞춰 라벨을 재설정
* 텍스트파일 형태의 라벨을 불러와 image, labels 두 개의 열로 이루어진 리스트로 저장
```
trainset_labels=['image','labels']
with open('/content/drive/MyDrive/mhw/data/labels/Trainset_label.txt', 'r', encoding='utf-16 le') as label:
    for i in range(100000):
        line=str(label.readline())
        line=line.replace('\ufeff', '')
        line=line.replace('\n', '')
        trainset_labels.append('img%d.png' %(i+1))
        trainset_labels.append(line)
    label.close()
```
* 100001 x 2 형태의 ndarray로 배열
```
trainset_labels=np.array(trainset_labels)
trainset_labels=trainset_labels.reshape(100001,2)
trainset_labels=pd.DataFrame(trainset_labels)
```
* csv 파일로 저장
```
trainset_labels.to_csv('/content/drive/MyDrive/mhw/data/labels/trainset_label.csv', 
                        index=False, 
                        header=False)
```
> ![스크린샷(18)](https://user-images.githubusercontent.com/101073973/204460766-0e134b08-90d7-4261-b640-a3ac532ab14a.png)
<br>

## 4. 데이터 전처리
* 데이터셋 분할
    - 데이터 학습 진행을 위해 불러온 데이터셋을 트레인셋과 테스트셋으로 나눔
    - 각 세트를 균등하게 나눌 수 있도록 scikit-learn의 train_test_split 기능을 이용
```
X=trainset_labels['image']
y=trainset_labels['labels']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)
```
> ![스크린샷(21)](https://user-images.githubusercontent.com/101073973/204726638-b4f03270-f087-4a9e-88f4-021a0e911774.png)<br>
각각 6만 개와 4만 개의 데이터로 나뉘었으며, 둘 다 5천 개의 라벨을 가지고 있어 정상적으로 분할된 것을 확인
* 트레인셋과 테스트셋으로 통합
train_labels과 test_labels로 데이터프레임을 생성 후 각각에 맞춰 데이터를 추가
>![스크린샷(22)](https://user-images.githubusercontent.com/101073973/204727372-dd55b0d8-ddcc-477c-bf21-1dd8e11ed8ef.png)
![스크린샷(23)](https://user-images.githubusercontent.com/101073973/204727384-e3a45bf8-6416-4157-a679-c023b56c5240.png)<br>
* 데이터 생성기 클래스
```
class DataGenerator(Sequence):
    def __init__(self, path, list_IDs, labels,img_sizeX, img_sizeY,
    batch_size, img_channel, num_classes):
     
        # 데이터셋 경로
        self.path = path
        # 데이터 이미지 개별 주소 [ DataFrame 형식 (image 주소, image 클래스) ]
        self.list_IDs = list_IDs
        # 데이터 라벨 리스트 [ DataFrame 형식 (image 주소, image 클래스) ]
        self.labels = labels
        # 가로 크기
        self.img_sizeX = img_sizeX
         # 세로 크기
        self.img_sizeY = img_sizeY
        # 학습 Batch 사이즈
        self.batch_size = batch_size
        # 이미지 채널 [RGB or Gray]
        self.img_channel = img_channel
        # 데이터 라벨의 클래스 수
        self.num_classes = num_classes
        # 전체 데이터 수
        self.indexes = np.arange(len(self.list_IDs))
   
    def __len__(self):
        len_ = int(len(self.list_IDs)/self.batch_size)
        if len_*self.batch_size < len(self.list_IDs):
            len_ += 1
        return len_
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
            
    def __data_generation(self, list_IDs_temp):
        X = np.zeros((self.batch_size, self.img_sizeX, self.img_sizeY, self.img_channel))
        y = np.zeros((self.batch_size, self.num_classes), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            img = cv2.imread(self.path+ID)
            X[i, ] = img
            y[i, ] = to_categorical(self.labels[i], num_classes=self.num_classes)
        return X, y
```
* 데이터 전처리
 - 전체 클래스 수
```
clss_num = len(train_labels['labels'].unique())
```
 - 라벨 원 핫 인코딩
```
labels_dict = dict(zip(train_labels['labels'].unique(), range(clss_num)))
train_labels = train_labels.replace({"labels": labels_dict})
test_labels = test_labels.replace({"labels": labels_dict})
```
 - 이미지의 크기(img_sizeX, img_sizeY), 채널(img_ch), 클래스 개수(num_class)와 배치 사이즈(batch_size) 설정
 - 생성기로 데이터 생성
```
train_generator = DataGenerator('/content/drive/MyDrive/mhw/data/images/trainset_images', train_labels['image'],
                                train_labels['labels'],
                                img_sizeX, img_sizeY,
                                batch_size, 
                                img_ch, num_class)

test_generator = DataGenerator('/content/drive/MyDrive/mhw/data/images/trainset_images', test_labels['image'],
                                test_labels['labels'],
                                img_sizeX, img_sizeY,
                                batch_size, 
                                img_ch, num_class)
```
## 5. 학습 및 평가
* 모델 설정
```
model=models.Sequential()
```
```
#인코딩
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(297, 48, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
#디코딩
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(933, activation='softmax'))
```
> ![스크린샷(20)](https://user-images.githubusercontent.com/101073973/204468251-ae489bf3-9403-437e-b3ea-986b7a6b2923.png)<br>
* 학습
```
checkpoint_path = 'D:/VS_Code_Workspace/mhw/Data/models/cp--{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0,
                                                 save_freq='epoch',
                                                 )  # 5번째 에포크마다 가중치를 저장
```
```
#학습과정 설정
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
```
history = model.fit_generator(train_generator, epochs=1)
```
* 평가
![스크린샷(1)](https://user-images.githubusercontent.com/101073973/209636591-c801d00b-182c-43f3-934b-ef81465aa8dc.png)
트레인셋의 정확도는 만족했지만 테스트셋의 정확도가 낮아 현재 단계에서는 원하는 정확도를 획득할 수 없었음.

## 6. 결론

* 이미지 처리 및 데이터 생성 클래스에서 문제가 발생한 것으로 보임
* 우선 현 단계에서 멈추고 근본적인 방식 혹은 레퍼런스 데이터를 바꿔야 원하는 수준으로 정확도를 향상시킬 수 있을 것으로 생각됨
* 데이터 전처리 및 CNN의 사용법에 대한 이해도는 높일 수 있었음
* 프로젝트를 다시 돌려서 정확도가 향상되는대로 업데이트할 계획


