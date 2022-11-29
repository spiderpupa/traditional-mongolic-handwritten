# traditional-mongolic-handwritten

## 개요

* 고전몽골어, 혹은 몽골 비칙(монгол бичиг)은 선형이며 좌종서 형태의 문자 체계로 위구르 문자를 바탕으로 만들어졌다. 종서 형태임에도 오른쪽에서 왼쪽으로 써나가지 않는 이유는 위구르인들이 중국의 표기 체계를 흉내내기 위해 그들의 문자를 반시계 방향으로 90도 회전했기 때문이라 한다. 몇몇 모음(o/u, ö/ü, 끝위치의 a/e) 및 자음들(t/d, k/ɡ, ž/y)을 구분할 수 없는 경우가 있으며 어두, 어중, 어미 등 위치에 따라 그 형태가 바뀌는데, 보통은 모음 조화와 모음의 순서가 필요하여 구분이 가능하기 때문에 문제 없이 읽을 수 있다. 

##### <고전몽골어 문자표>
![스크린샷(10)](https://user-images.githubusercontent.com/101073973/204441403-a9382958-93e1-483a-b41d-9c212a5de501.png)


* 이와 같은 고전몽골어의 특성은 사람이 읽기에는 문제가 없지만 기계가 학습하도록 만드는 데에는 복잡한 절차가 필요하다고 보았으며, 이미 데이터가 있는 라틴 알파벳, 한글 등에 비해 희소하다고 여겨 문자 인식을 시도할 가치가 있다고 생각한다. 학습에 필요한 데이터셋은 kaggle에 공유된 파일(https://www.kaggle.com/datasets/fandaoerji/mhw-dataset)을 가져왔으며, 10만 개의 데이터를 가진 trainset, 각각 5,000개와 14,085개의 데이터가 들어있는 testset 2개 및 이에 해당하는 라벨이 들어있다.  

* 원본 데이터셋은 .mat 확장자로 되어있어 전용 프로그램이 필요하며, 몽골어의 특성 상 알파벳 별보단 단어 별로 구분하는 편이 학습에 유리하다고 보았다. 데이터의 사이즈 또한 단어의 길이에 따라 제각각이기 때문에 python을 통해 데이터셋 내의 배열을 읽고, 가장 긴 이미지에 맞춰 패딩을 추가 후 각각 이미지 파일 형태로 저장하기로 한다.
<br>
<br>

## 데이터셋 처리 후 이미지 파일로 저장

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
> del(data)<br>
> del(lengthlist)<br>
> del(a)<br>
> del(image)<br>
> del(dataReshaped)<br>
> del(margin)<br>
> del(upper_padding)<br>
> del(lower_padding)<br>
> del(upper_margin)<br>
> del(lower_margin)<br>
> del(img)
<br>
<br>

## 데이터 라벨링
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

## 데이터 생성기 클래스
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

## 학습



