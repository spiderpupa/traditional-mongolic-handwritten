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

* 데이터셋을 불러와 확인, 형태 확인 후 필요한 데이터셋 위치 확인
> data=sio.loadmat('/content/drive/MyDrive/mhw/data/Trainset.mat')<br>
> print("\nTrainset.mat")<br>
> print('dtype: ', type(data))<br>
> for i in data.keys():<br>
>    print('label name: "{name}", item quantity: {length}'.format(name=i, length=len(data[i])))<br>
![스크린샷(11)](https://user-images.githubusercontent.com/101073973/204445318-d6ec6474-47a2-4fbc-87ce-2d397e0b8257.png)

* 데이터셋이 들어있는 열은 train_data 열이므로 여기에서 데이터를 추출, 길이 비교를 통해 가장 긴 이미지를 찾는다. (각 이미지의 폭은 48로 고정되어있어 따로 찾지 않았음)
<br>
![스크린샷(12)](https://user-images.githubusercontent.com/101073973/204447298-9ea2d940-d52a-4909-8532-fe4222628d49.png)

* 가장 큰 이미지의 길이는 299, 그보다 짧은 이미지들은 길이가 299가 되도록 패딩을 추가한다.
![스크린샷(15)](https://user-images.githubusercontent.com/101073973/204447974-632a8e4c-6696-4780-81dd-72ee2c7745e7.png)
* 우선 1 x 48 형태의 ndarray 하나를 생성하고, 각각의 이미지를 1 x n 형태로 reshape 후 위아래로 배열을 추가
* 그 후 배열을 1 x 299 형태로 다시 reshape, png 형태로 저장

##### <결과물>
![스크린샷(16)](https://user-images.githubusercontent.com/101073973/204449409-bcab6d6e-d454-49fa-9d75-f317bae635d1.png)

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
### TODO

