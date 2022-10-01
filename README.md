# Sign-Language-Translator
KW-Capstone-project :  Korean Sign-Language Translator

## 개요 
>2014년부터 2020년까지 통계청에 따르면 국내 등록 농인수는 꾸준히 증가하고 있다. 농인이 일상생활에서 느끼는 불편한 점으로 의사 소통의 어려움, 편의 시설 부족, 외출 시 동반자 부재, 주변의 시선 및 기타 등이 있었지만, 그 중 가장 불편한 점은 의사 소통의 어려움이 54%에 달했다. 그리하여 농인이 어려움 없이 의사소통 할 수 있는 수어 번역기 웹 서비스 플랫폼을 개발하였다.
딥러닝 모델로 학습된 수어 번역기에 농인이 수화를 진행하면 실시간 번역이 단어별로 진행이 되고, 번역된 결과를 한국어 문장으로 출력한다. 농인과 의사소통하는 상대방은 수화를 이해하지 못하더라도 농인이 말하고자 하는 바를 알 수 있다. 완성된 수어 번역기에서는 공공데이터 세트의 제한에 따라 장소를 식당으로 한정 지어 식당에서 농인이 요구할 수 있는 상황을 설계해 제작하였지만, 이후 장소를 식당에서 국한 시킬게 아니라 다른 곳에서도 필요에 맞게끔 영상 데이터 셋을 얻어 딥러닝 모델링을 새로 한다면 어디든 쓸 수 있고 장애인들이 비장애인처럼 일상생활에서 누릴 수 있는 서비스들을 편리하게 이용할 수 있는 Barrier Free를 구축할 수 있다.
>본 프로젝트의 결과로 얻을 수 있는 기대효과 및 활용방안은 크게 3가지로 정의했다. 
첫 번째, 농인의 권리보장을 위한 통역 서비스 실현 가능하다. 일반인들과 비교하면 의사소통에 어려움을 겪는 경우가 많은 농인의 알 권리 보장 및 편리성 증진으로 인한 삶의 질 향상 효과를 기대할 수 있다. 두 번째, 시 공간 제약이 없는 수어 번역 서비스 제공이다. 수어 통역사의 도움 없이도 언제 어디서든지 농인들을 위한 수어 번역 서비스를 제공이 가능할 것이다. 세 번째, 한국어 외 다양한 언어권 적용 가능성이다. 본 프로젝트에서는 언어권을 한국으로 설정하여 개발을 진행했지만, 한국어 수어가 아닌 다른 언어권의 수어 데이터를 준비한다면 한국어뿐만 아니라 다른 언어권에서도 서비스 제공이 가능하다.

<br>
<hr>

## 설계 과정 

- AIhub 수어 영상 데이터 셋
- 필요한 수어 영상 데이터의 부족으로 인해 직접 수어를 배워 수어 영상 데이터세트를 구축함

![image](https://user-images.githubusercontent.com/87630540/193425726-253e7ba8-6d2c-42e5-a051-3686a44a62d4.png)

- MediaPipe의 Hollistic 솔루션을 이용하여 웹캠으로 부터 얻은 영상에서 왼손, 오른손 Keypoints를 추출

![image](https://user-images.githubusercontent.com/87630540/193425822-a4bd5ab2-3357-42c7-9d73-74391ad5ec68.png)

- 총 45개의 수어 단어 세트

![image](https://user-images.githubusercontent.com/87630540/193425878-4226f8d8-eb32-4126-9b6f-915ee9bfa097.png)

- LSTM 모델링은 tensorflow의 keras를 통해 진행되었다. input_shape=(timestep, feature)에서 timestep은 하나의 영상을 구성하는 프레임 개수, feature에는 126개의 왼 손, 오른 손 3D*(x, y, z) keypoints로 파라미터로 전달했다. 또 예측하고자 하는 target data인 수화 단어들(actions)을 출력층에 전달해 줬다. 또, 본 프로젝트 모델링에서는  hidden layer에 Stacked LSTM을 사용하여 LSTM이 더 복잡한 task를 해결할 수 있도록 LSTM 모델의 복잡도를 높혔다. 

![image](https://user-images.githubusercontent.com/87630540/193425920-52e3eaee-767e-48ec-aed1-a4915d9656c3.png)

- Flask-SocketIO를 이용하여 사용자로 부터 받은 영상을 실시간으로 처리하여 예측된 결과를 응답으로 보내줌

<br>
<hr>

## 동작과정

- 시작하기 버튼을 통해서 Client의 Webcam 에 접근 권한을 얻음.
번역하기 버튼을 통해서 수어 번역을 시작하며 사용자에게 올바른 위치를 알려주기 위해서 FaceDetection API를
이용해서 사용자의 위치가 규격에 맞게 들어올 경우 번역을 시작함.

![image](https://user-images.githubusercontent.com/87630540/193426000-9bfe2844-0401-443e-859d-48bafdc8ac33.png)

- 사용자가 올바른 곳에 위치하였을 때 규격을 나타내는 상자가 붉은색으로 변하면서 번역 시작을 알림

![image](https://user-images.githubusercontent.com/87630540/193426024-c37fa59d-5444-4d34-996e-95dd62918c5f.png)

- 사용자가 수어 동작을 수행

![image](https://user-images.githubusercontent.com/87630540/193426034-2f24fae5-bf7d-44d4-8f34-de7dd8308c41.png)

- 사용자로부터 받은 30개의 frame이 하나의 영상이 되어 서버에서 예측, 이후에 예측된 단어를 Client에게 Return

![image](https://user-images.githubusercontent.com/87630540/193426040-587c1bfa-a902-4aa0-8ecc-015bf60874f9.png)

- 입력 받은 영상으로부터 단어를 예측할 수 없을 경우 재동작을 요청

![image](https://user-images.githubusercontent.com/87630540/193426085-c3d1d1bb-23aa-424f-a247-bf7d5eb8b30b.png)

- 예측한 단어가 잘못된 단어일 경우 단어 삭제 버튼을 이용하여 예측된 단어를 삭제 가능

![image](https://user-images.githubusercontent.com/87630540/193426099-e46e050b-9480-41ab-a584-90b8f71522e8.png)

- 단어 삭제 버튼을 통해서 잘못 입력된 부탁이라는 단어가 삭제된 것을 확인가능

![image](https://user-images.githubusercontent.com/87630540/193426116-deafa730-0a86-469f-b4d5-7df36b838506.png)

- 접시, 주세요 단어가 입력된 것을 확인 가능

![image](https://user-images.githubusercontent.com/87630540/193426125-28327815-39c4-4d5c-8a28-65b6ac1e6e29.png)

- 입력된 단어 ‘접시’, ‘주세요’를 바탕으로 예측된 ‘접시 주세요’ 문장을 출력

![image](https://user-images.githubusercontent.com/87630540/193426135-85397428-00e9-46db-9b05-63137c156c5a.png)

<br>
<hr>

## 결론

>본 프로젝트에서는 클라이언트로부터 생성한 프레임을 서버로 전달하고 서버는 받은 프레임을 저장했다가 하나의 영상을 만들 수 있는 양이 되면 학습모델의 예측할 데이터로 사용했다. 그리고 이는 이벤트 핸들러를 통해 진행되었다. 그러나 클라이언트로 부터 오는 빠른 데이터 전송이 서버에게 큰 부담으로 작용했고 클라이언트에서는 딜레이가 발생했다. 딜레이를 개선하기 위해 클라이언트와 서버사이의 시간 동기를 맞추거나, 클라이언트에서 데이터를 보낼 때 약간의 지연이 발생하도록 설계 했다. 약간의 딜레이가 여전히 존재했다. 이를 해결하지 못한 채 수어 번역 서비스 플랫폼 프로젝트를 마무리 했다.

