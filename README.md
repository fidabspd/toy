# Toy

각종 토이프로젝트 모음.

## Annoying Orange

오랜지가 되어버린 침착맨의 '55도발 왜하냐고'  

- `main.py`파일을 `annoying_orange/codes/`에서 실행.

## Fxxk You Mosaic

캠에 가운데 손가락을 들어올린 모습이 포착되면 해당 부분만 모자이크 처리한다.  

- `gather_dataset.py`를 실행하면 데이터 셋을 새로 수집할 수 있다.  
  가운데 손가락을 들어올린채 카메라 화면을 클릭하면 클릭 한번마다 데이터를 수집하여 해당 손가락 모양이 fxxk you 임을 표기한다.
- `fy_filter.py`를 실행하고 카메라에 대고 가운데 손가락을 들어올리면 해당 부분만 모자이크 처리된 화면을 볼 수 있다.

## Invisible Cloak

화면에서 특정 색깔만을 찾아내어 해당 부분만 원래의 이미지로 대체하여 투명망토의 효과를 낸다.

- `main.py`를 실행하면 데모를 볼 수 있다.

## Time Series

이더리움 코인의 가격과 삼성전자의 주가를 일별로 예측한다.

- `main.py`를 실행하여 결과를 확인할 수 있다.  
  - 실행은 `time_series/codes/`에서 실행하는 것을 권장한다.
  - 실행할 때 `--file_name` 옵션으로 `--file_name='ethereum_mod.csv'` 혹은 `--file_name='se_mod.csv'` 옵션이 필수적으로 필요하다.
  - 자세한 실행 arguments는 `main.py`파일 확인

## Mask Detection

영상이나 카메라에 등장하는 사람이 마스크를 썼는지 안썼는지 판별한다.

- `main.py`를 실행하여 예측 결과를 출력할 수 있다.
  - `--file_name` 옵션을 주지 않으면 컴퓨터에 설치된 카메라를 사용한다.
    기존의 영상을 사용하고 싶으면 이를 이용하면 된다.

## Chatbot

챗봇과 대화할 수 있다.

- `chatbot/codes/`로 이동하여 `chatbot.py`를 실행하면 테스트가 가능하다.
  - 하지만 아쉽게도 Nvidia GPU를 사용하는 환경에서만 작동 가능하다.
    cpu도 가능하게 하려면 할 수 있겠지만 생략.
