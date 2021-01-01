# ResNet
![캡처](https://user-images.githubusercontent.com/74402562/103434833-03349280-4c4a-11eb-86d6-d53baeb8b178.PNG)

기존 20계층 수준의 네트워크를 152계층까지 늘이는 성과를 거두었다.

ResNet의 구조
-------------
![resnet](https://user-images.githubusercontent.com/74402562/103434926-493e2600-4c4b-11eb-9264-082e2f7deda8.PNG)

눈에 띄는 Skip Connection이 주요한 역할을 한다.
- Skip Connection

  처음 제안되었던 Skip Connection 구조이다. Feature를 추출하기 전과 후를 더하는 특징이 있다. 일반구조에서 표현 가능한 것은 Residual구조에서도 표현 가능하다.
- Identity Mapping

  한 단위의 특징 맵을 추출하고 난 후에 활성 함수를 적용하는 것이 상식이었다. 하지만 개선된 구조에서는 Identity Mapping을 얻기 위해서 Pre-Activation을 제안하였다.
  - Pre-Activation
  
    Conv-BN-ReLU구조를 BN-ReLU-Conv구조로 변경한 것으로 성능이 개선되었다. 후자의 경우 Gradient Highway가 형성되어 극적인 효과를 얻는다.
