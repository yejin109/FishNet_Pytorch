# Custom FishNet by Pytorch
****** 

## References
- https://github.com/kevin-ssy/FishNet
- https://github.com/osmr/imgclsmob 

해당 프로젝트는 로컬에서 진행한 것으로 사용한 GPU는 1660S이기 때문에 원문에서 제안하는 모델을 경량화하여 구현한 것입니다. 

## Architecture
***** 

### Fish Block 

일종의 ResNet Residual 연산과 같은 역할을 하는 FishNet의 기본단위로 기본적인 convolution layer는 동일하다. 단,  Upsampling과 down sampling에 따라서 사용하는 residual이 달라지게 된다. <br> 

Upsampling의 경우에는 channel wise reduction이 들어가게 되는데, 각 채널별로 k개의 인근 채널들의 feature map의 합을 취하게 되는 과정으로 채널단에서 일어나는 연산이기 때문에 batch size와 feature map의 크기는 고정시키고 channel에서 size를 조정하는 방식으로 연산을 진행하게 된다. 

## Residual Block 

Fish Block을 기본단위로 사용하여 채널을 바꾸는 연산과 함께 모델의 표현력을 증가시키기 위해 채널 변화 없이 FishBlock을 추가한다. 이 때 로컬 GPU를 고려하여 지정하도록 한다. 

## SE Block 

tail이 끝난 뒤 body로 넘어가기 전에 SENet에서 사용하는 SE Block을 사용한다. 이 부분은 BackBone으로서 FishNet이 기존의 모델들(SENet, ResNet, U-Net 등)의 장점을 반영한 것이지 않을까 생각한다. 

## Score block 

최종 분류 스코어 연산 과정에서 Linear layer을 사용하는 것보다 1*1 Conv Layer를 사용하였다. 

## Forward 

FishNet이 U-Net처럼 이전 단계의 동일한 feature map size를 가지는 tensor들을 이후의 레이어에 사용하는 구조를 가지기 때문에 각 output마다 변수명을 달리하여 사용한다.
