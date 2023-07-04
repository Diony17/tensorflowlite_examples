# Metadate Writer

학습된 텐서플로우 모델을 안드로이드, iOS 등 모바일 플랫폼에서 돌릴 때에는 주로 TensorFlow Lite용 모델 인터페이스 생성 도구 등의 도구(TFLite Support Library)가 모델을 자동으로 처리할 수 있도록 설정하는 경우가 많다.  
이런 도구들은 메타데이터를 기반으로 작동하므로, 메타데이터가 모델에 포함되어 있으면 TFLite Support Library는 모델의 입력과 출력에 대한 정보를 자동으로 파악하고 적절한 전처리와 후처리를 수행할 수 있다.

https://www.tensorflow.org/lite/models/convert/metadata?hl=ko