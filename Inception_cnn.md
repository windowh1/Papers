# 논문구현 1
* 논문제목 : [Going deeper with convolutions] 
    * Inception version 1, GoogLeNet 으로 유명한 논문
* 개발환경
    * python 3.8
    * tensorflow v.2.9
* 아래 사이트의 도움을 많이 받았습니다.
    * https://ai.plainenglish.io/googlenet-inceptionv1-with-tensorflow-9e7f3a161e87

```python
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, Sequential, models, losses, Model
```

### Data Load
* MNIST 사용 : 28 * 28 크기
* GoogLeNet model은 224 * 224 * 3 크기의 data 사용
* MNIST datasets를 padding을 통해 224 * 224 * 3 크기로 맞출 것임
* 60000개의 train observation 중 5000개를 validation set으로 설정

```python
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print("mnist shape :", x_train.shape) #(60000, 28, 28) #(개수, width, height)
plt.imshow(x_train[1])
```

```python
x_train = x_train / 255 

# axis 위치에 dimension 하나 추가
x_train = tf.expand_dims(x_train, axis = 3)
print("mnist expand shape :", x_train.shape)
plt.imshow(x_train[1])
```

```python
x_train = tf.repeat(x_train, 3, axis = 3)
print("mnist repeat shape :", x_train.shape)
plt.imshow(x_train[1])
```

* x_test도 동일한 절차 밟기

```python
x_test = x_test / 255
x_test = tf.expand_dims(x_test, axis = 3)
x_test = tf.repeat(x_test, 3, axis = 3)
```

* y_data one-hot encoding

```python
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
print("y_train one-hot shape", y_train.shape)
```

* validation set 설정

```python
x_val = x_train[-5000:, :, :, :]
x_train = x_train[:-5000, :, :, :]
y_val = y_train[-5000:,]
y_train = y_train[:-5000,]
```

### Inception Model

```python
def inception(x
            , filters_1b1
            , filters_3b3_reduce
            , filters_3b3
            , filters_5b5_reduce
            , filters_5b5
            , filters_pool) :

    path1 = layers.Conv2D(filters = filters_1b1, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation = 'relu')(x)
    path2 = layers.Conv2D(filters = filters_3b3_reduce, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation = 'relu')(x)
    path2 = layers.Conv2D(filters = filters_3b3, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(path2)
    path3 = layers.Conv2D(filters = filters_5b5_reduce, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation = 'relu')(x)
    path3 = layers.Conv2D(filters = filters_5b5, kernel_size = (5, 5), strides = (1, 1), padding = 'same')(path3)
    path4 = layers.MaxPooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    path4 = layers.Conv2D(filters = filters_pool, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation = 'relu')(path4)

    return tf.concat([path1, path2, path3, path4], axis = -1)  
    # path1.shape : (None, 28, 28, path1_channel_num)
    # path2.shape : (None, 28, 28, path2_channel_num)
    # path3.shape : (None, 28, 28, path3_channel_num)
    # path4.shape : (None, 28, 28, path4_channel_num)
    # concat(..., axis = -1).shape : (None, 28, 28, path 1 + 2 + 3 + 4_channel_num)

```

```python
inp = tf.keras.Input(shape = (28, 28, 3))

# Resizing # shape = (224, 224, 3)
inp_resize = layers.Resizing(height = 224, width = 224, interpolation = 'bicubic')(inp)

# Conv shape = (112, 112, 64)
x = layers.Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), padding = 'same')(inp_resize)

# Max Pool shape = (56, 56, 64)
x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

# Conv shape = (56, 56, 192)
x = layers.Conv2D(filters = 192, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(x)

# Max Pool shape = (28, 28, 192)
x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

# Inception(3a) shape = (28, 28, 64 + 128 + 32 + 32 = 256)
x = inception(x, filters_1b1 = 64, filters_3b3_reduce = 96, filters_3b3 = 128, filters_5b5_reduce = 16, filters_5b5 = 32, filters_pool = 32)

# Inception(3b) shape = (28, 28, 128 + 192 + 96 + 64 = 480)
x = inception(x, filters_1b1 = 128, filters_3b3_reduce = 128, filters_3b3 = 192, filters_5b5_reduce = 32, filters_5b5 = 96, filters_pool = 64)

# Max Pool shape = (14, 14, 480)
x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

# Inception(4a) shape = (14, 14, 192 + 208 + 48 + 64 = 512)
x = inception(x, filters_1b1 = 192, filters_3b3_reduce = 96, filters_3b3 = 208, filters_5b5_reduce = 16, filters_5b5 = 48, filters_pool = 64)

# Auxiliary shape = (4, 4, 512)
aux1 = layers.AveragePooling2D(pool_size = (5, 5), strides = (3, 3), padding = 'valid')(x)
# Auxiliary shape = (4, 4, 128)
aux1 = layers.Conv2D(filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation = 'relu')(aux1)
# Auxiliary shape = (4 * 4 * 512)
aux1 = layers.Flatten()(aux1)
# Auxiliary shape = (1024)
aux1 = layers.Dense(units = 1024, activation = 'relu')(aux1)
aux1 = layers.Dropout(0.7)(aux1)
# Auxiliary shape = (10)
aux1 = layers.Dense(units = 10, activation = 'softmax')(aux1) # units = 10 : mnist data는 0 ~ 9 10종류

# Inception(4b) shape = (14, 14, 160 + 224 + 64 + 64 = 512)
x = inception(x, filters_1b1 = 160, filters_3b3_reduce = 112, filters_3b3 = 224, filters_5b5_reduce = 24, filters_5b5 = 64, filters_pool = 64)

# Inception(4c) shape = (14, 14, 128 + 256 + 64 + 64 = 512)
x = inception(x, filters_1b1 = 128, filters_3b3_reduce = 128, filters_3b3 = 256, filters_5b5_reduce = 24, filters_5b5 = 64, filters_pool = 64)

# Inception(4d) shape = (14, 14, 112 + 288 + 64 + 64 = 528)
x = inception(x, filters_1b1 = 112, filters_3b3_reduce = 144, filters_3b3 = 288, filters_5b5_reduce = 32, filters_5b5 = 64, filters_pool = 64)

# Auxiliary shape = (4, 4, 528)
aux2 = layers.AveragePooling2D(pool_size = (5, 5), strides = (3, 3), padding = 'valid')(x)
# Auxiliary shape = (4, 4, 128)
aux2 = layers.Conv2D(filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation = 'relu')(aux2)
# Auxiliary shape = (4 * 4 * 128)
aux2 = layers.Flatten()(aux2)
# Auxiliary shape = (1024)
aux2 = layers.Dense(units = 1024, activation = 'relu')(aux2)
aux2 = layers.Dropout(0.7)(aux2)
# Auxiliary shape = (10)
aux2 = layers.Dense(units = 10, activation = 'softmax')(aux2)

# Inception(4e) shape = (14, 14, 256 + 320 + 128 + 128 = 832)
x = inception(x, filters_1b1 = 256, filters_3b3_reduce = 160, filters_3b3 = 320, filters_5b5_reduce = 32, filters_5b5 = 128, filters_pool = 128)

# Max Pool shape = (7, 7, 832)
x = layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

# Inception(5a) shape = (7, 7, 256 + 320 + 128 + 128 = 832)
x = inception(x, filters_1b1 = 256, filters_3b3_reduce = 160, filters_3b3 = 320, filters_5b5_reduce = 32, filters_5b5 = 128, filters_pool = 128)

# Inception(5b) shape = (7, 7, 384 + 384 + 128 + 128 = 1024)
x = inception(x, filters_1b1 = 384, filters_3b3_reduce = 192, filters_3b3 = 384, filters_5b5_reduce = 48, filters_5b5 = 128, filters_pool = 128)

# Average Pool shape = (1, 1, 1024)
x = layers.GlobalAveragePooling2D()(x) # same as layers.AveragePooling2D(poolsize = 7, strides = 1)
x = layers.Dropout(0.4)(x)

# out shape = (10)
out = layers.Dense(units = 10, activation = 'softmax')(x)
```

```python
model = Model(inputs = inp, outputs = [out, aux1, aux2])
model.compile(optimizer = 'adam'
            , loss = [losses.categorical_crossentropy
                    , losses.categorical_crossentropy
                    , losses.categorical_crossentropy]
            , loss_weights = [1, 0.3, 0.3]
            , metrics = ['accuracy'])

model.fit(x_train, [y_train, y_train, y_train]
        , batch_size = 100
        , validation_data = (x_val, [y_val, y_val, y_val])
        , epochs = 10)
```
