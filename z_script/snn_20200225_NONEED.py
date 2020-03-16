from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras import backend as K
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model

img_shape = (384,384,1)

def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = Conv2D(filter, (1,1), activation="relu", **kwargs)(x)
    y = BatchNormalization()(y)
    y = Conv2D(filter,(3,3), activation="relu", **kwargs)(y)
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1],(1,1), **kwargs)(y)
    y = Add()([x,y])
    y = Activation("relu")(y)
    return y

def build_model(lr, l2, activation="sigmoid"):
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {"padding": "same", "kernel_regularizer": regul}

    ### branch model
    inp = Input(shape=img_shape)
    x = Conv2D(64,(9,9), activation="relu", **kwargs)(inp)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64,(3,3), activation="relu", **kwargs)(x)
    x = MaxPooling2D((2,2),strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (1,1), activation="relu", **kwargs)(x)
    for _ in range(4):
        x = subblock(x,64,**kwargs)
    x = MaxPooling2D((2,2),strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (1,1), activation="relu", **kwargs)(x)
    for _ in range(4):
        x = subblock(x,64,**kwargs)
    x = MaxPooling2D((2,2), stides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(384, (1,1), activation="relu", **kwargs)(x)
    for _ in range(4):
        x = subblock(x,96,**kwargs)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (1,1), activation="relu", **kwargs)(x)
    for _ in range(4):
        x = subblock(x,128,**kwargs)
    x = GlobalMaxPooling2D()(x)
    branch_model = Model(inp, x, name="branch")

    ### head model
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])
    x1 = Lambda(lambda x : x[0]*x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x : x[0]+x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0]-x[1]))([xa_inp,xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1,x2,x3,x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name="reshape1")(x)

    x = Conv2D(mid,(4,1),activation="relu",padding="valid")(x)
    x = Reshape((branch_model.output_shape[1],mid,1))(x)
    x = Conv2D(1,(1,mid),activation="linear",padding="valid")(x)
    x = Flatten(name="flatten")(x)

    x = Dense(1,use_bias=True,activation=activation,name="weighted_average")(x)
    head_model = Model([xa_inp, xb_inp], x, name="head")

    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa,xb])
    model = Model([img_a,img_b],x)
    model.compile(optim,loss="binary_crossentropy",metrics=["binary_crossentropy","acc"])
    return model, branch_model, head_model



