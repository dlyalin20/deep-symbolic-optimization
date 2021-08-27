from tensorflow import keras

# attempt to pass through a MLP to generate simpler corrections?

visible_1 = keras.Input(shape = X.shape)
cnn_1 = keras.layers.Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu')(visible_1)
dropout_1 = keras.layers.Dropout(0.1)(cnn_1)

visible_2 = keras.Input(shape = X.shape)

merge = keras.layers.Concatenate(axis = 3)([dropout_1, visible_2])

cnn_2 = keras.layers.Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(merge)
dropout_2 = keras.layers.Dropout(0.1)(cnn_2)

output = keras.layers.Conv2D(1, kernel_size = (3, 3), padding = 'same', activation = 'relu')(dropout_2)

model = keras.Model(inputs = [visible_1, visible_2], outputs = output)

model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(lr = 1e-3, decay = 1e-6))


def run(Xs, ys):
    global X, y
    X, y = Xs, ys