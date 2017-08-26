from keras.models import Model
from keras.layers import *

time_delays = 2

input_arrA = np.array([[-1,  1, -1,  1, -1,  1,-1, -1,-1, -1]], dtype=float).reshape((10, 1))
input_arrB = np.array([[ 1, -1, -1, -1, -1, -1, 1, -1, 1, -1]], dtype=float).reshape((10, 1))
input_arrC = np.array([[-1, -1,  1,  1,  1,  1, 1,  1, 1, -1]], dtype=float).reshape((10, 1))
input_arrF = np.array([[-1, -1, -1, -1,  1,  1, 1, -1,-1, -1]], dtype=float).reshape((10, 1))

target = np.array([[-1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
                     1, 1, 1, -1, 1, -1, -1, -1, -1, -1]], dtype=float).reshape((10, 2))

print(target)

inputA = Input((1,))
inputB = Input((1,))
inputC = Input((1,))
inputF = Input((1,))

outN2 = Dense(1, activation='sigmoid')(inputC)
outN3 = Dense(1, activation='sigmoid')(inputF)

layN1 = Dense(1)

outA = inputA
outB = inputB

# n - number of time delays
for i in range(time_delays):
    # unite A and B in one
    inputAB = Concatenate()([outA, outB])

    # pass through N1
    outN1 = layN1(inputAB)

    # sum results of N1 and N2 into A
    outA = Add()([outN1, outN2])

    # this is constant for all the passes except the first
    outB = outN3  # looks like B is never changing in your image....

finalOut = Concatenate()([outA, outB])
model = Model(inputs=[inputA, inputB, inputC, inputF], outputs=finalOut)
print(model.layers[0].set_weights([[2]]))
print(model.layers[0].get_weights())
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit([input_arrA, input_arrB, input_arrC, input_arrF], target, epochs=5000, batch_size=1, verbose=2)
# validation_data=(x_test, y_test)
predict = model.predict(data)
