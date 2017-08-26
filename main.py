from keras.models import Model
from keras.layers import *

inputA = np.array([[-1,  1, -1,  1, -1,  1,-1, -1,-1, -1]], dtype=float)
inputB = np.array([[ 1, -1, -1, -1, -1, -1, 1, -1, 1, -1]], dtype=float)
inputC = np.array([[-1, -1,  1,  1,  1,  1, 1,  1, 1, -1]], dtype=float)
inputD = np.array([[-1, -1, -1, -1,  1,  1, 1, -1,-1, -1]], dtype=float)

targetA = np.array([[-1,  1,  1, 1, 1, 1,  1,  1, -1, -1]], dtype=float)
targetB = np.array([[-1, -1, -1, 1, 1, 1, -1, -1, -1, -1]], dtype=float)

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
for i in range(n):
    # unite A and B in one
    inputAB = Concatenate()([outA, outB])

    # pass through N1
    outN1 = layN1(inputAB)

    # sum results of N1 and N2 into A
    outA = Add()([outN1, outN2])

    # this is constant for all the passes except the first
    outB = outN3  # looks like B is never changing in your image....

    finalOut = Concatenate()([outA, outB])
    model = Model([inputA, inputB, inputC, inputF], finalOut)
