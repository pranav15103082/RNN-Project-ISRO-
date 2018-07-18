from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from keras.layers import Dense, Conv1D, Input, MaxPooling1D, Flatten, Dropout
from keras import Sequential
from keras.utils import np_utils
from PIL import Image
import filereader
import tempreader
import array
import sys
row =1500
col = 1000
bands = 8
c_c = 5


def ReadBilFile(bil,bands,pixels):
    extract_band = 1
    image = np.zeros([pixels, bands], dtype=np.uint16)
    gdal.GetDriverByName('EHdr').Register()
    img = gdal.Open(bil)
    while bands >= extract_band:
        bandx = img.GetRasterBand(extract_band)
        datax = bandx.ReadAsArray()
        temp = datax
        store = temp.reshape(pixels)
        for i in range(pixels):
            image[i][extract_band - 1] = store[i]
        extract_band = extract_band + 1
    return image


pixels = row * col
y_test = np.zeros([row * col], dtype=np.uint16)
x_test = ReadBilFile("data/apex8bands", bands, pixels)
x_test = x_test.reshape(row*col, bands,1)
values = []
path = ["buildings","Forest","Grassland","river"]
dict1 = {"buildings": 0, "Forest": 0, "Grassland": 0, "river": 0}

for address in path:
    with open("8_Band/"+address, "rb") as f:
        b = array.array("H")
        b.fromfile(f, 800)
        if sys.byteorder == "little":
           b.byteswap()
        for v in b:
          values.append(v)
          dict1[address] += 1

ll = (len(values))
rex = ll // bands
print(ll, rex)
'''from here'''
f_in = np.zeros([ll], dtype=np.uint16)
x = 0
for i in range(ll):
        f_in[x] = values[i]
        x += 1

sh = int(rex // bands)
y_train = np.zeros([(dict1["buildings"] + dict1["Forest"] + dict1["Grassland"] + dict1["river"] ) // 8], dtype=np.uint16)
print(
    (dict1["buildings"] + dict1["Forest"] + dict1["Grassland"] + dict1["river"] ) )
for i in range(dict1["buildings"] // 8):
    y_train[i] = 1
for i in range(dict1["Forest"] // 8):
    y_train[dict1["buildings"] // 8 + i] = 2
for i in range(dict1["Grassland"] // 8):
    y_train[(dict1["buildings"] + dict1["Forest"]) // 8 + i] = 3
for i in range(dict1["river"] // 8):
    y_train[(dict1["buildings"] + dict1["Forest"] + dict1["Grassland"]) // 8 + i] = 4

'''
till here
'''
x_train = f_in.reshape(rex , bands)

#seed = 7
#np.random.seed(seed)

x_train = x_train / (2**16-1)
x_test = x_test / (2**16-1)
num_pixels = bands

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = c_c
y_test_new = np.zeros([pixels, c_c], dtype=np.uint8)

print(x_test)
print(20*'#')
print(x_train)
print(20*'#')
print(y_test)
print(20*'#')
print(y_train)

print(x_test.shape)
print(x_train.shape)
print(y_train.shape)
print(y_test.shape)

X = x_train.reshape(400, 8,1 )

model = Sequential()
#model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
model.add(Conv1D(2 ** 2, 1, activation="relu", padding='same', input_shape=[bands, 1]))
model.add(MaxPooling1D(2))
model.add(Conv1D(2 ** 3, 1, activation="relu", padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(2 ** 4, 1, activation="relu", padding='same'))
#model.add(MaxPooling1D(2))
#model.add(Conv1D(2 ** 1, 1, activation="relu", padding='same'))
model.add(Flatten())
model.add(Dropout(0.01))
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X, y_train, batch_size=20, epochs=2000, verbose=2)
#model.fit(x_train, y_train, batch_size=10, epochs=1000, verbose=2)
y_test_new = model.predict(x_test, batch_size=20)
print(20*'%')
#print(y_test_new[:,1])
print(y_test_new)
#print(np.squeeze(y_test_new))
print(20*'%')
y_test1 = np.argmax(y_test_new, axis=1)
print(30*'*')
print("this is predicted output")

#img = x_test.reshape(row, col, bands)
#plt.imshow(img)
#plt.show()
#result = Image.fromarray((img * 255).astype('uint8'))
#result.save('image.tiff')

"""
k = y_test_new.reshape(row, col, bands)
plt.imshow(k)
plt.show()
result = Image.fromarray((k * 255).astype('uint8'))
result.save('image.tiff')
"""
mul=2**16-1
img = y_test_new[:,1].reshape(row, col)
#img = y_test1.reshape(row, col)
plt.imshow(img*mul)
plt.colorbar()
plt.show()
result = Image.fromarray(((img * mul)).astype('uint16'))
result.save('Classified_images/2_buildings_new.tiff')

img = y_test_new[:,2].reshape(row, col)
#img = y_test1.reshape(row, col)
plt.imshow(img*mul)
plt.colorbar()
plt.show()
result = Image.fromarray(((img * mul)).astype('uint16'))
result.save('Classified_images/2_forest_new.tiff')

img = y_test_new[:,3].reshape(row, col)
#img = y_test1.reshape(row, col)
plt.imshow(img*mul)
plt.colorbar()
plt.show()
result = Image.fromarray(((img * mul)).astype('uint16'))
result.save('Classified_images/2_Grass_new.tiff')

img = y_test_new[:,4].reshape(row, col)
#img = y_test1.reshape(row, col)
plt.imshow(img*mul)
plt.colorbar()
plt.show()
result = Image.fromarray(((img * mul)).astype('uint16'))
result.save('Classified_images/2_river_new.tiff')

img = y_test_new[:,5].reshape(row, col)
#img = y_test1.reshape(row, col)
plt.imshow(img*mul)
plt.colorbar()
plt.show()
result = Image.fromarray(((img * mul)).astype('uint16'))
#result.save('Classified_images/1_Wheat_1.tiff')

img = y_test_new[:,6].reshape(row, col)
#img = y_test1.reshape(row, col)
plt.imshow(img*255)
plt.colorbar()
plt.show()
result = Image.fromarray(((img * 255*10)).astype('uint8'))
#result.save('Classified_images/1_Grassland_new.tiff')

print("img created")
#model.save('Classification_models/Classfication_model.hdf5')


