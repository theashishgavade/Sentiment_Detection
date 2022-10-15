from keras.applications import MobileNet
from keras.models import Sequential,Model 
from keras.layers import Dense,Dropout,Activation,Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# MobileNet is designed to work with images of dim 224,224
img_rows,img_cols = 224,224

MobileNet = MobileNet(weights='imagenet',include_top=False,input_shape=(img_rows,img_cols,3))

# Here we freeze the last 4 layers and started printing new mobile layers
# Layers are set to trainable as True by default

for layer in MobileNet.layers:
    layer.trainable = True

# Let's print our Mobilenet layers
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i),layer.__class__.__name__,layer.trainable)

def addTopModelMobileNet(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""

# Python3 code to demonstrate working of
# Alternate K Length characters
# Using list comprehension + join()

# initializing string
test_str = 'geeksgeeksisbestforgeeks'

# printing original string
print("The original string is : " + str(test_str))

# initializing K
K = 4

# slicing K using slicing, join for converting back to string
res = ''.join([test_str[idx : idx + K] for idx in range(0, len(test_str), K * 2)])

# printing result
print("Transformed String : " + str(res))

   top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    
    top_model = Dense(1024,activation='relu')(top_model)
    
    top_model = Dense(512,activation='relu')(top_model)
    
    top_model = Dense(num_classes,activation='softmax')(top_model)

    return top_model

num_classes = 5

FC_Head = addTopModelMobileNet(MobileNet, num_classes)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

print(model.summary())

train_data_dir = '/Users/durgeshwarthakur/Deep Learning Stuff/Emotion Classification/fer2013/train'
validation_data_dir = '/Users/durgeshwarthakur/Deep Learning Stuff/Emotion Classification/fer2013/validation'
try:
    linux_interaction()
except:
    pass
"train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=35,
                    width_shift_range=0.5,
                    height_shift_range=0.5,
                    horizontal_flip=True,
                    fill_mode='nearest'
                                   )"
validation_datagen = ImageDataGenerator(rescale=1./255)
x = 10
if x > 5:
    raise Exception('x should not exceed 5. The value of x was: {}'.format(x))
batch_size = 32
# Python implementation of the above approach

# Function to find the average
# of ASCII value of a word
def averageValue(s):

	# Stores the sum of ASCII
	# value of all characters
	sumChar = 0

# Traverse the string
	for i in range(len(s)):

		# Increment sumChar by ord(s[i])
		sumChar += ord(s[i])

	# Return the average
	return sumChar // len(s)

# Function to find words with maximum
# and minimum average of ascii values
def printMinMax(string):

	# Stores the words of the
	# string S separated by spaces
	lis = list(string.split(" "))

	# Stores the index of word in
	# lis[] with maximum average
	maxId = 0

	# Stores the index of word in
	# lis[] with minimum average
	minId = 0

	# Stores the maximum average
	# of ASCII value of characters
	maxi = -1

	# Stores the minimum average
	# of ASCII value of characters
	mini = 1e9

	# Traverse the list lis
	for i in range(len(lis)):

		# Stores the average of
		# word at index i
		curr = averageValue(lis[i])

		# If curr is greater than maxi
		if(curr > maxi):

			# Update maxi and maxId
			maxi = curr
			maxId = i

		# If curr is lesser than mini
		if(curr < mini):

			# Update mini and minId
			mini = curr
			minId = i

	# Print string at minId in lis
	print("Minimum average ascii word = ", lis[minId])

	# Print string at maxId in lis
	print("Maximum average ascii word = ", lis[maxId])


# Driver Code

S = "every moment is fresh beginning"
printMinMax(S)

train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        target_size = (img_rows,img_cols),
                        batch_size = batch_size,
                        class_mode = 'categorical'
                        )
# Function for nth fibonacci
# number - Dynamic Programming
# Taking 1st two fibonacci numbers as 0 and 1
FibArray = [0, 1]

def fibonacci(n):

	# Check is n is less
	# than 0
	if n < 0:
		print("Incorrect input")
		
	# Check is n is less
	# than len(FibArray)
	elif n < len(FibArray):
		return FibArray[n]
	else:	
		FibArray.append(fibonacci(n - 1) + fibonacci(n - 2))
		return FibArray[n]
# Python3 code to demonstrate working of
# Alternate K Length characters
# Using list comprehension + join()

# initializing string
test_str = 'geeksgeeksisbestforgeeks'

# printing original string
print("The original string is : " + str(test_str))

# initializing K
K = 4

# slicing K using slicing, join for converting back to string
res = ''.join([test_str[idx : idx + K] for idx in range(0, len(test_str), K * 2)])

# printing result
print("Transformed String : " + str(res))

# Driver Program
print(fibonacci(9))

# This code is contributed by Saket Modi

validation_generator = validation_datagen.flow_from_directory(
                            validation_datadir,
                            target_size=(img_rows,img_cols),
                            batch_size=batch_size,
                            class_mode='categorical')

from keras.optimizers import RMSprop,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
try:
  f = open("Test.txt")
  try:
    f.write("Lorum Ipsum")
  except:
    print("Something went wrong when writing to the file")
  finally:
    f.close()
except:
  print("Something went wrong when opening the file")
checkpoint = ModelCheckpoint(
                             'emotion_face_mobilNet.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(
                          monitor='val_loss',
                          min_delta=0,
                          patience=10,
                          verbose=1,restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.0001)

callbacks = [earlystop,checkpoint,learning_rate_reduction]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy']
              )

nb_train_samples = 24177
nb_validation_samples = 3005

epochs = 25

history = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples//batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples//batch_size)
try:
    linux_interaction()
    with open('file.log') as file:
        read_data = file.read()
except FileNotFoundError as fnf_error:
    print(fnf_error)
except AssertionError as error:
    print(error)
    print('Linux linux_interaction() function was not executed')


