import tensorflow as tf

class conv_block(tf.keras.Model):
    def __init__(self, filter_out, kernel_size):
        super(conv_block, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filter_out,(1,1),(1,1))
        
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')
        
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(4*filter_out,(1,1),(1,1))
        
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.shortcut = tf.keras.layers.Conv2D(4*filter_out,(1,1), (1,1))
        

    def call(self, x, training=False, mask=None):
        h=self.bn1(x,training=training)
        h=tf.nn.relu(h)
        h=self.conv1(h)
        
        h=self.bn2(h,training=training)
        h=tf.nn.relu(h)
        h=self.conv2(h)
        
        h=self.bn3(h,training=training)
        h=tf.nn.relu(h)
        h=self.conv3(h)

        x_shortcut=self.bn4(x,training=training)
        x_shortcut=tf.nn.relu(x_shortcut)
        x_shortcut=self.shortcut(x_shortcut)
        
        h=tf.keras.layers.Add()([h,x_shortcut])
        
        return h
        
class identity_block(tf.keras.Model):
    def __init__(self, filter_out, kernel_size):
        super(identity_block, self).__init__()

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filter_out,(1,1),(1,1))
        
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')
        
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(4*filter_out,(1,1),(1,1))
        

    def call(self, x, training=False, mask=None):
        h=self.bn1(h,training=training)
        h=tf.nn.relu(h)
        h=self.conv1(h)
        
        h=self.bn2(h,training=training)
        h=tf.nn.relu(h)
        h=self.conv2(h)
        
        h=self.bn3(h,training=training)
        h=tf.nn.relu(h)
        h=self.conv3(h)

        h=tf.keras.layers.Add()([h,x])
        
        return h
    
class convolutional_block(tf.keras.Model):
    def __init__(self, filter_out, kernel_size):
        super(convolutional_block, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filter_out,(1,1),(2,2))
        
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')
        
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(4*filter_out,(1,1),(1,1))
        
        self.bn4 = tf.keras.layers.BatchNormalization()  
        self.shortcut = tf.keras.layers.Conv2D(4*filter_out,(1,1),(2,2))
        
    def call(self, x, training=False, mask=None):
        h=self.bn1(h,training=training)
        h=tf.nn.relu(h)
        h=self.conv1(h)
        
        h=self.bn2(h,training=training)
        h=tf.nn.relu(h)
        h=self.conv2(h)
        
        h=self.bn3(h,training=training)
        h=tf.nn.relu(h)
        h=self.conv3(h)

        x_shortcut=self.bn4(x)
        x_shortcut=tf.nn.relu(x_shortcut)
        x_shortcut=self.shortcut(x_shortcut)
        
        h=tf.keras.layers.Add()([h, x_shortcut])
        
        return h
        
inputs=tf.keras.layers.Input(shape=(224,224,3))

net=tf.keras.layers.ZeroPadding2D((3,3))(inputs)
net=tf.keras.layers.Conv2D(filters=64,kernel_size=(7,7),strides=(2,2))(net)
net=tf.keras.layers.BatchNormalization()(net)
net=tf.nn.relu(net)
net=tf.keras.layers.ZeroPadding2D((1,1))(net)

net=tf.keras.layers.MaxPooling2D((3,3),(2,2))(net)
net=conv_block(64, (3,3))(net)
net=identity_block(64, (3,3))(net)
net=identity_block(64, (3,3))(net)
        
net=convolutional_block(128, (3,3))(net)
net=identity_block(128, (3,3))(net)
net=identity_block(128, (3,3))(net)
net=identity_block(128, (3,3))(net)
        
net=convolutional_block(256, (3,3))(net)
net=identity_block(256, (3,3))(net)
net=identity_block(256, (3,3))(net)
net=identity_block(256, (3,3))(net)
net=identity_block(256, (3,3))(net)
net=identity_block(256, (3,3))(net)
        
net=convolutional_block(512, (3,3))(net)
net=identity_block(512, (3,3))(net)
net=identity_block(512, (3,3))(net)

net=tf.keras.layers.BatchNormalization()(net)
net=tf.nn.relu()(net)        
net=tf.keras.layers.GlobalAveragePooling2D()(net)
        
net=tf.keras.layers.Flatten()(net)
net=tf.keras.layers.Dense(num_classes, activation='softmax')(net)

model=tf.keras.Model(inputs=inputs,outputs=net)
