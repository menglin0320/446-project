from network import Network

class VGG16(Network):
    alpha = [0, 0, 0, 1, 1]
    beta  = [1, 1, 1, 1, 1]
    def setup(self):
        (self.conv(3, 3,   3,  64, name='vgg_conv1_1')
             .conv(3, 3,  64,  64, name='vgg_conv1_2')
             .pool(name='vgg_pool1')
             .conv(3, 3,  64, 128, name='vgg_conv2_1')
             .conv(3, 3, 128, 128, name='vgg_conv2_2')
             .pool(name='vgg_pool2')
             .conv(3, 3, 128, 256, name='vgg_conv3_1')
             .conv(3, 3, 256, 256, name='vgg_conv3_2')
             .conv(3, 3, 256, 256, name='vgg_conv3_3')
             .pool(name='vgg_pool3')
             .conv(3, 3, 256, 512, name='vgg_conv4_1')
             .conv(3, 3, 512, 512, name='vgg_conv4_2')
             .conv(3, 3, 512, 512, name='vgg_conv4_3')
             .pool(name='vgg_pool4')
             .conv(3, 3, 512, 512, name='vgg_conv5_1')
             .conv(3, 3, 512, 512, name='vgg_conv5_2')
             .conv(3, 3, 512, 512, name='vgg_conv5_3')
             .pool(name='vgg_pool5')
             .fc_layer(4096,name='vgg_fc6')
             .fc_layer(4096,name='vgg_fc7'))
             
    def fc7(self):
        return self.vardict['vgg_fc7']
        
class I2V(Network):
    alpha = [0,0,1,1,10]
    beta  = [0.1,1,10,10,100]
    def setup(self):
        (self.conv(3, 3,   3,  64, name='conv1_1')
             .pool()
             .conv(3, 3,  64, 128, name='conv2_1')
             .pool()
             .conv(3, 3, 128, 256, name='conv3_1')
             .conv(3, 3, 256, 256, name='conv3_2')
             .pool()
             .conv(3, 3, 256, 512, name='conv4_1')
             .conv(3, 3, 512, 512, name='conv4_2')
             .pool()
             .conv(3, 3, 512, 512, name='conv5_1')
             .conv(3, 3, 512, 512, name='conv5_2')
             .pool())

    def y(self):
        return [self.vardict['conv1_1'], self.vardict['conv2_1'], self.vardict['conv3_2'], self.vardict['conv4_2'], self.vardict['conv5_2']]
