# Configuration to DCGAN

# size of the latent z vector
nz = 100
# Number of channels, use 1 if grayscale
nc=3

# Number of generator filters
ngf = 64

# Number of discriminator filters
ndf = 64

# Learning rate This is the defualt specified in the paper
# They go against 1e-1 as it is too high
lr = 1e-2
# Again the defualt was 0.9 but the paper suggests to keep 0.5 to stabilize the training

# Adam Hyperparemter
beta1 = 0.5 

# Number of epochs
EPOCHS = 10

# Batch Size
batchSize = 16

# Image Size
imageSize = (226, 226)

