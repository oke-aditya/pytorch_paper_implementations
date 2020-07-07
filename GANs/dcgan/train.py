from dcgan import Generator, Discriminator, weights_init
import torch
from tqdm import tqdm
import torch.nn as nn
import engine
import torch.nn.init as init
# from config import *
import config
import torch.optim as optim
import data

if __name__ == "__main__":
    # Create the nets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, dataloader = data.create_dataset()

    gen_net = Generator().to(device)
    gen_net.apply(weights_init)
    
    dis_net = Discriminator().to(device)
    dis_net.apply(weights_init)
    
    # Imporant. We need to add noise to images to learn properly
    fixed_noise = torch.randn(config.batchSize, config.nz, 1, 1, device=device) 
    real_label = 1
    fake_label = 0
    
    criterion = nn.BCELoss()

    # We need 2 seperate optimizers, the Generator and the Discriminator
    gen_opt = optim.Adam(gen_net.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    dis_opt = optim.Adam(dis_net.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    # For checkpointing purposes
    max_err =  99999999999999999999

    for epoch in tqdm(range(config.EPOCHS)):
        err_gen, err_disc = engine.train_step(dataloader, criterion, gen_net, dis_net, gen_opt, dis_opt, device)
        print("Epochs = {}, Generator error = {}, Discriminator error = {}".format(epoch, err_gen, err_disc))

        if(err_gen + err_disc < max_err):
            print("Checkpointing the better model")
            torch.save(gen_net.state_dict(), f"Generator_{epoch}.pt")
            torch.save(dis_net.state_dict(), f"Discriminator_{epoch}.pt")
            


