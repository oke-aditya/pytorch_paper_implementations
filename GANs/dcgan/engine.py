import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_step(dataloader, criterion, gen_net, dis_net, gen_opt, dis_opt, device):
    real_label = 1
    fake_label = 0
    for i, data in enumerate(dataloader, 0):
        # Train the discriminator 
        
        # train with real
        dis_net.zero_grad()
        # Get a real dat from data loader
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        # Get the ral label
        label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)
        # Output from the discriminator
        output = dis_net(real_cpu)
        # Real loss
        disc_real_loss = criterion(output, label)
        disc_real_loss.backward()
        D_x = output.mean().item()

        # train with fake
        # Randomly create a noisy data 
        noise = torch.randn(batch_size, config.nz, 1, 1, device=device)
        # Fit the generator on this fake data
        fake = gen_net(noise)
        label.fill_(fake_label)
        # Get the output of discriminator on the fake
        output = dis_net(fake.detach())
        
        # Fake loss
        disc_fake_loss = criterion(output, label)
        disc_fake_loss.backward()

        D_G_z1 = output.mean().item()
        # Total dsicrimintator error.
        err_disc = disc_real_loss + disc_fake_loss
        dis_opt.step()
    
        # Train the Generator now
        # Update G network: maximize log(D(G(z)))
        gen_net.zero_grad()
        label.fill_(real_label) # Fake labels are real label for Generator
        # Get the generator output
        outputs = gen_net(fake)
        err_gen = criterion(output, label)
        err_gen.backward()
        D_G_z2 = output.mean().item()
        gen_opt.step()

    return err_gen.item(), err_disc.item()



