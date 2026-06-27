# This is a minimal implementation of a 1D variational autoencoder
import argparse, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt


np.random.seed(0); torch.manual_seed(0)
pi, mu, sig = np.array([.45, .25, .30]), np.array([-.35, 1.5, -2.0]), np.array([.45, .25, .30])
latent_dim, beta = 4, .1

parser = argparse.ArgumentParser()
parser.add_argument("--save-frames", action="store_true", help="save one progress image every --frame-every steps, then merge them into a 10s gif")
parser.add_argument("--frame-every", type=int, default=1000)
args = parser.parse_args()


def sample(n):
    k = np.random.choice(len(pi), n, p=pi)
    return np.random.normal(mu[k], sig[k]).astype("float32")[:, None]


def pdf(x):
    z = np.exp(-.5*((x[None]-mu[:,None])/sig[:,None])**2)/(sig[:,None]*np.sqrt(2*np.pi))
    return (pi[:,None]*z).sum(0)


encoder = nn.Sequential(nn.Linear(1,16), nn.Tanh(), nn.Linear(16,8), nn.Tanh())
enc_mu, enc_logvar = nn.Linear(8, latent_dim), nn.Linear(8, latent_dim)
decoder = nn.Sequential(nn.Linear(latent_dim,8), nn.Tanh(), nn.Linear(8,16), nn.Tanh(), nn.Linear(16,1))
opt = torch.optim.Adam(list(encoder.parameters()) + list(enc_mu.parameters()) + list(enc_logvar.parameters()) + list(decoder.parameters()), lr=1e-4)


def encode(x):
    h = encoder(x)
    return enc_mu(h), enc_logvar(h)


def decode(z):
    return decoder(z)


def vae(x):
    z_mu, z_logvar = encode(x)
    z = z_mu + torch.exp(.5 * z_logvar) * torch.randn_like(z_mu)  # reparameterization trick
    return decode(z), z_mu, z_logvar


@torch.no_grad()
def reconstruct(n=50000):
    x = torch.from_numpy(sample(n))
    z_mu, _ = encode(x)
    return x.numpy().ravel(), decode(z_mu).numpy().ravel()


@torch.no_grad()
def generate(n=50000):
    z = torch.randn(n, latent_dim)
    return decode(z).numpy().ravel()


def save_plot(path, title="VAE result", n=50000):
    grid = np.linspace(-3, 3, 1000)
    _, recon = reconstruct(n); generated = generate(n)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
    ax[0].plot(grid, pdf(grid)); ax[0].fill_between(grid, pdf(grid), alpha=.35); ax[0].set_title("target distribution")
    ax[1].hist(recon, bins=120, range=(-3,3), density=True); ax[1].plot(grid, pdf(grid), "r--", lw=1); ax[1].set_title("reconstruction")
    ax[2].hist(generated, bins=120, range=(-3,3), density=True); ax[2].plot(grid, pdf(grid), "r--", lw=1); ax[2].set_title("generated from z ~ N(0,I)")
    for a in ax: a.set_xlim(-3, 3); a.set_ylim(0, 0.5); a.grid(alpha=.2)
    fig.suptitle(title)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_gif(frame_paths, gif_path="vae_training.gif", seconds=10):
    from PIL import Image
    duration = int(seconds * 1000 / len(frame_paths))
    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    for frame in frames: frame.close()


steps, batch_size = int(os.getenv("MINI_VAE_STEPS", "100000")), 2048
frame_dir, frame_paths = Path("vae_frames"), []
if args.save_frames: frame_dir.mkdir(exist_ok=True)

for t in range(steps + 1):
    x = torch.from_numpy(sample(batch_size))
    x_hat, z_mu, z_logvar = vae(x)
    recon = F.mse_loss(x_hat, x)
    kl = .5 * (z_mu.square() + z_logvar.exp() - 1 - z_logvar).mean()
    loss = recon + beta * kl

    opt.zero_grad(); loss.backward(); opt.step()
    if t % 3000 == 0: print(f"step {t:5d}  recon {recon.item():.3f}  kl {kl.item():.3f}  loss {loss.item():.3f}")
    if args.save_frames and t % args.frame_every == 0:
        frame_path = frame_dir / f"step_{t:06d}.png"
        save_plot(frame_path, title=f"VAE training, step={t}", n=20000)
        frame_paths.append(frame_path)

save_plot("vae_result.png")
if args.save_frames and frame_paths:
    save_gif(frame_paths)
    print(f"saved {len(frame_paths)} frames to {frame_dir}/ and gif to vae_training.gif")
print("saved final plot to vae_result.png")
plt.show()
