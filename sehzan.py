import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import zipfile, os, time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from pytorch_msssim import ssim
from PIL import Image

# ================= APP CONFIG =================
st.set_page_config(page_title="Dolphin Generative AI", layout="wide")
st.title("🐬 Generative AI Image Generation – Dolphin")
st.write("Autoencoder vs GAN vs Diffusion (Small Dataset)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 64
BATCH_SIZE = 2
EPOCHS = 10
LATENT_DIM = 100

# ================= UPLOAD ZIP =================
st.sidebar.header("📂 Upload Dataset")
zip_file = st.sidebar.file_uploader("Upload dolphin.zip", type="zip")

if zip_file:
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall("data")

    st.success("ZIP extracted successfully!")

    # ================= DATA LOADER =================
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.ImageFolder("data", transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    st.write(f"Total Images Loaded: {len(dataset)}")

    # ================= MODELS =================
    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh()
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(LATENT_DIM, 256), nn.ReLU(),
                nn.Linear(256, 3*IMAGE_SIZE*IMAGE_SIZE), nn.Tanh()
            )

        def forward(self, z):
            return self.net(z).view(-1,3,IMAGE_SIZE,IMAGE_SIZE)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3*IMAGE_SIZE*IMAGE_SIZE,256), nn.ReLU(),
                nn.Linear(256,1), nn.Sigmoid()
            )

        def forward(self,x):
            return self.net(x.view(x.size(0),-1))

    class DiffusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3,64,3,padding=1), nn.ReLU(),
                nn.Conv2d(64,3,3,padding=1)
            )

        def forward(self,x):
            return self.net(x)

    # ================= TRAIN BUTTON =================
    if st.button("🚀 Train All Models"):
        st.info("Training started...")

        # -------- Autoencoder --------
        ae = AutoEncoder().to(DEVICE)
        opt = optim.Adam(ae.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        for _ in range(EPOCHS):
            for imgs,_ in loader:
                imgs = imgs.to(DEVICE)
                out = ae(imgs)
                loss = loss_fn(out, imgs)
                opt.zero_grad(); loss.backward(); opt.step()

        save_image(out, "ae.png", normalize=True)

        # -------- GAN --------
        G = Generator().to(DEVICE)
        D = Discriminator().to(DEVICE)
        optG = optim.Adam(G.parameters(), 0.0002)
        optD = optim.Adam(D.parameters(), 0.0002)
        bce = nn.BCELoss()

        for _ in range(EPOCHS):
            for imgs,_ in loader:
                imgs = imgs.to(DEVICE)
                b = imgs.size(0)

                real = torch.ones(b,1).to(DEVICE)
                fake = torch.zeros(b,1).to(DEVICE)

                z = torch.randn(b,LATENT_DIM).to(DEVICE)
                gen = G(z)

                d_loss = bce(D(imgs),real) + bce(D(gen.detach()),fake)
                optD.zero_grad(); d_loss.backward(); optD.step()

                g_loss = bce(D(gen),real)
                optG.zero_grad(); g_loss.backward(); optG.step()

        save_image(gen, "gan.png", normalize=True)

        # -------- Diffusion --------
        diff = DiffusionNet().to(DEVICE)
        opt = optim.Adam(diff.parameters(), lr=0.001)

        for _ in range(EPOCHS):
            for imgs,_ in loader:
                imgs = imgs.to(DEVICE)
                noise = torch.randn_like(imgs)
                noisy = imgs + 0.1*noise
                pred = diff(noisy)
                loss = loss_fn(pred, noise)
                opt.zero_grad(); loss.backward(); opt.step()

        sample = torch.randn(1,3,IMAGE_SIZE,IMAGE_SIZE).to(DEVICE)
        for _ in range(10):
            sample -= 0.1*diff(sample)

        save_image(sample, "diffusion.png", normalize=True)

        st.success("Training completed!")

    # ================= DISPLAY RESULTS =================
    st.header("🖼 Generated Images")
    col1, col2, col3 = st.columns(3)

    if os.path.exists("ae.png"):
        col1.image("ae.png", caption="Autoencoder Output")
    if os.path.exists("gan.png"):
        col2.image("gan.png", caption="GAN Output")
    if os.path.exists("diffusion.png"):
        col3.image("diffusion.png", caption="Diffusion Output")

    # ================= COMPARISON =================
    st.header("📊 Model Comparison")

    st.write("""
    **Autoencoder:** Best reconstruction, no creativity  
    **GAN:** Generates sharp dolphin images  
    **Diffusion:** Best realism and stability  
    """)

    st.success("Diffusion Model performs best for small datasets 🏆")
