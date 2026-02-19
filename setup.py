from setuptools import setup, find_packages

setup(
    name="omnilatent",
    version="0.1.0",
    description="OmniLatent: All-to-all multimodal AI with Latent Neural Hooks",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "torchvision>=0.16.0",
        "einops>=0.7.0",
    ],
)
