from setuptools import setup, find_packages

setup(
    name="omnilatent",
    version="0.1.0",
    description="OmniLatent: All-to-all multimodal AI with Latent Neural Hooks",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "coco": ["torchvision>=0.16.0", "Pillow>=9.0"],
        "audio": ["torchaudio>=2.1.0"],
        "video": ["torchaudio>=2.1.0", "torchvision>=0.16.0"],
        "pdf": ["PyMuPDF>=1.23.0"],
        "all": [
            "torchvision>=0.16.0",
            "torchaudio>=2.1.0",
            "Pillow>=9.0",
            "PyMuPDF>=1.23.0",
        ],
        "dev": ["pytest>=7.0", "ruff>=0.1.0"],
    },
)
