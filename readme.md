# Depth-Based Image Transparency Editor

## Description

Python tool with a Tkinter interface for creating transparent images. Leverages AI depth estimation (Hugging Face Transformers DPT model) to determine transparency based on a user-defined threshold. Saves output PNGs.

## Features

* Graphical User Interface (GUI) built with Tkinter.
* Uses Intel's DPT (Dense Prediction Transformers) model via the Hugging Face `transformers` library for monocular depth estimation.
* Interactively adjust the transparency threshold using a slider.
* Pixels with depth values *below* the threshold become transparent.
* Displays the original image and a preview of the transparent output.
* Saves the resulting transparent image as a PNG file.
* Optionally saves the generated depth map as a grayscale PNG file.
* Attempts to use CUDA (GPU) for faster processing if available, otherwise falls back to CPU.

## Requirements

The following Python libraries are required[cite: 1]:

* torch>=1.8.0
* transformers>=4.0.0
* numpy>=1.18.0
* Pillow>=8.0.0
* tkinter (usually included with Python standard library)

## Installation

1.  **Clone the repository (or download the code files):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note:* Ensure your `torch` installation is compatible with your system (CPU-only or specific CUDA version if you have an NVIDIA GPU). See [PyTorch installation instructions](https://pytorch.org/get-started/locally/) if needed.

## Usage

Run the main script from your terminal, providing the path to the input image:

```bash
python main.py path/to/your/image.jpg