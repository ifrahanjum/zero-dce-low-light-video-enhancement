# 🌙 Zero-DCE: Zero-Reference Low-Light Video Enhancement


> **Live Demo:** [Check out the web app here!](https://zero-dce-low-light-video-enhancement-khe2awlud8vtvl8rgtfkhb.streamlit.app/) 

An end-to-end, zero-reference low-light video enhancement pipeline built with PyTorch, OpenCV, and Streamlit. This project processes dark images and videos in real-time without ever needing paired "ground-truth" datasets for training.

---

## ✨ Features

- **Zero-Reference Learning:** Trained entirely without paired normal-light/low-light images.
- **Full Video Support:** Processes `.mp4` and `.mov` files frame-by-frame with smooth temporal consistency.
- **Automated Web-Ready Codec Conversion:** Automatically uses FFmpeg subprocesses to convert OpenCV's native `mp4v` output into browser-compatible `H.264` format.
- **Side-by-Side Comparison:** Dynamically stitches original and enhanced frames together for easy visual evaluation.
- **Cloud-Optimized Memory:** Engineered with dynamic downscaling and aggressive garbage collection to prevent Out-Of-Memory (OOM) crashes on constrained cloud servers.

---

## 🧠 Under the Hood: The Architecture

Unlike traditional models that generate pixels from scratch, **DCE-Net** acts as an automated photo editor, predicting image-specific exposure curves.

1. **The Network:** A highly efficient, 7-layer Convolutional Neural Network (CNN) with symmetrical skip connections. Total size: **~0.08 Million parameters**.
2. **The Parameter Map:** The network outputs a 24-channel parameter map, representing 8 iterations of 3 color channels (RGB).
3. **The Physics Equation:** The light enhancement is applied using a pixel-wise, differentiable quadratic curve:
   `x_new = x + A(x^2 - x)`
4. **Unsupervised Loss Functions:** The model learns through four mathematically enforced constraints:
   - **Exposure Loss:** Pushes patch-mean luminance toward a well-exposed target of 0.6.
   - **Color Constancy Loss:** Minimizes differences between spatial RGB means to prevent color casts.
   - **Spatial Consistency Loss:** Preserves contrast gradients between neighboring regions.
   - **Illumination Smoothness Loss:** Applies Total Variation (TV) to ensure lighting adjustments are smooth and natural.

---

## 📂 Repository Structure

```text
Zero-DCE-Web-App/
│
├── core/
│   ├── model.py             # DCENet Architecture & Curve application math
│   └── losses.py            # Custom physics-based loss functions
│
├── app.py                   # Streamlit deployment frontend & OpenCV video pipeline
├── train.py                 # Modular training execution script
├── zero_dce_best-updated.pth # Pre-trained model weights
├── requirements.txt         # Python dependencies
└── packages.txt             # System-level dependencies (FFmpeg) for cloud deployment
```

## 🚀 Run It Locally
To run this project on your own machine:

1. Clone the repository:
```Bash
git clone [https://github.com/aaquib1303/zero-dce-low-light-video-enhancement](https://github.com/aaquib1303/zero-dce-low-light-video-enhancement)
cd zero-dce-low-light-video-enhancement
```

2. Install dependencies:
```Bash
pip install -r requirements.txt
```
(Note: You will also need FFmpeg installed on your system path for video processing).

3. Boot the application:
```Bash
python -m streamlit run app.py
```
Built for real-time inference and deployed via Streamlit Community Cloud.


