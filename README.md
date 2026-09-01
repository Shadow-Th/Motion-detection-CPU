# Real-Time Motion Detection Pipeline (CPU & CUDA Accelerated)

A robust computer vision pipeline built with Python and OpenCV that performs real-time background subtraction, noise reduction, and motion segmentation. It features both a CPU-bound implementation utilizing `BackgroundSubtractorMOG2` and an optional modular GPU-accelerated path via OpenCV CUDA streams (`cv2.cuda`).

---

## **System Architecture & Core Mechanisms**

The pipeline processes live video streams frame-by-frame, applying spatial filtering and statistical background modeling to isolate dynamic motion from static environments with minimal computational overhead.

### **1. Video Ingestion & Grayscale Conversion**
* **Stream Handling:** Captures live video feeds using `cv2.VideoCapture(0)`.
* **Dimensionality Reduction:** Converts each incoming BGR frame to single-channel grayscale (`cv2.COLOR_BGR2GRAY`) to streamline matrix computations and reduce downstream processing latency.

### **2. Spatial Filtering & Noise Suppression**
* **Gaussian Blurring:** Applies a $7 \times 7$ Gaussian kernel (`cv2.GaussianBlur`) to smooth out high-frequency pixel noise and lighting fluctuations, preventing false-positive motion triggers.

### **3. Background Subtraction & Mask Generation**
* **MOG2 Algorithm:** Implements Mixture of Gaussians (`cv2.createBackgroundSubtractorMOG2`) to model background pixels dynamically, tracking lighting changes and adapting to stationary objects over time.
* **Binarization:** Applies a strict binary threshold (`cv2.THRESH_BINARY`) on the foreground mask to isolate significant motion zones cleanly.

---

## **Technical Stack & Requirements**
* **Core Language:** Python (3.10+)
* **Computer Vision:** OpenCV (`cv2` with optional CUDA support modules)
* **Numerical Processing:** NumPy

---

## **Execution Instructions**

1. Ensure dependencies are installed:
   ```bash
   pip install opencv-python numpy
