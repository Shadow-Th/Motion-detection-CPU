# Real-Time Motion Detection & Frame Differencing Pipeline

A high-performance computer vision and video processing pipeline built in Python to detect, isolate, and track dynamic movement across video frames with minimal CPU overhead.

---

## **System Architecture & Core Mechanisms**

Standard motion detection scripts often suffer from high latency, noise sensitivity, and heavy compute costs. This engine implements an optimized multi-stage processing pipeline designed for real-time efficiency on CPU hardware.

### **1. Frame Differencing & Preprocessing**
* **Grayscale Conversion & Gaussian Blur:** Incoming video frames are converted to grayscale to reduce dimensionality, followed by a spatial Gaussian blur to filter out high-frequency pixel noise and lighting flicker.
* **Temporal Differencing:** Computes absolute differences between consecutive frames to isolate moving pixels from static background elements.

### **2. Thresholding & Contour Extraction**
* **Binarization:** Applies an adaptive binary threshold to separate significant motion from minor background variance.
* **Morphological Operations:** Utilizes erosion and dilation algorithms to remove false-positive pixel fragments and close broken contours around moving objects.
* **Bounding Box Tracking:** Identifies valid object contours and maps bounding coordinates for real-time tracking and logging.

---

## **Technical Stack**
* **Core Language:** Python (3.10+)
* **Computer Vision & Math:** OpenCV (`cv2`), NumPy, SciPy
* **Performance:** Optimized vector operations and memory management for low-latency CPU execution
