# import cv2
# import numpy as np

# count = cv2.cuda.getCudaEnabledDeviceCount()
# if count == 0:
#     print("No GPU detected, falling back to CPU.")

# def detect_motion_gpu():
#     cap = cv2.VideoCapture(0)
#     fgbg = cv2.cuda.createBackgroundSubtractorMOG2()
#     stream = cv2.cuda.Stream()
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         gpu_frame = cv2.cuda_GpuMat()
#         gpu_frame.upload(frame)
#         gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY, stream=stream)
#         gpu_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (7, 7), 0.5)
#         gpu_blur_result = gpu_blur.apply(gpu_gray, stream=stream)
#         gpu_mask = fgbg.apply(gpu_blur_result, learningRate=0.01, stream=stream)
#         host_mask = gpu_mask.download()
#         _, thresh = cv2.threshold(host_mask, 200, 255, cv2.THRESH_BINARY)
#         cv2.imshow('Motion Detection (GPU Accelerated)', thresh)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     detect_motion_gpu()

import cv2

def detect_motion_cpu():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    print("Starting Motion Detection... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        fg_mask = fgbg.apply(blur)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Motion Mask (White = Motion)', thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_motion_cpu()