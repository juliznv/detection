import cv2
import time

cap=cv2.VideoCapture(0)
cap.set(3,900)
cap.set(4,900)

while(cap.isOpened()):
    ret_flag, Vshow = cap.read()
    cv2.imshow('Capture', Vshow)
    k=cv2.waitKey(1)
    if k==ord('s'):
        print('222222')
        print(cap.get(3))
        print(cap.get(4))
    elif k==ord('q'):
        print('完成')
        break
    print('摄像头捕获成功')
    # pass
    # time.sleep(1)
cap.release()
cv2.destoryAllWindows()