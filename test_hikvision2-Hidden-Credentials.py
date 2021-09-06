import cv2
import queue
import threading

q = queue.Queue()


def receive():
    print("start Reveive")
    cap = cv2.VideoCapture("rtsp://USR_name:PASSWORD@192.168.1.74:554/doc/page/preview.asp")
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)
        if not p2.is_alive():
            break


def display():
    print("Start Displaying")
    while True:
        if not q.empty():
            frame = q.get()
            cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    p1 = threading.Thread(target=receive)
    p1.deamon = True
    p2 = threading.Thread(target=display)
    p2.deamon = True
    p1.start()
    p2.start()
