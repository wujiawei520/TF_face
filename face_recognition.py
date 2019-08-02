import os
import cv2
import numpy as np
import tensorflow as tf
import face_train as myconv
import load_data as dataSet

IMGSIZE = 64

def testfromcamera(chkpoint):

    camera = cv2.VideoCapture(0)
    haarClassifier = cv2.CascadeClassifier('D:/workSpace/python3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    pathlabelpair, indextoname = dataSet.getUserLabel('./faceData')
    output = myconv.cnnLayer(len(pathlabelpair))
    predict = output
    saver = tf.train.Saver()
    color = (0, 255, 0)
    with tf.Session() as sess:

        saver.restore(sess, chkpoint)
        num = 1
        while True:

            if (num <= 1000):
                print('It`s your No.%s image.' % num)
                ok, frame = camera.read()
                if not ok:
                    break

                grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faceRects = haarClassifier.detectMultiScale(grey_img,
                                                        scaleFactor=1.2,
                                                        minNeighbors=5,
                                                        minSize=(32, 32))

                if len(faceRects) > 0:
                    for faceRect in faceRects:
                        x, y, w, h = faceRect
                        face = frame[y - 10:y + h + 10, x - 10:x + w + 10]
                        face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                        test_x = np.array([face])
                        test_x = test_x.astype(np.float32) / 255.0

                        res = sess.run([predict, tf.argmax(output, 1)],
                                       feed_dict={myconv.x_data: test_x,
                                       myconv.keep_prob:1.0})

                        print(res)
                        print(indextoname[res[1][0]])

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, indextoname[res[1][0]], (x, y-20),
                                    font, 1, (255, 0, 0), 2)
                        frame = cv2.rectangle(frame, (x - 10, y - 10),(x + w + 10, y + h + 10), color, 2)
                        num += 1
            
                cv2.imshow('test face', frame)
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    break
            else:
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    savepath = './model/face.ckpt'

    # 不存在模型则训练
    if os.path.exists(savepath + '.meta') is False:
        train_x, train_y, test_x, test_y = dataSet.getData('./faceData')
        myconv.train(train_x, train_y, test_x, test_y, savepath)

    # 识别人脸
    else:
        testfromcamera(savepath)