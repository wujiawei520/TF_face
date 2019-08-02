import os
import cv2
import random
import numpy as np

def createdir(*args):
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

def relight(imgsrc, alpha=1, bias=0):
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc

def getFaceFromCamera(window_name, camera_id, pic_num, path_name, user_name):
    #创建文件夹
    createdir(path_name)

    #新建显示窗口
    cv2.namedWindow(window_name)

    #获取摄像头视频
    camera = cv2.VideoCapture(camera_id)

    #选用haar人脸识别分类器
    haarClassifier = cv2.CascadeClassifier('D:/workSpace/python3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')

    #识别出人脸后要画的边框的颜色(RGB格式)
    color = (0, 255, 0)

    num = 1
    while camera.isOpened():
        if (num <= pic_num):
            print('It`s your No.%s image.' % num)

            # 读帧
            ok, frame = camera.read()
            if not ok:
                break

            #将当前桢图像转换成灰度图像
            grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #人脸检测，1.2为图片缩放比例,5为最小有效检测数，32为图片最小尺寸
            faceRects = haarClassifier.detectMultiScale(grey_img,
                                                        scaleFactor=1.2,
                                                        minNeighbors=5,
                                                        minSize=(32, 32))

            #检测到人脸
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect

                    #将当前帧保存为图片
                    img_name = '%s/%d.jpg' % (path_name, num)
                    image = frame[y - 10:y + h + 10, x - 10:x + w + 10]

                    #改变图像的亮度，增加图像的对比性，可以识别不同光源下的人脸
                    image = relight(image, random.uniform(0.5, 1.5), random.randint(-50, 50))
                    image = cv2.resize(image, (64, 64))
                    cv2.imwrite(img_name, image)

                    #显示当前捕捉到了多少张人脸图片,putText(图片，文字，首字符左下坐标，字体，字体颜色，字体粗细)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, '%s: No.%d' % (user_name, num), (x, y-30),
                               font, 1, (255, 0, 0), 2)

                    #画出矩形框(图片，端点1，端点2，线段颜色，线段粗细)
                    img = cv2.rectangle(frame, (x - 10, y - 10),(x + w + 10, y + h + 10), color, 2)
                    num += 1

            #显示图片
            cv2.imshow(window_name, frame)

            #显示完一帧图像后等待的毫秒数,ESC退出
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            break

    #释放摄像头并销毁所有窗口
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    name = input('please input yourename: ')
    getFaceFromCamera(
        "catch face",
        0,
        200,
        os.path.join('./faceData', name),
        name)