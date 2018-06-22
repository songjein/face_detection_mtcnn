from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import os
import time

import cv2 
from mtcnn.mtcnn import MTCNN

# tf warning log OFF 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

"""
    Global Constant
"""
RESIZE_THRES = 500 # resize 기준 너비
CONFIDENCE_THRES = 0.85 # 이하의 confidence를 가지는 얼굴들은 제거한다
RESULT_KEY = "RESULT" # subprocess의 응답 딕셔너리(return_dict)의 Key 

"""
    Subprocess가 MTCNN 객체를 공유하기 위한 수단
    * 참조 https://docs.python.org/3/library/multiprocessing.html
"""
class CustomManager(BaseManager):
    pass
CustomManager.register("MTCNN", MTCNN)

cm = CustomManager()
cm.start()
detector = cm.MTCNN()

"""
    얼굴 인식을 수행하고 얼굴 영역을 리턴
    여러 얼굴이 인식될 경우 하나의 박스로 합친다

    Params
    - imageBinary: 이미지 바이너리 데이터 (BGR 이미지라 가정)
    - timeout: 양의 정수형, 단위는 밀리세컨드
               기본값 300
    - excludeFaceRate: 0~1 사이 실수, 얼굴인식 영역이 한개있고 이미지 크기와 동일할 경우 1
                       얼굴 영역이 excludeFaceRate 이하인 경우 측정 포함 x
                       입력값 None 일경우 사용 x
    Return
    - [faceAreaMinX, faceAreaMinY, faceAreaMaxX, faceAreaMaxy]

    * process를 이용한 timeout 구현: https://dreamix.eu/blog/webothers/timeout-function-in-python-3
"""
def getFaceArea(imageBinary, timeout_ms=300, excludeFaceRate=None):
    manager = Manager()
    return_dict = manager.dict()

    fd_process = Process(target=face_detection, args=(imageBinary, excludeFaceRate, return_dict, detector))
    fd_process.start()

    timeout_sec = timeout_ms * 0.001
    fd_process.join(timeout=timeout_sec) 
    fd_process.terminate()

    return return_dict[RESULT_KEY]

"""
    getFaceArea에서 불리는 subprocess로 얼굴 인식을 수행한다	

    Params
    - return_dict: 부모 프로세스와 공유되는 딕셔너리이다. RESULT_KEY로 접근
    - detector   : 부모 프로세스와 공유되는 MTCNN객체

    Returns
    - 공유 딕셔너리에 얼굴인식 박스 담김
"""
def face_detection(image, excludeFaceRate, return_dict, detector):
    resize_ratio = 1 
    return_dict[RESULT_KEY] = None 

    image_width  = image.shape[1]
    image_height = image.shape[0]
    if image_width > RESIZE_THRES:
        resize_ratio = RESIZE_THRES / image_width 
        wh_option = (RESIZE_THRES, int(image_height*resize_ratio))
        image = cv2.resize(image, wh_option)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_image)
    if not faces: return

    faces = remove_low_confidence(faces)
    box = merge_faces_as_box(faces)
    restore_ratio = 1 / resize_ratio 
    box = [i * restore_ratio for i in box] 
    return_dict[RESULT_KEY] = box 

    box_width  = box[2] - box[0]
    box_height = box[3] - box[1]
    fd_area_ratio = (box_width * box_height) / (image_width * image_height)
    if excludeFaceRate and (fd_area_ratio < excludeFaceRate):
        return_dict[RESULT_KEY] = None	

"""
    얼굴인식의 confidence가 낮은 것(노이즈)은 제거한다
"""
def remove_low_confidence(faces, thres=CONFIDENCE_THRES):
    return [f for f in faces if f['confidence'] > thres]

"""
    여러개의 얼굴 영역을 하나의 박스로 합친다
"""
def merge_faces_as_box(faces):
    # most top y pos
    mty = 9999
    # most left x pos
    mlx = 9999
    # most bottom y pos
    mby = -1
    # most right x pos
    mrx = -1

    for f in faces:
        face_box = f["box"] # [x, y, width, height]
        mty = min(mty, face_box[1])
        mlx = min(mlx, face_box[0])
        mby = max(mby, face_box[1] + face_box[3])
        mrx = max(mrx, face_box[0] + face_box[2])
    
    return (mlx, mty, mrx, mby)


"""
    테스트
"""
if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join
    import imghdr
    IMAGE_PATH = '/home/jeinsong/example'
    files = [join(IMAGE_PATH, f) for f in listdir(IMAGE_PATH) 
        if isfile(join(IMAGE_PATH, f)) and imghdr.what(join(IMAGE_PATH, f))]
    diffs = []
    for idx, filename in enumerate(files):
        image = cv2.imread(filename)
        stime = time.time()
        res = getFaceArea(image, 300, None)
        diffs.append(time.time() - stime)
        print (res)
        print ("-------------------------------------------")
    
    print(diffs)
