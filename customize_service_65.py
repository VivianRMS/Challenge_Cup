from PIL import Image
import copy
import sys
import traceback
import os
import numpy as np
import time
import cv2
import re
import json
from input_reader import InputReader
from tracker import Tracker
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from MAR import mouth_upper_ratio
from MAR import mouth_lower_ratio


from models.experimental import attempt_load
from utils1.general import check_img_size
from tempfile import NamedTemporaryFile
from utils1.torch_utils import TracedModel
from detect import detect
#from model_service.pytorch_model_service import PTServingBaseService


#class fatigue_driving_detection(PTServingBaseService):
class fatigue_driving_detection():
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path

        self.capture = 'test.mp4'

        self.frame_num=0

        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.first = True

        self.standard_pose = [180, 40, 80]
        self.look_around_frame = 0
        self.eyes_closed_frame = 0
        self.mouth_open_frame = 0
        self.use_phone_frame = 0
        # lStart, lEnd) = (42, 48)
        self.lStart = 42
        self.lEnd = 48
        # (rStart, rEnd) = (36, 42)
        self.rStart = 36
        self.rEnd = 42
        # (mStart, mEnd) = (49, 66)
        self.mStart = 49
        self.mEnd = 66
        self.EYE_AR_THRESH = 0.15
        self.MOUTH_AR_THRESH = 0.6
        self.frame_3s = self.fps * 3
        self.face_detect = 0

        self.weights = "best.pt"
        self.imgsz = 640

        self.device = 'cpu'  # 大赛后台使用CPU判分

        model = attempt_load(model_path, map_location=self.device)
        self.stride = int(model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

        self.model = TracedModel(model, self.device, self.imgsz)


        self.need_reinit = 0
        self.failures = 0

        self.tracker = Tracker(self.width, self.height, threshold=None, max_threads=4, max_faces=4,
                          discard_after=10, scan_every=3, silent=True, model_type=3,
                          model_dir=None, no_gaze=False, detection_threshold=0.6,
                          use_retinaface=0, max_feature_updates=900,
                          static_model=True, try_hard=False)

        # self.temp = NamedTemporaryFile(delete=False)  # 用来存储视频的临时文件

    def _preprocess(self, data):
        # preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        with open(self.capture, 'wb') as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                    # self.capture = self.temp.name  # Pass temp.name to VideoCapture()
                except Exception:
                    return {"message": "There was an error processing the file"}
        return 'ok'

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        if visualize: 
            print(data)
        result = {"result": {"category": 0, "duration": 6000}}

        self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps)
        source_name = self.input_reader.name
        now = time.time()

        self.look_around_frame = 0
        self.eyes_closed_frame = 0
        self.mouth_open_frame = 0
        self.use_phone_frame = 0

        self.look_around_failure = 0
        self.eyes_closed_failure = 0
        self.mouth_open_failure = 0
        self.use_phone_failure = 0

        self.failure_threshold_normal = 5 #Reset to 0 when there are three failures! 
        self.failure_threshold_phone = 2 #Reset to 0 when there are three failures! 

        self.look_around_add_flag = False

        self.cut_range = []
        self.cut_picture_width = 0

        idx = 0
        while self.input_reader.is_open():
            self.look_around_add_flag = False # reset! 
            if visualize:
                print("Inferring No.{0} image".format(idx))
            idx += 1
            
            if not self.input_reader.is_open() or self.need_reinit == 1:
                self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps, use_dshowcapture=False, dcap=None)
                if self.input_reader.name != source_name:
                    print(f"Failed to reinitialize camera and got {self.input_reader.name} instead of {source_name}.")
                    # sys.exit(1)
                self.need_reinit = 2
                time.sleep(0.02)
                continue
            if not self.input_reader.is_ready():
                time.sleep(0.02)
                continue

            ret, frame = self.input_reader.read()

            self.need_reinit = 0

            try:
                #print("IDX: {}".format(idx))
                if idx == 1:
                    bbox = detect(self.model, frame, self.stride, self.imgsz)
                    #print(bbox)
                    choose_id = 0
                    now_id = 0
                    previous = 0
                    flag = False
                    for box in bbox:
                        if box[0] == 0:
                            #print(box[1])
                            if (box[1][2] * box[1][3] * (box[1][0] + box[1][2]/2) > previous):
                                flag = True
                                previous = box[1][2] * box[1][3] * (box[1][0] + box[1][2]/2)
                                choose_id = now_id
                        now_id += 1
                    if flag:
                        self.cut_range = bbox[choose_id][1]


                if frame is not None:
                    self.frame_num += 1

                    # in case the image is not in the right size
                    frame = cv2.resize(frame, (self.width, self.height))
                    # 剪裁主驾驶位
                    if self.cut_range == []:
                        frame = frame[:, 700:1920, :]
                    else:
                        left_most = int((self.cut_range[0] - self.cut_range[2]/2) * (0.81) * 1920)
                        right_most = int((self.cut_range[0] + self.cut_range[2]/2) * (1.2) * 1920)
                        self.cut_picture_width = right_most - left_most
                        down_most = int((self.cut_range[1] - self.cut_range[3]/2) * (0.7) * 1920)
                        up_most = int((self.cut_range[1] + self.cut_range[3]/2) * (1.3) * 1920)
                        if left_most > 550:
                            
                            frame = frame[:, left_most:right_most, :]
                        else:
                            frame = frame[:, 700:1920, :]


                    #whether_detect = False
                    
                    faces = self.tracker.predict(frame)
                    if len(faces) > 0:

                        face_num = 0
                        max_x = 0
                        for face_num_index, f in enumerate(faces):
                            if max_x <= f.bbox[3]:
                                face_num = face_num_index
                                max_x = f.bbox[3]

                        f = faces[face_num]
                        f = copy.copy(f)

                        # for visualization of face counter points 
                        if visualize:
                            for pt in f.lms:
                                cv2.circle(
                                    frame,
                                    (int(pt[1]), int(pt[0])),
                                    3,
                                    (255, 255, 255),
                                    1)

                        # 检测是否转头
                        if np.abs(self.standard_pose[0] - f.euler[0]) >= 60 or np.abs(self.standard_pose[1] - f.euler[1]) >= 60 or \
                                np.abs(self.standard_pose[2] - f.euler[2]) >= 60:
                            self.look_around_frame += 1
                            self.look_around_failure = 0
                            self.look_around_add_flag = True
                            if visualize:
                                print("MEMEMEMEME")
                                print(">>>>>-------Look around: {}".format(self.look_around_frame))
                            if self.look_around_frame >= self.frame_3s:
                                result['result']['category'] = 4
                                break
                            elif self.look_around_frame >= self.frame_3s/3 and self.mouth_open_frame >= self.look_around_frame:
                                print("RESET!!!")
                                self.mouth_open_frame = 0
                            #whether_detect = True

                        # 检测是否闭眼
                        # extract the left and right eye coordinates, then use the
                        # coordinates to compute the eye aspect ratio for both eyes
                        leftEye = f.lms[self.lStart:self.lEnd]
                        rightEye = f.lms[self.rStart:self.rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        # average the eye aspect ratio together for both eyes
                        ear = (leftEAR + rightEAR) / 2.0

                        if ear < self.EYE_AR_THRESH:
                            self.eyes_closed_frame += 1
                            self.eyes_closed_failure = 0
                            if visualize:
                                print("-----------------Close eye: {}".format(self.eyes_closed_frame))
                            if self.eyes_closed_frame >= self.frame_3s:
                                result['result']['category'] = 1
                                break
                        elif self.eyes_closed_failure >= self.failure_threshold_normal:
                            self.eyes_closed_frame = 0
                        else: 
                            self.eyes_closed_failure += 1
                            #whether_detect = True
                        # print(ear, eyes_closed_frame)

                        # 检测是否张嘴
                        mar = mouth_aspect_ratio(f.lms)
                        mur = mouth_upper_ratio(f.lms)
                        mlr = mouth_lower_ratio(f.lms)

                        if mar > self.MOUTH_AR_THRESH and mur < 1 and mlr < 1:
                            self.mouth_open_frame += 1
                            self.mouth_open_failure = 0
                            if visualize:
                                print("OOOOO--------Mouth_open: {}".format(self.mouth_open_frame))
                            if self.mouth_open_frame >= self.frame_3s:
                                result['result']['category'] = 2
                                break
                            elif self.mouth_open_frame >= self.frame_3s/2: 
                                self.eyes_closed_frame = 0
                        elif self.mouth_open_failure >= self.failure_threshold_normal:
                            self.mouth_open_frame = 0
                        else:
                            self.mouth_open_failure += 1
                        
                        if mar > self.MOUTH_AR_THRESH and (mur >= 1 or mlr >= 1):
                            self.look_around_frame += 1
                            self.look_around_failure = 0
                            self.look_around_add_flag = True
                            if visualize:
                                print(">>>>>-------Look around: {}".format(self.look_around_frame))
                            if self.look_around_frame >= self.frame_3s:
                                result['result']['category'] = 4
                                break
                            elif self.look_around_frame >= self.frame_3s/3 and self.mouth_open_frame >= self.look_around_frame:
                                print("RESET!!!")
                                self.mouth_open_frame = 0
                            #whether_detect = True
                            
                            #whether_detect = True
#                         print(mar)

#                         print(len(f.lms), f.euler)
                    else:
                        self.look_around_frame += 1
                        self.look_around_failure = 0
                        self.look_around_add_flag = True
                        if visualize:
                            print(">>>>>>>>---------Look around: {}".format(self.look_around_frame))
                        if self.look_around_frame >= self.frame_3s:
                            result['result']['category'] = 4
                            break
                        elif self.look_around_frame >= self.frame_3s/3 and self.mouth_open_frame >= self.look_around_frame:
                            self.mouth_open_frame = 0
                            print("RESET!!!")
                        # ???


                    if self.look_around_add_flag == False: 
                        self.look_around_failure += 1
                    if self.look_around_failure >= self.failure_threshold_normal: 
                        self.look_around_frame = 0

                        # 检测驾驶员是否接打电话 以及低头的人脸

                    if self.frame_num % 3 == 1:                    
                        bbox = detect(self.model, frame, self.stride, self.imgsz)
                            # print(results)

                        use_phone_flag = False
                        for box in bbox:
                            #if box[0] == 0:
                            #self.face_detect = 1
                            if box[0] == 1:
                                self.use_phone_frame += 1
                                self.use_phone_failure = 0
                                self.use_phone_flag = True
                                if visualize:
                                    print("PPPPPPPPPP--------------Use Phone: {}".format(self.use_phone_frame))
                                break
                        
                        if self.use_phone_frame >= self.frame_3s/6 - 3 and self.look_around_frame >= self.frame_3s/3: 
                            self.look_around_frame = 0
                            print("RESET!")
                
                        if self.use_phone_failure >= self.failure_threshold_phone: 
                            self.use_phone_frame = 0
                        else: 
                            self.use_phone_failure += 1
                        
                        # for visualization of detection bounding box
                        if visualize:
                            x1 = int((box[1][0]-box[1][2]/2)*(self.cut_picture_width))
                            y1 = int((box[1][1]-box[1][3]/2)*self.height)
                            x2 = int((box[1][0]+box[1][2]/2)*(self.cut_picture_width))
                            y2 = int((box[1][1]+box[1][3]/2)*self.height)
                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (255, 255, 255),
                                1)
                            
                    if self.use_phone_frame >= (self.frame_3s/3)-2:
                        result['result']['category'] = 3
                        break


                    result['result']['category'] = 0

                    self.failures = 0

                    # opencv image showing code
                    if visualize:
                        cv2.imshow("frame", frame)
                        key = cv2.waitKey(50)


                else:
                    break
            except Exception as e:
                if e.__class__ == KeyboardInterrupt:
                    print("Quitting")
                    break
                traceback.print_exc()
                self.failures += 1
                if self.failures > 30:   # 失败超过30次就默认返回
                    break
                    
            del frame
        final_time = time.time()
        duration = int(np.round((final_time - now) * 1000))
        result['result']['duration'] = duration
        return result

    def _postprocess(self, data):
        os.remove(self.capture)
        return data


if __name__ == "__main__":

    mode = "Exam"

    if mode == "Examine":
        # Set if you want to visualize the image
        visualize = True

        # First set the location of the model
        folder_name = "/fatigue_driver/"
        detector = fatigue_driving_detection("aaa", folder_name+"best.pt")

        # Second set the location of the video
        file_name = folder_name+"Fatigue_driving_detection_video/night_man_001_40_1.mp4"
        
        # data is not used acuatly
        data = {
            0: {
                file_name : file_name
            }
        }

        # pass the file_name directly to the capture variable to read
        detector.capture = file_name
        result = detector._inference(data)

        print("Result of ", file_name, "is below:")
        print(result)
    
    else: 
        visualize = False
        dir_folder_path = "Fatigue_driving_detection_video"

        # First set the location of the model
        folder_name = "/fatigue_driver/"
        detector = fatigue_driving_detection("aaa", folder_name+"best.pt")

        all_false_file = []
        length = 0

        for filename in os.listdir(dir_folder_path):
            length += 1
            if length <= 75:
                image_name = os.path.join(dir_folder_path, filename)
                data = {
                        0: {
                            image_name : image_name
                            }
                        }

                # pass the file_name directly to the capture variable to read
                detector.capture = image_name
                result = detector._inference(data)

                print("Result of ", filename, "is below:")
                print(result)
                
                patternstr = r"_\d{2}_"
                match = re.search(patternstr, filename)
                predict_result = []
                true_result = []

                if match:
                    matchstr = match.group()
                    predict_result.append(result["result"]["category"])

                    if matchstr[2] == "1" or matchstr[1:3] == "00":
                        true_result.append(0)
                        print(result["result"]["category"] ==  0)
                        if not(result["result"]["category"] ==  0):
                            all_false_file.append(filename)
                    else:
                        true_result.append(int(matchstr[1]))
                        print(result["result"]["category"] == int(matchstr[1]))
                        if not(result["result"]["category"] ==  int(matchstr[1])):
                            all_false_file.append(filename)
        
        with open("result1.txt", "w") as file:
            file.write("All video numbers: {}\n".format(length))
            file.write("False video numbers: {}\n".format(len(all_false_file)))
            for item in all_false_file:
                file.write(item + "\n")

            # Calculate F1 score
            file.write("----------------F1 score----------------\n")

            predict_result = np.array(predict_result)
            true_result = np.array(true_result)
            categories = [0, 1, 2, 3, 4]
            F1_score_each = []
            for category in categories:
                TP = np.sum((predict_result == true_result) == (predict_result == category))
                FP = np.sum((predict_result != true_result) == (predict_result == category))
                FN = np.sum((predict_result == true_result) == (true_result == category))
                Precision = float(TP) / (TP + FP)
                Recall = float(TP) / (TP + FN)
                F1_score_temp = 2*Precision*Recall/(Precision + Recall)
                F1_score_each.append(F1_score_temp)
                file.write("For categories {}, the F1-score is {}\n".format(category, F1_score_temp))
            F1_score_each.fillna(0)
            F1_score = sum(F1_score_each) / 5.0
            file.write("The average F1_score is \n")
            file.write(str(F1_score) + "\n")


