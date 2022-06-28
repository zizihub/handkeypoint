# MNN Engine
### 当前模型

| Name                           | Checkpoints        | Scripts                     |
| ------------------------------ | ------------------ | --------------------------- |
| BlazeFace                      | lite_blazeface.mnn | Face_detection.py           |
| BlazePalm                      | palm_detection.mnn | Palm_detection.py           |
| BlazePalm + HandLandmarks      | hand_landmark.mnn  | Hand_gesture_landmark.py    |
| BlazePalm + GestureRecognition | Repvgg-a0-hgr.mnn  | Hand_gesture_recognition.py |



### 使用规范

**examples/**
推理脚本路径

**models/mnn/**
mnn模型checkpoint路径

**src/**
模型依赖文件

**tools/**
模型转换工具