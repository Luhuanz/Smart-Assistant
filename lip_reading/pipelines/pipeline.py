#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import torch
import pickle
from configparser import ConfigParser

from pipelines.model import AVSR  #pipelines/model.py 里读 AVSR 这个类
from pipelines.data.data_module import AVSRDataLoader


class InferencePipeline(torch.nn.Module):
    def __init__(self, config_filename='configs/LRS3_V_WER19.1.ini', detector="mediapipe", face_track=False, device="cuda:0"):
        super(InferencePipeline, self).__init__()
        assert os.path.isfile(config_filename), f"config_filename: {config_filename} does not exist."

        config = ConfigParser()
        config.read(config_filename)

        # modality configuration modality=video
        modality = config.get("input", "modality")

        self.modality = modality
        # data configuration
        input_v_fps = config.getfloat("input", "v_fps")
        model_v_fps = config.getfloat("model", "v_fps")

        # model configuration
        model_path = config.get("model","model_path")
        model_conf = config.get("model","model_conf")

        # language model configuration
        rnnlm = config.get("model", "rnnlm")
        rnnlm_conf = config.get("model", "rnnlm_conf")
        penalty = config.getfloat("decode", "penalty")
        ctc_weight = config.getfloat("decode", "ctc_weight")
        lm_weight = config.getfloat("decode", "lm_weight")
        beam_size = config.getint("decode", "beam_size")

        self.dataloader = AVSRDataLoader(modality, speed_rate=input_v_fps/model_v_fps, detector=detector)
        #保证语言连贯、合理、纠错
        self.model = AVSR(modality, model_path, model_conf, rnnlm, rnnlm_conf, penalty, ctc_weight, lm_weight, beam_size, device)
        if face_track and self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from pipelines.detectors.mediapipe.detector import LandmarksDetector
                self.landmarks_detector = LandmarksDetector()
        else:
            self.landmarks_detector = None


    def process_landmarks(self, data_filename, landmarks_filename):
        #只有在你是 modality = video（纯视觉）或 audiovisual（听说同步）的时候，才会提 landmarks；
        if self.modality == "audio":
            return None
        if self.modality in ["video", "audiovisual"]:
            if isinstance(landmarks_filename, str):
                landmarks = pickle.load(open(landmarks_filename, "rb"))
            else:
                landmarks = self.landmarks_detector(data_filename)
            return landmarks


    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        #data.shape = [40, 1, 96, 96]  # 举个例子：40帧，1通道，96x96 视频 → 裁出嘴 → 变成一批标准化的灰度嘴图帧，打包成模型可以直接使用的输入张量。

        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript