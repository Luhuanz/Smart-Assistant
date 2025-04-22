# MIT License
#
# Copyright (c) 2025 Amanvir Parhar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import os
import time
import torch
import hydra
import cv2
import signal
from concurrent.futures import ThreadPoolExecutor
from pipelines.pipeline import InferencePipeline
from pynput import keyboard as pkb  # pip install pynput

class Lipread:
    def __init__(self, vsr_model: InferencePipeline):
        self.vsr_model = vsr_model
        self.recording = False
        self._stop = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25
        self._last_transcript = ""

        # 捕捉 Ctrl+C 结束
        signal.signal(signal.SIGINT, self._sigint_handler)

        # 全局监听空格键：按一次开始，按一次结束
        listener = pkb.Listener(on_press=self._on_key)
        listener.daemon = True
        listener.start()

    def _sigint_handler(self, signum, frame):
        print("\n[Lipread] SIGINT caught, stopping…")
        self._stop = True

    def _on_key(self, key):
        # 全局捕捉 空格 切换录制
        if key == pkb.Key.space:
            self.recording = not self.recording
            state = "STARTED" if self.recording else "STOPPED"
            print(f"[Lipread] Recording {state}")
        # 可以额外用 ESC 退出
        elif key == pkb.Key.esc:
            print("[Lipread] ESC pressed, stopping…")
            self._stop = True

    def perform_inference(self, video_path: str):
        # 只做唇读
        result = self.vsr_model(video_path)
        print(f"[Lipread] Transcript: {result}")
        os.remove(video_path)

    def start_webcam(self):
        # 自动探测可用摄像头
        cap = None
        for idx in range(5):
            tmp = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if tmp.isOpened():
                cap = tmp
                print(f"[Lipread] Using /dev/video{idx}")
                break
            tmp.release()
        if cap is None:
            raise RuntimeError("No usable camera found (tried indices 0–4)")

        # 主循环
        out = None
        frame_count = 0
        output_path = ""
        last_time = time.time()

        while not self._stop:
            now = time.time()
            if now - last_time < self.frame_interval:
                time.sleep(0.005)
                continue
            last_time = now

            ret, frame = cap.read()
            if not ret:
                continue
            # 压缩 + 灰度
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.frame_compression])
            gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

            if self.recording:
                if out is None:
                    output_path = f"{self.output_prefix}{int(now*1000)}.mp4"
                    h, w = gray.shape
                    out = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        self.fps,
                        (w, h),
                        False
                    )
                out.write(gray)
                frame_count += 1
            else:
                if out is not None:
                    out.release()
                    if frame_count >= self.fps * 2:
                        # 异步唇读
                        self.executor.submit(self.perform_inference, output_path)
                    else:
                        os.remove(output_path)
                    out = None
                    frame_count = 0

        # 结束时清理
        cap.release()
        if out:
            out.release()
        print("[Lipread] Camera released, exiting.")

    def get_transcript(self) -> str:
        return self._last_transcript


@hydra.main(config_path="hydra_configs", config_name="default")
def main(cfg):
    vsr = InferencePipeline(
        face_track=True
    )
    print("Model loaded successfully!")

    lp = Lipread(vsr)
    lp.start_webcam()


if __name__ == '__main__':
    main()
