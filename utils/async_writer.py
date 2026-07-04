"""
异步图像写入器

在后台线程中写入图像，避免阻塞主循环。
"""

import cv2
import numpy as np
import queue
import threading


class AsyncLivePreviewWriter:
    """
    后台线程图像写入器。

    Usage
    -----
    >>> writer = AsyncLivePreviewWriter(max_queue_size=4)
    >>> writer.start()
    >>> writer.enqueue('output/frame_001.jpg', frame)
    >>> writer.close()
    """

    def __init__(self, max_queue_size: int = 4):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def enqueue(self, path: str, image: np.ndarray):
        """添加写入任务，队列满时丢弃最旧的"""
        try:
            self.queue.put_nowait((path, image))
        except queue.Full:
            try:
                self.queue.get_nowait()  # 丢弃最旧
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((path, image))
            except queue.Full:
                pass

    def _run(self):
        while not self.stop_event.is_set():
            try:
                path, image = self.queue.get(timeout=0.5)
                cv2.imwrite(path, image)
            except queue.Empty:
                continue

    def close(self, timeout: float = 5.0):
        # 先设停止信号，但让线程有机会处理完剩余帧
        self.stop_event.set()
        # 排空队列中剩余的帧
        while not self.queue.empty():
            try:
                path, image = self.queue.get_nowait()
                cv2.imwrite(path, image)
            except queue.Empty:
                break
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
