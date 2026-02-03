"""
Screen Capture Module for Clash Royale Object Detection
Supports: Windows screen capture, emulator windows, and scrcpy streams
"""

import numpy as np
import cv2
from typing import Optional, Tuple
import time

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("Warning: mss not installed. Install with: pip install mss")


class ScreenCapture:
    """
    Cross-platform screen capture for Clash Royale detection.
    Supports full screen, region capture, and window capture.
    """
    
    def __init__(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
        fps_limit: int = 30
    ):
        """
        Initialize screen capture.
        
        Args:
            region: (left, top, width, height) to capture. None for full screen.
            target_size: (width, height) to resize captured frames.
            fps_limit: Maximum frames per second to capture.
        """
        self.region = region
        self.target_size = target_size
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / fps_limit
        self.last_capture_time = 0
        
        if not MSS_AVAILABLE:
            raise ImportError("mss library is required for screen capture")
        
        self.sct = mss.mss()
        
        # Get monitor info
        if region is None:
            # Use primary monitor
            self.monitor = self.sct.monitors[1]
        else:
            self.monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3]
            }
    
    def capture(self) -> np.ndarray:
        """
        Capture a single frame from the screen.
        
        Returns:
            numpy array of the captured frame in BGR format.
        """
        # Rate limiting
        current_time = time.time()
        elapsed = current_time - self.last_capture_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        
        # Capture screen
        screenshot = self.sct.grab(self.monitor)
        
        # Convert to numpy array (BGRA format)
        frame = np.array(screenshot)
        
        # Convert BGRA to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Resize if target size specified
        if self.target_size is not None:
            frame = cv2.resize(frame, self.target_size)
        
        self.last_capture_time = time.time()
        
        return frame
    
    def capture_continuous(self):
        """
        Generator that yields frames continuously.
        
        Yields:
            numpy array of captured frames in BGR format.
        """
        while True:
            yield self.capture()
    
    def get_monitor_info(self) -> dict:
        """Get information about available monitors."""
        return {
            "monitors": self.sct.monitors,
            "current": self.monitor
        }
    
    def set_region(self, left: int, top: int, width: int, height: int):
        """Update the capture region."""
        self.region = (left, top, width, height)
        self.monitor = {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        }
    
    def release(self):
        """Release resources."""
        self.sct.close()


class WindowCapture:
    """
    Capture a specific window by title (Windows only).
    Useful for capturing emulator windows.
    """
    
    def __init__(
        self,
        window_title: str,
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize window capture.
        
        Args:
            window_title: Partial or full title of the window to capture.
            target_size: (width, height) to resize captured frames.
        """
        self.window_title = window_title
        self.target_size = target_size
        self.hwnd = None
        
        try:
            import win32gui
            import win32ui
            import win32con
            self.win32gui = win32gui
            self.win32ui = win32ui
            self.win32con = win32con
            self._find_window()
        except ImportError:
            raise ImportError(
                "pywin32 is required for window capture. "
                "Install with: pip install pywin32"
            )
    
    def _find_window(self):
        """Find the window handle by title."""
        def callback(hwnd, windows):
            if self.win32gui.IsWindowVisible(hwnd):
                title = self.win32gui.GetWindowText(hwnd)
                if self.window_title.lower() in title.lower():
                    windows.append(hwnd)
            return True
        
        windows = []
        self.win32gui.EnumWindows(callback, windows)
        
        if windows:
            self.hwnd = windows[0]
        else:
            raise ValueError(f"Window with title '{self.window_title}' not found")
    
    def capture(self) -> np.ndarray:
        """
        Capture the window content.
        
        Returns:
            numpy array of the captured window in BGR format.
        """
        if self.hwnd is None:
            self._find_window()
        
        # Get window dimensions
        left, top, right, bottom = self.win32gui.GetClientRect(self.hwnd)
        width = right - left
        height = bottom - top
        
        # Get device context
        hwndDC = self.win32gui.GetWindowDC(self.hwnd)
        mfcDC = self.win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        
        # Create bitmap
        saveBitMap = self.win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)
        
        # Copy window content
        saveDC.BitBlt(
            (0, 0), (width, height),
            mfcDC, (0, 0),
            self.win32con.SRCCOPY
        )
        
        # Convert to numpy array
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        
        frame = np.frombuffer(bmpstr, dtype=np.uint8)
        frame = frame.reshape((height, width, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Cleanup
        self.win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        self.win32gui.ReleaseDC(self.hwnd, hwndDC)
        
        # Resize if needed
        if self.target_size is not None:
            frame = cv2.resize(frame, self.target_size)
        
        return frame
    
    def release(self):
        """Release resources."""
        pass


class ImageLoader:
    """
    Load images from files or video for testing/debugging.
    """
    
    def __init__(
        self,
        source: str,
        target_size: Optional[Tuple[int, int]] = None,
        loop: bool = True
    ):
        """
        Initialize image/video loader.
        
        Args:
            source: Path to image file, video file, or directory of images.
            target_size: (width, height) to resize frames.
            loop: Whether to loop video playback.
        """
        self.source = source
        self.target_size = target_size
        self.loop = loop
        self.cap = None
        self.images = []
        self.current_index = 0
        
        self._load_source()
    
    def _load_source(self):
        """Load the image/video source."""
        import os
        
        if os.path.isfile(self.source):
            # Check if it's a video
            ext = os.path.splitext(self.source)[1].lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    raise ValueError(f"Could not open video: {self.source}")
            else:
                # Single image
                img = cv2.imread(self.source)
                if img is None:
                    raise ValueError(f"Could not load image: {self.source}")
                self.images = [img]
        
        elif os.path.isdir(self.source):
            # Directory of images
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for filename in sorted(os.listdir(self.source)):
                ext = os.path.splitext(filename)[1].lower()
                if ext in extensions:
                    img_path = os.path.join(self.source, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        self.images.append(img)
            
            if not self.images:
                raise ValueError(f"No images found in: {self.source}")
        else:
            raise ValueError(f"Invalid source: {self.source}")
    
    def capture(self) -> Optional[np.ndarray]:
        """
        Get the next frame.
        
        Returns:
            numpy array of the frame in BGR format, or None if end reached.
        """
        frame = None
        
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        return None
                else:
                    return None
        else:
            if self.current_index >= len(self.images):
                if self.loop:
                    self.current_index = 0
                else:
                    return None
            
            frame = self.images[self.current_index].copy()
            self.current_index += 1
        
        if frame is not None and self.target_size is not None:
            frame = cv2.resize(frame, self.target_size)
        
        return frame
    
    def capture_continuous(self):
        """Generator that yields frames continuously."""
        while True:
            frame = self.capture()
            if frame is None:
                break
            yield frame
    
    def release(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()


def select_capture_region():
    """
    Interactive tool to select screen capture region.
    
    Returns:
        Tuple (left, top, width, height) of selected region.
    """
    print("Press Enter to capture current screen...")
    input()
    
    sct = mss.mss()
    screenshot = sct.grab(sct.monitors[1])
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Create a window and let user select region
    print("Draw a rectangle around the game window, then press Enter")
    roi = cv2.selectROI("Select Game Region", img, fromCenter=False)
    cv2.destroyAllWindows()
    
    if roi[2] > 0 and roi[3] > 0:
        return (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    return None


if __name__ == "__main__":
    # Test screen capture
    print("Testing screen capture...")
    
    # Select region interactively
    region = select_capture_region()
    if region:
        print(f"Selected region: {region}")
        
        capture = ScreenCapture(region=region, fps_limit=10)
        
        print("Capturing frames. Press 'q' to quit...")
        for frame in capture.capture_continuous():
            cv2.imshow("Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        capture.release()
        cv2.destroyAllWindows()
