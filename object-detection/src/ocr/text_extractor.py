"""
OCR text extraction for Clash Royale game elements.

Extracts timer, elixir count, and tower HP values from screenshots.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Try PaddleOCR first, fall back to EasyOCR
try:
    from paddleocr import PaddleOCR
    OCR_ENGINE = "paddle"
except ImportError:
    try:
        import easyocr
        OCR_ENGINE = "easy"
    except ImportError:
        OCR_ENGINE = None


@dataclass
class OCRResult:
    """Result of OCR extraction."""
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2


@dataclass
class TimerResult:
    """Parsed timer result."""
    minutes: int
    seconds: int
    total_seconds: int
    is_overtime: bool
    raw_text: str


@dataclass
class GameOCRResults:
    """All OCR results from a game screenshot."""
    timer: Optional[TimerResult] = None
    elixir: Optional[int] = None
    player_king_hp: Optional[int] = None
    player_left_princess_hp: Optional[int] = None
    player_right_princess_hp: Optional[int] = None
    enemy_king_hp: Optional[int] = None
    enemy_left_princess_hp: Optional[int] = None
    enemy_right_princess_hp: Optional[int] = None


class TextExtractor:
    """OCR engine for extracting text from Clash Royale screenshots."""

    def __init__(
        self,
        engine: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """Initialize OCR engine.

        Args:
            engine: OCR engine to use ('paddle', 'easy', or None for auto).
            use_gpu: Whether to use GPU acceleration.
        """
        self.engine_type = engine or OCR_ENGINE

        if self.engine_type is None:
            raise ImportError(
                "No OCR engine available. Install paddleocr or easyocr:\n"
                "  pip install paddleocr paddlepaddle\n"
                "  pip install easyocr"
            )

        if self.engine_type == "paddle":
            self.engine = PaddleOCR(
                lang="en",
                use_textline_orientation=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        elif self.engine_type == "easy":
            import easyocr
            self.engine = easyocr.Reader(
                ["en"],
                gpu=use_gpu,
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine_type}")

    def extract_text(
        self,
        image: Union[str, np.ndarray],
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[OCRResult]:
        """Extract text from image or region.

        Args:
            image: Image path or numpy array (BGR format).
            region: Optional (x1, y1, x2, y2) to crop before OCR.

        Returns:
            List of OCR results.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        # Crop to region if specified
        if region is not None:
            x1, y1, x2, y2 = region
            img = img[y1:y2, x1:x2]

        results = []

        if self.engine_type == "paddle":
            ocr_output = self.engine.ocr(img)
            if ocr_output:
                page = ocr_output[0] if ocr_output else None
                # New PaddleOCR API returns list of dicts
                if isinstance(page, dict):
                    texts = page.get("rec_texts", [])
                    scores = page.get("rec_scores", [])
                    polys = page.get("dt_polys", [])
                    for i, (text, score) in enumerate(zip(texts, scores)):
                        if polys and i < len(polys):
                            pts = polys[i]
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                        else:
                            bbox = None
                        if region is not None and bbox is not None:
                            x1, y1, _, _ = region
                            bbox = (bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1)
                        results.append(OCRResult(text=text, confidence=score, bbox=bbox))
                # Old PaddleOCR API returns nested lists
                elif isinstance(page, list):
                    for line in page:
                        bbox_points = line[0]
                        text = line[1][0]
                        confidence = line[1][1]
                        xs = [p[0] for p in bbox_points]
                        ys = [p[1] for p in bbox_points]
                        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                        if region is not None:
                            x1, y1, _, _ = region
                            bbox = (bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1)
                        results.append(OCRResult(text=text, confidence=confidence, bbox=bbox))

        elif self.engine_type == "easy":
            ocr_output = self.engine.readtext(img)
            for item in ocr_output:
                bbox_points = item[0]
                text = item[1]
                confidence = item[2]

                # Convert polygon to bbox
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

                # Adjust for region offset
                if region is not None:
                    x1, y1, _, _ = region
                    bbox = (bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1)

                results.append(OCRResult(text=text, confidence=confidence, bbox=bbox))

        return results

    def preprocess_for_ocr(
        self,
        image: np.ndarray,
        enhance_contrast: bool = True,
        threshold: bool = False,
        invert: bool = False,
    ) -> np.ndarray:
        """Preprocess image for better OCR results.

        Args:
            image: Input image (BGR format).
            enhance_contrast: Apply CLAHE contrast enhancement.
            threshold: Apply binary thresholding.
            invert: Invert colors (useful for white text on dark background).

        Returns:
            Preprocessed image.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Enhance contrast
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # Apply threshold
        if threshold:
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed
        if invert:
            gray = cv2.bitwise_not(gray)

        # Convert back to BGR for OCR
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class GameTextExtractor(TextExtractor):
    """Specialized text extractor for Clash Royale game elements."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Default regions for 540x960 resolution
        # These should be calibrated for actual screenshots
        self.default_regions = {
            "timer": (400, 5, 535, 45),
            "elixir": (125, 912, 175, 948),
            # HP regions are approximate - need calibration
            "player_king_hp": (240, 695, 300, 720),
            "player_left_princess_hp": (85, 615, 145, 635),
            "player_right_princess_hp": (395, 615, 455, 635),
            "enemy_king_hp": (240, 175, 300, 200),
            "enemy_left_princess_hp": (85, 285, 145, 305),
            "enemy_right_princess_hp": (395, 285, 455, 305),
        }

    def scale_regions(
        self,
        img_width: int,
        img_height: int,
        base_width: int = 540,
        base_height: int = 960,
    ) -> Dict[str, Tuple[int, int, int, int]]:
        """Scale default regions to actual image dimensions."""
        x_scale = img_width / base_width
        y_scale = img_height / base_height

        scaled = {}
        for name, (x1, y1, x2, y2) in self.default_regions.items():
            scaled[name] = (
                int(x1 * x_scale),
                int(y1 * y_scale),
                int(x2 * x_scale),
                int(y2 * y_scale),
            )
        return scaled

    def parse_timer(self, text: str) -> Optional[TimerResult]:
        """Parse timer text into structured result.

        Args:
            text: Raw text that might contain timer (e.g., "Time left: 2:30")

        Returns:
            TimerResult if parsing successful, None otherwise.
        """
        # Clean up text
        text = text.strip()

        # Try various patterns
        patterns = [
            r"(\d+):(\d{2})",  # M:SS or MM:SS
            r"(\d+)\.(\d{2})",  # Some OCR mistakes : for .
            r"Time\s*left[:\s]*(\d+)[:\.](\d{2})",  # Full format
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                total_seconds = minutes * 60 + seconds

                # Overtime starts after 3:00 (180 seconds) are over
                # Game shows countdown, so overtime is when we see 4:00+ remaining
                # Actually, regular time is 3:00, so >180s remaining means overtime
                is_overtime = total_seconds > 180

                return TimerResult(
                    minutes=minutes,
                    seconds=seconds,
                    total_seconds=total_seconds,
                    is_overtime=is_overtime,
                    raw_text=text,
                )

        return None

    def parse_elixir(self, text: str) -> Optional[int]:
        """Parse elixir count from text.

        Args:
            text: Raw text that should be a number 0-10.

        Returns:
            Elixir count if valid, None otherwise.
        """
        # Clean up text
        text = text.strip()

        # Handle common OCR mistakes
        text = text.replace("O", "0").replace("o", "0")
        text = text.replace("I", "1").replace("l", "1")

        # Extract digits
        match = re.search(r"\d+", text)
        if match:
            value = int(match.group())
            if 0 <= value <= 10:
                return value

        return None

    def parse_hp(self, text: str) -> Optional[int]:
        """Parse HP value from text.

        Args:
            text: Raw text that should be a number.

        Returns:
            HP value if valid, None otherwise.
        """
        # Clean up text
        text = text.strip()

        # Remove common OCR noise
        text = re.sub(r"[^0-9]", "", text)

        if text:
            value = int(text)
            # Valid HP ranges (rough estimates)
            if 0 <= value <= 10000:
                return value

        return None

    def extract_game_text(
        self,
        image: Union[str, np.ndarray],
        regions: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    ) -> GameOCRResults:
        """Extract all game text elements from a screenshot.

        Args:
            image: Screenshot image (path or array).
            regions: Custom region definitions. Uses scaled defaults if None.

        Returns:
            GameOCRResults with all extracted values.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        height, width = img.shape[:2]

        # Get regions
        if regions is None:
            regions = self.scale_regions(width, height)

        results = GameOCRResults()

        # Extract timer
        try:
            timer_region = regions.get("timer")
            if timer_region:
                timer_crop = img[timer_region[1]:timer_region[3], timer_region[0]:timer_region[2]]
                # Preprocess for timer (usually light text on dark)
                timer_prep = self.preprocess_for_ocr(timer_crop, enhance_contrast=True)
                timer_ocr = self.extract_text(timer_prep)
                for ocr_result in timer_ocr:
                    timer = self.parse_timer(ocr_result.text)
                    if timer:
                        results.timer = timer
                        break
        except Exception as e:
            pass  # Skip on error

        # Extract elixir
        try:
            elixir_region = regions.get("elixir")
            if elixir_region:
                elixir_crop = img[elixir_region[1]:elixir_region[3], elixir_region[0]:elixir_region[2]]
                elixir_prep = self.preprocess_for_ocr(elixir_crop, enhance_contrast=True)
                elixir_ocr = self.extract_text(elixir_prep)
                for ocr_result in elixir_ocr:
                    elixir = self.parse_elixir(ocr_result.text)
                    if elixir is not None:
                        results.elixir = elixir
                        break
        except Exception as e:
            pass

        # Extract HP values
        hp_fields = [
            ("player_king_hp", "player_king_hp"),
            ("player_left_princess_hp", "player_left_princess_hp"),
            ("player_right_princess_hp", "player_right_princess_hp"),
            ("enemy_king_hp", "enemy_king_hp"),
            ("enemy_left_princess_hp", "enemy_left_princess_hp"),
            ("enemy_right_princess_hp", "enemy_right_princess_hp"),
        ]

        for region_name, field_name in hp_fields:
            try:
                hp_region = regions.get(region_name)
                if hp_region:
                    hp_crop = img[hp_region[1]:hp_region[3], hp_region[0]:hp_region[2]]
                    hp_prep = self.preprocess_for_ocr(hp_crop, enhance_contrast=True)
                    hp_ocr = self.extract_text(hp_prep)
                    for ocr_result in hp_ocr:
                        hp = self.parse_hp(ocr_result.text)
                        if hp is not None:
                            setattr(results, field_name, hp)
                            break
            except Exception as e:
                pass

        return results

    def visualize_regions(
        self,
        image: Union[str, np.ndarray],
        regions: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Draw OCR regions on image for debugging.

        Args:
            image: Input image.
            regions: Region definitions.
            output_path: Path to save result.

        Returns:
            Image with drawn regions.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        height, width = img.shape[:2]

        if regions is None:
            regions = self.scale_regions(width, height)

        colors = {
            "timer": (0, 255, 255),  # Yellow
            "elixir": (255, 0, 255),  # Magenta
        }
        default_color = (0, 255, 0)  # Green for HP

        for name, (x1, y1, x2, y2) in regions.items():
            color = colors.get(name, default_color)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if output_path:
            cv2.imwrite(output_path, img)

        return img


def create_extractor(use_gpu: bool = True) -> GameTextExtractor:
    """Create a GameTextExtractor instance.

    Args:
        use_gpu: Whether to use GPU acceleration.

    Returns:
        GameTextExtractor instance.
    """
    return GameTextExtractor(use_gpu=use_gpu)
