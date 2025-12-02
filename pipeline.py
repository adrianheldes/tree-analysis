"""
Forest Analysis Pipeline v2.0
==============================
Analyzes MP4 video to extract forest inventory data.

Features:
- Green vegetation segmentation (HSV-based + optional SAM)
- Tree trunk detection via edge/contour analysis
- Color-based species hints (bark color, foliage)
- Density and canopy coverage estimation
- Outputs structured ForestPlan JSON

Author: Adrian
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass, field, asdict
from datetime import date
from enum import Enum
import json
import os
from pathlib import Path

from video_reader import VideoReader
from tree_struct import (
    ForestPlan, Site, Stand, ManagementActivity,
    ValuableSite, MapLayer, EconomicInfo
)


# =============================================================================
# CONFIGURATION
# =============================================================================
class AnalysisMode(Enum):
    """Analysis modes for different scenarios."""
    FAST = "fast"           # Quick analysis, fewer frames
    STANDARD = "standard"   # Balanced
    DETAILED = "detailed"   # More frames, higher accuracy


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Video processing
    frame_interval: int = 3                    # Seconds between frames
    max_frames: Optional[int] = None           # Limit frames (None = all)

    # Analysis mode
    mode: AnalysisMode = AnalysisMode.STANDARD

    # Detection settings
    min_tree_height_ratio: float = 0.15        # Min tree height as % of frame
    green_threshold: Tuple[int, int] = (35, 85)  # HSV hue range for vegetation
    brown_threshold: Tuple[int, int] = (10, 25)  # HSV hue range for bark

    # Output
    output_dir: str = "output"
    save_debug_frames: bool = False

    # Optional ML models
    use_yolo: bool = False                     # Use YOLO if available
    yolo_model: str = "yolov8l.pt"             # Options: yolov8n/s/m/l/x.pt (larger = more accurate)

    # DeepForest - specialized tree detection
    use_deepforest: bool = False               # Use DeepForest for tree crown detection
    deepforest_score_thresh: float = 0.3       # Confidence threshold for DeepForest

    # SAM - Segment Anything Model
    use_sam: bool = False                      # Use SAM for precise segmentation
    sam_model: str = "vit_b"                   # Options: vit_b, vit_l, vit_h (larger = more accurate)


# =============================================================================
# FRAME ANALYSIS - Core Computer Vision
# =============================================================================
class FrameAnalyzer:
    """
    Analyzes individual frames using computer vision.
    Supports optional ML models: YOLO, DeepForest, SAM for enhanced accuracy.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._yolo_model = None
        self._deepforest_model = None
        self._sam_predictor = None

        if config.use_yolo:
            self._load_yolo()
        if config.use_deepforest:
            self._load_deepforest()
        if config.use_sam:
            self._load_sam()

    def _load_yolo(self):
        """Try to load YOLO model."""
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.config.yolo_model)
            print(f"[INFO] YOLO model loaded: {self.config.yolo_model}")
        except Exception as e:
            print(f"[WARN] YOLO not available: {e}")
            self._yolo_model = None

    def _load_deepforest(self):
        """Try to load DeepForest model for tree crown detection."""
        try:
            from deepforest import main as deepforest_main
            self._deepforest_model = deepforest_main.deepforest()
            self._deepforest_model.use_release()
            print("[INFO] DeepForest model loaded (tree crown detection)")
        except Exception as e:
            print(f"[WARN] DeepForest not available: {e}")
            self._deepforest_model = None

    def _load_sam(self):
        """Try to load Segment Anything Model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            import torch

            # Model checkpoint paths
            sam_checkpoints = {
                "vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth"
            }

            model_type = self.config.sam_model
            checkpoint = sam_checkpoints.get(model_type, sam_checkpoints["vit_b"])

            # Check if checkpoint exists
            if not os.path.exists(checkpoint):
                print(f"[WARN] SAM checkpoint not found: {checkpoint}")
                print(f"[INFO] Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
                self._sam_predictor = None
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            self._sam_predictor = SamPredictor(sam)
            print(f"[INFO] SAM model loaded: {model_type} on {device}")
        except Exception as e:
            print(f"[WARN] SAM not available: {e}")
            self._sam_predictor = None

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single frame.

        Returns dict with:
        - vegetation_mask: binary mask of green areas
        - vegetation_percent: % of frame that is vegetation
        - tree_regions: list of detected tree bounding boxes
        - trunk_lines: detected vertical lines (potential trunks)
        - color_profile: dominant colors analysis
        - brightness: average brightness
        - is_forest: boolean classification
        """
        h, w = frame.shape[:2]

        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Vegetation detection (green areas) - enhanced with SAM if available
        veg_mask, veg_percent = self._detect_vegetation(hsv)

        # 1b. Enhance vegetation mask with SAM if available
        if self._sam_predictor is not None:
            veg_mask, veg_percent = self._enhance_with_sam(frame, veg_mask)

        # 2. Bark/trunk detection (brown vertical areas)
        trunk_mask, trunk_lines = self._detect_trunks(hsv, gray)

        # 3. Tree region detection (CV-based)
        tree_regions = self._detect_tree_regions(veg_mask, trunk_mask, h, w)

        # 4. Enhance with DeepForest (specialized tree crown detection)
        if self._deepforest_model is not None:
            df_detections = self._deepforest_detect(frame)
            tree_regions = self._merge_detections(tree_regions, df_detections)

        # 5. Enhance with YOLO (general object detection)
        if self._yolo_model is not None:
            ml_detections = self._yolo_detect(frame)
            tree_regions = self._merge_detections(tree_regions, ml_detections)

        # 6. Color analysis for species hints
        color_profile = self._analyze_colors(frame, hsv)

        # 7. Scene classification
        brightness = np.mean(gray)
        is_forest = veg_percent > 20 and len(tree_regions) > 0

        # 8. Canopy coverage estimation
        canopy_coverage = self._estimate_canopy_coverage(veg_mask, h)

        return {
            "vegetation_mask": veg_mask,
            "vegetation_percent": round(veg_percent, 2),
            "trunk_mask": trunk_mask,
            "trunk_lines": trunk_lines,
            "tree_regions": tree_regions,
            "tree_count": len(tree_regions),
            "color_profile": color_profile,
            "brightness": round(brightness, 2),
            "is_forest": is_forest,
            "canopy_coverage": round(canopy_coverage, 2),
            "frame_size": (w, h)
        }

    def _detect_vegetation(self, hsv: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect green vegetation using HSV thresholding."""
        h_low, h_high = self.config.green_threshold

        # Green range in HSV
        lower_green = np.array([h_low, 40, 40])
        upper_green = np.array([h_high, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Calculate percentage
        total_pixels = mask.shape[0] * mask.shape[1]
        green_pixels = np.count_nonzero(mask)
        percent = (green_pixels / total_pixels) * 100

        return mask, percent

    def _detect_trunks(self, hsv: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, List]:
        """Detect tree trunks using color and edge detection."""
        h, w = gray.shape

        # Brown/bark color detection
        h_low, h_high = self.config.brown_threshold
        lower_brown = np.array([h_low, 30, 30])
        upper_brown = np.array([h_high, 200, 180])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # Edge detection for vertical lines
        edges = cv2.Canny(gray, 50, 150)

        # Hough lines for trunk detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                                minLineLength=h*0.1, maxLineGap=20)

        trunk_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly vertical (tree trunk)
                if abs(x2 - x1) < 30:  # Nearly vertical
                    angle = np.arctan2(abs(y2-y1), abs(x2-x1)) * 180 / np.pi
                    if angle > 70:  # More than 70 degrees from horizontal
                        trunk_lines.append({
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2),
                            "length": int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
                        })

        return brown_mask, trunk_lines

    def _detect_tree_regions(self, veg_mask: np.ndarray, trunk_mask: np.ndarray,
                             h: int, w: int) -> List[Dict]:
        """Find tree regions by combining vegetation and trunk detection."""
        # Combine masks
        combined = cv2.bitwise_or(veg_mask, trunk_mask)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        min_height = int(h * self.config.min_tree_height_ratio)
        min_area = (w * h) * 0.01  # At least 1% of frame

        regions = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Filter by size - trees should be tall
            if ch >= min_height and area >= min_area:
                # Calculate aspect ratio (trees are taller than wide)
                aspect = ch / max(cw, 1)

                if aspect > 0.5:  # Reasonably tall
                    regions.append({
                        "bbox": [x, y, x + cw, y + ch],
                        "area": int(area),
                        "aspect_ratio": round(aspect, 2),
                        "confidence": min(0.9, area / (w * h) * 10)
                    })

        # Sort by area (largest first) and limit
        regions.sort(key=lambda r: r["area"], reverse=True)
        return regions[:50]  # Max 50 trees per frame

    def _yolo_detect(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLO detection if available."""
        if self._yolo_model is None:
            return []

        try:
            results = self._yolo_model(frame, verbose=False, conf=0.3)
            detections = []

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    name = result.names[cls]

                    # Look for tree-related classes or plants
                    if name in ["potted plant", "tree", "plant"]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(box.conf[0]),
                            "class": name,
                            "source": "yolo"
                        })

            return detections
        except Exception as e:
            print(f"[WARN] YOLO detection failed: {e}")
            return []

    def _deepforest_detect(self, frame: np.ndarray) -> List[Dict]:
        """Run DeepForest detection for tree crowns."""
        if self._deepforest_model is None:
            return []

        try:
            # DeepForest expects RGB, OpenCV loads as BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Predict tree crowns
            predictions = self._deepforest_model.predict_image(
                image=rgb_frame,
                return_plot=False
            )

            detections = []
            if predictions is not None and len(predictions) > 0:
                for _, row in predictions.iterrows():
                    if row['score'] >= self.config.deepforest_score_thresh:
                        detections.append({
                            "bbox": [
                                int(row['xmin']), int(row['ymin']),
                                int(row['xmax']), int(row['ymax'])
                            ],
                            "confidence": float(row['score']),
                            "class": row.get('label', 'Tree'),
                            "source": "deepforest"
                        })

            return detections
        except Exception as e:
            print(f"[WARN] DeepForest detection failed: {e}")
            return []

    def _enhance_with_sam(self, frame: np.ndarray, initial_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Enhance vegetation segmentation using SAM."""
        if self._sam_predictor is None:
            total_pixels = initial_mask.shape[0] * initial_mask.shape[1]
            green_pixels = np.count_nonzero(initial_mask)
            return initial_mask, (green_pixels / total_pixels) * 100

        try:
            # Convert BGR to RGB for SAM
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._sam_predictor.set_image(rgb_frame)

            # Find seed points from initial mask (center of vegetation regions)
            contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                total_pixels = initial_mask.shape[0] * initial_mask.shape[1]
                green_pixels = np.count_nonzero(initial_mask)
                return initial_mask, (green_pixels / total_pixels) * 100

            # Get centroids of largest contours as prompts
            point_coords = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    point_coords.append([cx, cy])

            if not point_coords:
                total_pixels = initial_mask.shape[0] * initial_mask.shape[1]
                green_pixels = np.count_nonzero(initial_mask)
                return initial_mask, (green_pixels / total_pixels) * 100

            # Run SAM with point prompts
            point_coords_np = np.array(point_coords)
            point_labels = np.ones(len(point_coords))  # 1 = foreground

            masks, scores, _ = self._sam_predictor.predict(
                point_coords=point_coords_np,
                point_labels=point_labels,
                multimask_output=True
            )

            # Use the mask with highest score
            best_mask_idx = np.argmax(scores)
            sam_mask = masks[best_mask_idx].astype(np.uint8) * 255

            # Combine SAM mask with initial mask (union)
            combined_mask = cv2.bitwise_or(initial_mask, sam_mask)

            # Calculate percentage
            total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
            green_pixels = np.count_nonzero(combined_mask)
            percent = (green_pixels / total_pixels) * 100

            return combined_mask, percent

        except Exception as e:
            print(f"[WARN] SAM enhancement failed: {e}")
            total_pixels = initial_mask.shape[0] * initial_mask.shape[1]
            green_pixels = np.count_nonzero(initial_mask)
            return initial_mask, (green_pixels / total_pixels) * 100

    def _merge_detections(self, cv_regions: List[Dict],
                          ml_detections: List[Dict]) -> List[Dict]:
        """Merge CV and ML detections, removing duplicates."""
        all_regions = cv_regions.copy()

        for ml_det in ml_detections:
            # Check if overlaps with existing
            is_duplicate = False
            ml_box = ml_det["bbox"]

            for cv_reg in cv_regions:
                cv_box = cv_reg["bbox"]
                iou = self._calculate_iou(ml_box, cv_box)
                if iou > 0.3:
                    is_duplicate = True
                    break

            if not is_duplicate:
                all_regions.append(ml_det)

        return all_regions

    def _calculate_iou(self, box1: List, box2: List) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _analyze_colors(self, frame: np.ndarray, hsv: np.ndarray) -> Dict:
        """Analyze color distribution for species hints."""
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]

        # Color categories
        colors = {
            "dark_green": 0,    # Spruce, Pine
            "light_green": 0,   # Birch, Aspen
            "yellow_green": 0,  # Deciduous autumn
            "brown": 0,         # Bark, dead wood
            "gray": 0           # Dead trees, rocks
        }

        # Count pixels in each category
        total = h_channel.size

        # Dark green (conifers): hue 35-70, high saturation
        dark_green_mask = (h_channel >= 35) & (h_channel <= 70) & (s_channel > 80)
        colors["dark_green"] = np.count_nonzero(dark_green_mask) / total * 100

        # Light green (deciduous): hue 35-85, lower saturation
        light_green_mask = (h_channel >= 35) & (h_channel <= 85) & (s_channel <= 80) & (s_channel > 30)
        colors["light_green"] = np.count_nonzero(light_green_mask) / total * 100

        # Yellow-green (autumn): hue 25-40
        yellow_mask = (h_channel >= 20) & (h_channel <= 40)
        colors["yellow_green"] = np.count_nonzero(yellow_mask) / total * 100

        # Brown (bark): hue 10-25
        brown_mask = (h_channel >= 10) & (h_channel <= 25) & (s_channel > 30)
        colors["brown"] = np.count_nonzero(brown_mask) / total * 100

        return {k: round(v, 2) for k, v in colors.items()}

    def _estimate_canopy_coverage(self, veg_mask: np.ndarray, height: int) -> float:
        """Estimate canopy coverage from upper portion of frame."""
        # Look at top 40% of frame for canopy
        canopy_zone = veg_mask[:int(height * 0.4), :]

        if canopy_zone.size == 0:
            return 0.0

        coverage = np.count_nonzero(canopy_zone) / canopy_zone.size * 100
        return coverage


# =============================================================================
# SPECIES ESTIMATOR
# =============================================================================
class SpeciesEstimator:
    """Estimate tree species from visual characteristics."""

    # Species profiles based on color/texture
    SPECIES_PROFILES = {
        "Spruce (Picea abies)": {
            "dark_green": (15, 100),   # High dark green
            "conifer_shape": True,
            "bark_color": "gray-brown"
        },
        "Pine (Pinus sylvestris)": {
            "dark_green": (10, 80),
            "conifer_shape": True,
            "bark_color": "red-brown"
        },
        "Birch (Betula)": {
            "light_green": (15, 100),
            "deciduous": True,
            "bark_color": "white"
        },
        "Oak (Quercus)": {
            "dark_green": (10, 50),
            "deciduous": True,
            "bark_color": "dark-gray"
        },
        "Aspen (Populus tremula)": {
            "light_green": (10, 80),
            "deciduous": True,
            "bark_color": "gray-green"
        },
        "Alder (Alnus)": {
            "light_green": (5, 60),
            "deciduous": True,
            "bark_color": "dark"
        }
    }

    def estimate(self, color_profile: Dict, canopy_coverage: float) -> List[Dict]:
        """
        Estimate likely species based on color analysis.

        Returns list of species with confidence scores.
        """
        estimates = []

        dark_green = color_profile.get("dark_green", 0)
        light_green = color_profile.get("light_green", 0)

        # High dark green suggests conifers
        if dark_green > 15:
            estimates.append({
                "species": "Spruce (Picea abies)",
                "confidence": min(0.7, dark_green / 30),
                "reason": "High dark green vegetation detected"
            })
            estimates.append({
                "species": "Pine (Pinus sylvestris)",
                "confidence": min(0.5, dark_green / 40),
                "reason": "Conifer characteristics"
            })

        # Light green suggests deciduous
        if light_green > 10:
            estimates.append({
                "species": "Birch (Betula)",
                "confidence": min(0.6, light_green / 25),
                "reason": "Light green foliage detected"
            })
            estimates.append({
                "species": "Aspen (Populus tremula)",
                "confidence": min(0.4, light_green / 30),
                "reason": "Deciduous characteristics"
            })

        # Sort by confidence
        estimates.sort(key=lambda x: x["confidence"], reverse=True)

        return estimates[:4]  # Top 4 species


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class ForestPipeline:
    """
    Main analysis pipeline v2.0

    Processes video frames and generates comprehensive ForestPlan output.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.frame_analyzer = FrameAnalyzer(self.config)
        self.species_estimator = SpeciesEstimator()

        # Results storage
        self.frame_results: List[Dict] = []
        self.aggregated_stats: Dict = {}

    def analyze_video(self, video_path: str) -> ForestPlan:
        """
        Analyze video and return ForestPlan.

        Args:
            video_path: Path to MP4 video

        Returns:
            ForestPlan with all analysis data
        """
        print("\n" + "=" * 60)
        print("🌲 FOREST ANALYSIS PIPELINE v2.0")
        print("=" * 60)
        print(f"Video: {video_path}")
        print(f"Mode: {self.config.mode.value}")
        print(f"Frame interval: {self.config.frame_interval}s")
        print()

        # Validate video
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Reset state
        self.frame_results = []

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Process video
        self._process_video(video_path)

        # Aggregate results
        self._aggregate_results()

        # Build forest plan
        plan = self._build_forest_plan(video_path)

        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 60)

        return plan

    def _process_video(self, video_path: str):
        """Process video frames."""
        reader = VideoReader(
            video_path=video_path,
            interval_seconds=self.config.frame_interval,
            out_dir=None
        )

        frame_gen = reader.extract_frames()
        frame_count = 0

        print("[PROCESSING FRAMES]")

        for timestamp, frame in frame_gen:
            frame_count += 1

            if self.config.max_frames and frame_count > self.config.max_frames:
                break

            # Analyze frame
            result = self.frame_analyzer.analyze(frame)
            result["timestamp"] = timestamp
            result["frame_number"] = frame_count

            self.frame_results.append(result)

            # Progress output
            status = "🌲" if result["is_forest"] else "⬜"
            print(f"  Frame {frame_count:3d} @ {timestamp:4d}s | "
                  f"Trees: {result['tree_count']:2d} | "
                  f"Veg: {result['vegetation_percent']:5.1f}% | "
                  f"Canopy: {result['canopy_coverage']:5.1f}% {status}")

            # Save debug frame if enabled
            if self.config.save_debug_frames:
                self._save_debug_frame(frame, result, timestamp)

        print(f"\n  Total frames processed: {frame_count}")

    def _save_debug_frame(self, frame: np.ndarray, result: Dict, timestamp: int):
        """Save annotated debug frame."""
        debug = frame.copy()

        # Draw tree regions
        for region in result.get("tree_regions", []):
            bbox = region["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            conf = region.get("confidence", 0)
            color = (0, int(255 * conf), 0)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

        # Draw trunk lines
        for line in result.get("trunk_lines", []):
            cv2.line(debug, (line["x1"], line["y1"]),
                    (line["x2"], line["y2"]), (0, 0, 255), 2)

        # Add text overlay
        cv2.putText(debug, f"Trees: {result['tree_count']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(debug, f"Veg: {result['vegetation_percent']:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        path = os.path.join(self.config.output_dir, f"debug_{timestamp:05d}.jpg")
        cv2.imwrite(path, debug)

    def _aggregate_results(self):
        """Aggregate frame-level results into summary statistics."""
        if not self.frame_results:
            self.aggregated_stats = {}
            return

        n = len(self.frame_results)

        # Tree counts
        tree_counts = [r["tree_count"] for r in self.frame_results]

        # Vegetation percentages
        veg_percents = [r["vegetation_percent"] for r in self.frame_results]

        # Canopy coverage
        canopy_values = [r["canopy_coverage"] for r in self.frame_results]

        # Forest frames
        forest_frames = sum(1 for r in self.frame_results if r["is_forest"])

        # Aggregate color profiles
        color_totals = {}
        for r in self.frame_results:
            for color, pct in r.get("color_profile", {}).items():
                color_totals[color] = color_totals.get(color, 0) + pct
        avg_colors = {k: round(v / n, 2) for k, v in color_totals.items()}

        self.aggregated_stats = {
            "frames_analyzed": n,
            "forest_frames": forest_frames,
            "forest_ratio": round(forest_frames / n, 2),
            "trees": {
                "min": min(tree_counts),
                "max": max(tree_counts),
                "mean": round(sum(tree_counts) / n, 1),
                "total_detections": sum(tree_counts)
            },
            "vegetation": {
                "min": round(min(veg_percents), 1),
                "max": round(max(veg_percents), 1),
                "mean": round(sum(veg_percents) / n, 1)
            },
            "canopy_coverage": {
                "min": round(min(canopy_values), 1),
                "max": round(max(canopy_values), 1),
                "mean": round(sum(canopy_values) / n, 1)
            },
            "color_profile": avg_colors
        }

    def _build_forest_plan(self, video_path: str) -> ForestPlan:
        """Build ForestPlan from aggregated results."""
        stats = self.aggregated_stats
        video_name = Path(video_path).stem

        plan = ForestPlan(name=f"Analysis: {video_name}")

        # === SITE ===
        site_type = self._classify_site_type(stats)
        nutrient = self._estimate_nutrient_level(stats)
        timber = self._estimate_timber_potential(stats)

        site = Site(
            name=f"Site from {video_name}",
            type=site_type,
            nutrient_level=nutrient,
            timber_potential=timber,
            notes=f"Analyzed {stats.get('frames_analyzed', 0)} frames. "
                  f"Forest coverage: {stats.get('forest_ratio', 0)*100:.0f}%"
        )
        plan.add_site(site)

        # === STAND ===
        species_estimates = self.species_estimator.estimate(
            stats.get("color_profile", {}),
            stats.get("canopy_coverage", {}).get("mean", 0)
        )

        species_list = [s["species"] for s in species_estimates[:3]]
        if not species_list:
            species_list = ["Undetermined"]

        dev_class = self._estimate_development_class(stats)

        stand = Stand(
            species=species_list,
            age=None,  # Cannot determine from video
            size_ha=None,  # Requires GPS data
            development_class=dev_class,
            notes=f"Avg {stats.get('trees', {}).get('mean', 0):.0f} trees/frame. "
                  f"Canopy: {stats.get('canopy_coverage', {}).get('mean', 0):.1f}%. "
                  f"Species confidence: {species_estimates[0]['confidence']*100:.0f}% for {species_estimates[0]['species']}" if species_estimates else ""
        )
        plan.add_stand(stand)

        # === MAPS ===
        nls_map = MapLayer(
            name="National Land Survey Finland (Karttapaikka)",
            description="Finnish national map service for detailed forest maps",
            url="https://karttapaikka.fi/",
            metadata={
                "integration": "Add GPS coordinates for direct linking",
                "layers": ["forest", "terrain", "property boundaries"]
            }
        )
        plan.add_map(nls_map)

        # === ECONOMIC INFO ===
        econ = self._estimate_economics(stats, species_estimates)
        plan.set_economic_info(econ)

        # === VALUABLE SITES ===
        if stats.get("canopy_coverage", {}).get("mean", 0) > 60:
            valuable = ValuableSite(
                name="High Canopy Coverage Area",
                protection_status="candidate",
                recreational_value="medium",
                notes="Dense canopy detected - may have biodiversity value"
            )
            plan.add_valuable_site(valuable)

        return plan

    def _classify_site_type(self, stats: Dict) -> str:
        """Classify site type from statistics."""
        forest_ratio = stats.get("forest_ratio", 0)
        veg_mean = stats.get("vegetation", {}).get("mean", 0)
        tree_mean = stats.get("trees", {}).get("mean", 0)

        if forest_ratio < 0.2:
            return "barren"
        elif tree_mean < 3:
            return "sparse"
        elif tree_mean < 8:
            return "grove"
        elif veg_mean > 50:
            return "dense_forest"
        else:
            return "mixed_forest"

    def _estimate_nutrient_level(self, stats: Dict) -> str:
        """Estimate nutrient level."""
        veg = stats.get("vegetation", {}).get("mean", 0)
        canopy = stats.get("canopy_coverage", {}).get("mean", 0)

        score = veg * 0.5 + canopy * 0.5

        if score > 50:
            return "high"
        elif score > 25:
            return "medium"
        else:
            return "low"

    def _estimate_timber_potential(self, stats: Dict) -> str:
        """Estimate timber potential."""
        tree_mean = stats.get("trees", {}).get("mean", 0)
        colors = stats.get("color_profile", {})

        # Conifers have higher timber value
        conifer_score = colors.get("dark_green", 0)

        if tree_mean > 10 and conifer_score > 15:
            return "high"
        elif tree_mean > 5:
            return "medium"
        else:
            return "low"

    def _estimate_development_class(self, stats: Dict) -> str:
        """Estimate stand development class."""
        canopy = stats.get("canopy_coverage", {}).get("mean", 0)
        tree_mean = stats.get("trees", {}).get("mean", 0)

        if canopy > 60 and tree_mean > 10:
            return "mature"
        elif canopy > 30:
            return "pole_stage"
        elif tree_mean > 3:
            return "young"
        else:
            return "regeneration"

    def _estimate_economics(self, stats: Dict, species: List[Dict]) -> EconomicInfo:
        """Estimate economic value."""
        # Price estimates EUR/m³
        PRICES = {
            "Spruce": 75,
            "Pine": 70,
            "Birch": 45,
            "Oak": 200,
            "default": 40
        }

        # Get price for dominant species
        price = PRICES["default"]
        if species:
            for sp_name, sp_price in PRICES.items():
                if sp_name in species[0].get("species", ""):
                    price = sp_price
                    break

        # Rough volume estimate
        tree_total = stats.get("trees", {}).get("total_detections", 0)
        frames = stats.get("frames_analyzed", 1)

        # Assume ~0.3 m³ per unique tree detection (accounting for same trees in multiple frames)
        unique_trees_estimate = tree_total / max(frames * 0.5, 1)
        volume = unique_trees_estimate * 0.3
        value = volume * price

        return EconomicInfo(
            estimated_value_eur=round(value, 2),
            timber_sales_potential_m3=round(volume, 2),
            notes=f"Estimate based on ~{unique_trees_estimate:.0f} unique trees. "
                  f"Price: {price} EUR/m³. Requires field verification."
        )

    def get_summary(self) -> Dict:
        """Get analysis summary."""
        return {
            "aggregated_stats": self.aggregated_stats,
            "frame_count": len(self.frame_results),
            "config": asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else str(self.config)
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================
def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="🌲 Forest Analysis Pipeline v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py video.mp4
  python pipeline.py video.mp4 --interval 2 --output results.json
  python pipeline.py video.mp4 --mode detailed --save-debug
  python pipeline.py video.mp4 --use-yolo --use-deepforest  # Enhanced ML detection
  python pipeline.py video.mp4 --use-all-models             # Maximum accuracy
        """
    )

    parser.add_argument("video", help="Path to MP4 video file")
    parser.add_argument("--interval", "-i", type=int, default=3,
                       help="Seconds between frames (default: 3)")
    parser.add_argument("--output", "-o", default="forest_analysis.json",
                       help="Output JSON file")
    parser.add_argument("--mode", "-m", choices=["fast", "standard", "detailed"],
                       default="standard", help="Analysis mode")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to analyze")
    parser.add_argument("--save-debug", action="store_true",
                       help="Save debug frames with annotations")

    # ML Model options
    parser.add_argument("--use-yolo", action="store_true",
                       help="Use YOLOv8 for enhanced object detection")
    parser.add_argument("--yolo-model", default="yolov8l.pt",
                       choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                       help="YOLO model size (default: yolov8l.pt)")
    parser.add_argument("--use-deepforest", action="store_true",
                       help="Use DeepForest for specialized tree crown detection")
    parser.add_argument("--use-sam", action="store_true",
                       help="Use SAM (Segment Anything Model) for precise segmentation")
    parser.add_argument("--sam-model", default="vit_b",
                       choices=["vit_b", "vit_l", "vit_h"],
                       help="SAM model size (default: vit_b)")
    parser.add_argument("--use-all-models", action="store_true",
                       help="Enable all ML models for maximum accuracy")

    args = parser.parse_args()

    # Handle --use-all-models flag
    if args.use_all_models:
        args.use_yolo = True
        args.use_deepforest = True
        args.use_sam = True

    # Build config
    mode_map = {
        "fast": AnalysisMode.FAST,
        "standard": AnalysisMode.STANDARD,
        "detailed": AnalysisMode.DETAILED
    }

    config = PipelineConfig(
        frame_interval=args.interval,
        max_frames=args.max_frames,
        mode=mode_map[args.mode],
        save_debug_frames=args.save_debug,
        use_yolo=args.use_yolo,
        yolo_model=args.yolo_model,
        use_deepforest=args.use_deepforest,
        use_sam=args.use_sam,
        sam_model=args.sam_model
    )

    # Run pipeline
    pipeline = ForestPipeline(config)

    try:
        plan = pipeline.analyze_video(args.video)

        # Print summary
        print(f"\n📊 FOREST PLAN SUMMARY")
        print("-" * 40)
        print(json.dumps(plan.summary(), indent=2))

        # Save results
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(plan.to_json())
        print(f"\n💾 Results saved to: {args.output}")

        # Also save detailed stats
        stats_file = args.output.replace(".json", "_stats.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(pipeline.get_summary(), f, indent=2, default=str)
        print(f"📈 Stats saved to: {stats_file}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

    return 0


if __name__ == "__main__":
    exit(main())