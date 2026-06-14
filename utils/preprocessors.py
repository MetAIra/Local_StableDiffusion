"""ControlNet プリプロセッサモジュール

ポーズ検出、エッジ検出、深度推定を行うプリプロセッサを提供
"""
import cv2
import numpy as np
from PIL import Image


class ControlNetPreprocessors:
    """ControlNet用のプリプロセッサを管理するクラス"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.openpose_detector = None
        self.depth_estimator = None
        self.lineart_detector = None
        self.scribble_detector = None

    def _load_openpose(self):
        """OpenPose検出器をロード"""
        if self.openpose_detector is None:
            try:
                from controlnet_aux import OpenposeDetector
                self.openpose_detector = OpenposeDetector.from_pretrained(
                    "lllyasviel/ControlNet"
                )
                print("OpenPose detector loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load OpenPose detector: {e}")
                print("Falling back to simple edge detection for pose")
        return self.openpose_detector

    def _load_depth_estimator(self):
        """深度推定モデルをロード"""
        if self.depth_estimator is None:
            try:
                from controlnet_aux import MidasDetector
                self.depth_estimator = MidasDetector.from_pretrained(
                    "lllyasviel/Annotators"
                )
                print("Depth estimator loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load depth estimator: {e}")
                print("Falling back to simple depth estimation")
        return self.depth_estimator

    def _load_lineart_detector(self):
        """Lineart検出器をロード"""
        if self.lineart_detector is None:
            try:
                from controlnet_aux import LineartDetector
                self.lineart_detector = LineartDetector.from_pretrained(
                    "lllyasviel/Annotators"
                )
                print("Lineart detector loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load Lineart detector: {e}")
                print("Falling back to Canny edge detection for lineart")
        return self.lineart_detector

    def _load_scribble_detector(self):
        """Scribble検出器をロード"""
        if self.scribble_detector is None:
            try:
                from controlnet_aux import HEDdetector
                self.scribble_detector = HEDdetector.from_pretrained(
                    "lllyasviel/Annotators"
                )
                print("Scribble (HED) detector loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load Scribble detector: {e}")
                print("Falling back to simple edge detection for scribble")
        return self.scribble_detector

    def detect_openpose(self, image: Image.Image) -> Image.Image:
        """OpenPoseでポーズを検出

        Args:
            image: 入力画像

        Returns:
            ポーズ検出結果の画像
        """
        detector = self._load_openpose()

        if detector is not None:
            try:
                pose_image = detector(image)
                return pose_image
            except Exception as e:
                print(f"OpenPose detection failed: {e}")

        # フォールバック: エッジ検出を使用
        print("Using edge detection as fallback for pose")
        return self.detect_canny(image, low_threshold=50, high_threshold=150)

    def detect_canny(
        self,
        image: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> Image.Image:
        """Cannyエッジ検出

        Args:
            image: 入力画像
            low_threshold: 下限閾値
            high_threshold: 上限閾値

        Returns:
            エッジ検出結果の画像
        """
        # PILからOpenCV形式に変換
        img_array = np.array(image)

        # RGBからグレースケールに変換
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Cannyエッジ検出
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # 3チャンネルに変換（ControlNet用）
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(edges_rgb)

    def estimate_depth(self, image: Image.Image) -> Image.Image:
        """深度を推定

        Args:
            image: 入力画像

        Returns:
            深度マップ画像
        """
        estimator = self._load_depth_estimator()

        if estimator is not None:
            try:
                depth_image = estimator(image)
                return depth_image
            except Exception as e:
                print(f"Depth estimation failed: {e}")

        # フォールバック: 簡易深度推定（明度ベース）
        print("Using brightness-based depth as fallback")
        return self._simple_depth_estimation(image)

    def _simple_depth_estimation(self, image: Image.Image) -> Image.Image:
        """簡易深度推定（フォールバック用）

        明度を元に擬似的な深度マップを生成
        """
        img_array = np.array(image)

        # グレースケールに変換
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # ガウシアンブラーで平滑化
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # コントラストを強調
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

        # 3チャンネルに変換
        depth_rgb = cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        return Image.fromarray(depth_rgb)

    def detect_lineart(self, image: Image.Image) -> Image.Image:
        """Lineartで線画を抽出

        Args:
            image: 入力画像

        Returns:
            線画抽出結果の画像
        """
        detector = self._load_lineart_detector()

        if detector is not None:
            try:
                lineart_image = detector(image)
                return lineart_image
            except Exception as e:
                print(f"Lineart detection failed: {e}")

        # フォールバック: Cannyエッジ検出を使用
        print("Using Canny edge detection as fallback for lineart")
        return self.detect_canny(image, low_threshold=50, high_threshold=100)

    def detect_scribble(self, image: Image.Image) -> Image.Image:
        """Scribble（HED）で落書き風に変換

        Args:
            image: 入力画像

        Returns:
            落書き変換結果の画像
        """
        detector = self._load_scribble_detector()

        if detector is not None:
            try:
                # scribble=Trueでよりラフな線に
                scribble_image = detector(image, scribble=True)
                return scribble_image
            except Exception as e:
                print(f"Scribble detection failed: {e}")

        # フォールバック: 太めのエッジ検出
        print("Using thick edge detection as fallback for scribble")
        return self._simple_scribble(image)

    def _simple_scribble(self, image: Image.Image) -> Image.Image:
        """簡易スクリブル変換（フォールバック用）"""
        img_array = np.array(image)

        # グレースケールに変換
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # ガウシアンブラーで平滑化
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # エッジ検出
        edges = cv2.Canny(blurred, 30, 100)

        # 膨張処理で線を太く
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 3チャンネルに変換
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(edges_rgb)

    def process_tile(self, image: Image.Image) -> Image.Image:
        """Tile用の処理（画像をそのまま返す、または軽くぼかす）

        Args:
            image: 入力画像

        Returns:
            Tile用の制御画像
        """
        # Tileは元画像のディテールを維持するため、そのまま返すか軽い処理のみ
        # 軽いガウシアンブラーを適用してノイズを減らす
        img_array = np.array(image)
        blurred = cv2.GaussianBlur(img_array, (3, 3), 0)
        return Image.fromarray(blurred)

    def preprocess(
        self,
        image: Image.Image,
        preprocessor_type: str,
        **kwargs
    ) -> Image.Image:
        """指定されたタイプのプリプロセスを実行

        Args:
            image: 入力画像
            preprocessor_type: プリプロセッサの種類
                ("openpose", "canny", "depth", "lineart", "scribble", "tile")
            **kwargs: 追加のパラメータ

        Returns:
            プリプロセス結果の画像
        """
        if preprocessor_type == "openpose":
            return self.detect_openpose(image)
        elif preprocessor_type == "canny":
            low = kwargs.get("low_threshold", 100)
            high = kwargs.get("high_threshold", 200)
            return self.detect_canny(image, low, high)
        elif preprocessor_type == "depth":
            return self.estimate_depth(image)
        elif preprocessor_type == "lineart":
            return self.detect_lineart(image)
        elif preprocessor_type == "scribble":
            return self.detect_scribble(image)
        elif preprocessor_type == "tile":
            return self.process_tile(image)
        else:
            raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")


# グローバルインスタンス
preprocessors = ControlNetPreprocessors()
