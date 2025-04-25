# Movie Maker AI/modules/ai/depth.py

import torch
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from PIL import Image
import numpy as np
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DepthEstimationHandler:
    """
    Handles Monocular Depth Estimation using Intel's DPT models.
    """
    def __init__(self, model_id="Intel/dpt-hybrid-midas"):
        self.model_id = model_id
        self.model = None
        self.feature_extractor = None
        self.device = None
        self.dtype = None # DPT typically runs well in float32 even on GPU

        # --- Device Setup ---
        if torch.cuda.is_available():
            self.device = "cuda"
            # DPT models often work fine and fast in float32, but float16 is an option
            self.dtype = torch.float32 # Or torch.float16
            logging.info(f"CUDA device found: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            logging.warning("CUDA not available. Depth Estimation will run on CPU (will be slower).")

        logging.info(f"Initializing Depth Estimation Handler with model: {self.model_id}")
        logging.info(f"Target device: {self.device}, Compute dtype: {self.dtype}")

        try:
            self.feature_extractor = DPTFeatureExtractor.from_pretrained(self.model_id)
            self.model = DPTForDepthEstimation.from_pretrained(
                self.model_id,
                low_cpu_mem_usage=True,
                torch_dtype=self.dtype # Load model with target dtype
                )
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            logging.info("DPT Depth Estimation model and feature extractor loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load DPT model or feature extractor: {e}", exc_info=True)
            raise

    def estimate_depth(self, image_path: str) -> Image.Image | None:
        """
        Estimates the depth map for a single image file.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            PIL.Image.Image | None: A PIL Image representing the normalized
                                    depth map (0-255 grayscale), or None if failed.
        """
        if not self.model or not self.feature_extractor:
            logging.error("Depth Estimation model/extractor not initialized.")
            return None
        if not os.path.exists(image_path):
            logging.error(f"Input image file not found: {image_path}")
            return None

        try:
            image = Image.open(image_path).convert("RGB")
            original_size = image.size

            logging.info(f"Processing image for depth estimation: {image_path}")
            # Prepare image for the model
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            # Move inputs to the correct device
            # Handle dtype explicitly for pixel_values
            pixel_values = inputs['pixel_values'].to(self.device, dtype=self.dtype)
            inputs = {'pixel_values': pixel_values}


            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original size
            # Use torch.float32 for interpolation input for stability, then move back if needed
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1).float(), # Ensure float32 for interpolate
                size=original_size[::-1], # PIL size is (width, height), interpolate needs (height, width)
                mode="bicubic",
                align_corners=False,
            )

            # Squeeze, move to CPU, convert to numpy
            output = prediction.squeeze().cpu().numpy()

            # Normalize the depth map to 0-255 for visualization
            # Avoid division by zero if the output is all zeros
            max_val = np.max(output)
            if max_val > 0:
                 formatted = (output * 255 / max_val).astype("uint8")
            else:
                 formatted = np.zeros_like(output, dtype="uint8")

            depth_image = Image.fromarray(formatted).convert('L') # Ensure it's grayscale

            logging.info("Depth estimation successful.")
            return depth_image

        except Exception as e:
            logging.error(f"Error during depth estimation for {image_path}: {e}", exc_info=True)
            return None

    def estimate_and_save_depth(self, image_path: str, output_path: str) -> bool:
        """
        Estimates depth for an image and saves the normalized map.

        Args:
            image_path (str): Path to the input image file.
            output_path (str): Path to save the output depth map (PNG).

        Returns:
            bool: True on success, False on failure.
        """
        start_time = time.time()
        depth_map_image = self.estimate_depth(image_path)
        if depth_map_image:
            try:
                # Ensure output directory exists
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                depth_map_image.save(output_path, "PNG")
                end_time = time.time()
                logging.info(f"Depth map saved to: {output_path} (took {end_time - start_time:.2f}s)")
                return True
            except Exception as e:
                logging.error(f"Failed to save depth map to {output_path}: {e}", exc_info=True)
                return False
        else:
            return False

# Example usage (optional, for direct testing)
# if __name__ == "__main__":
#     print("Testing DepthEstimationHandler...")
    # Create a dummy input image file path for testing structure
    # Replace with a real image path if you want to run the test directly
    # test_input_image = "output/images/generated_sdturbo_image_01.png" # Assumes image gen ran
    # test_output_depth = "output/depth/test_depth_map.png"

    # if os.path.exists(test_input_image):
    #     try:
    #         handler = DepthEstimationHandler()
    #         if handler.model:
    #             print("\n--- Depth Estimation Test ---")
    #             success = handler.estimate_and_save_depth(test_input_image, test_output_depth)
    #             if success:
    #                 print(f"Test depth map generated: {test_output_depth}")
    #             else:
    #                 print("Test depth map generation failed.")
    #         else:
    #             print("DepthEstimation Handler failed to initialize model.")
    #     except Exception as e:
    #         print(f"DepthEstimationHandler test failed: {e}")
    # else:
    #     print(f"Skipping Depth Estimation test: Input image not found at '{test_input_image}'. Run image generation test first.")