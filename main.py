# main.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import argparse
import os
import sys
from PIL import Image, ImageTk, ImageOps
import numpy as np
import logging

# --- Configuration ---
# Assumes depth.py is in the same directory or accessible via PYTHONPATH
try:
    # Try direct import first, assuming depth.py is sibling or in path
    from depth import DepthEstimationHandler
    logging.info("Imported DepthEstimationHandler directly.")
except ImportError:
    logging.warning("Direct import failed. Modifying sys.path (this is less ideal)...")
    # Fallback: Adjust path if needed (e.g., if it's in a subdirectory)
    # Example: DEPTH_MODULE_FOLDER = "modules/ai"
    DEPTH_MODULE_FOLDER = "." # Current directory as default fallback
    base_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(base_dir, DEPTH_MODULE_FOLDER))
    if module_path not in sys.path:
         sys.path.insert(0, module_path)
    try:
         from depth import DepthEstimationHandler
         logging.info(f"Imported DepthEstimationHandler after adding {module_path} to path.")
    except ImportError as e_fallback:
         messagebox.showerror("Import Error", f"Failed to import DepthEstimationHandler.\nMake sure 'depth.py' is accessible.\nError: {e_fallback}")
         sys.exit(1)
except Exception as e:
    messagebox.showerror("Error", f"An unexpected error occurred during import: {e}")
    sys.exit(1)


# Configure logging (optional for GUI, useful for debugging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TransparencyApp:
    def __init__(self, master, image_path):
        self.master = master
        self.image_path = image_path
        self.original_pil_image = None
        self.depth_pil_image = None      # Store the depth map PIL image
        self.tk_original_image = None
        self.tk_transparent_image = None
        self.current_transparent_pil_image = None # Store the PIL version for saving

        self.master.title("Transparency Editor")
        self.master.configure(bg='#f0f0f0') # Light grey background

        # --- Load Image and Depth Map ---
        if not self._load_images():
            self.master.destroy() # Close window if loading fails
            return

        # --- GUI Elements ---
        # Frame for Images
        image_frame = ttk.Frame(master, padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True)

        # Original Image Display
        original_frame = ttk.Frame(image_frame, padding="5")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(original_frame, text="Original Image").pack()
        self.original_label = ttk.Label(original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        if self.tk_original_image:
            self.original_label.config(image=self.tk_original_image)

        # Transparent Image Display
        transparent_frame = ttk.Frame(image_frame, padding="5")
        transparent_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(transparent_frame, text="Transparent Preview").pack()
        # Use a Canvas for background color control
        self.transparent_canvas = tk.Canvas(transparent_frame, bg="lime green", bd=0, highlightthickness=0)
        self.transparent_canvas.pack(fill=tk.BOTH, expand=True)
        # We'll place the image onto the canvas later

        # Frame for Controls
        control_frame = ttk.Frame(master, padding="10")
        control_frame.pack(fill=tk.X)

        # Slider
        self.slider_label = ttk.Label(control_frame, text="Transparency Threshold (Lower depth = transparent): 0.00")
        self.slider_label.pack(side=tk.LEFT, padx=5)
        self.slider = ttk.Scale(control_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=self._update_transparency_from_slider)
        self.slider.set(0.5) # Default value
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Buttons Frame
        button_frame = ttk.Frame(master, padding="10")
        button_frame.pack(fill=tk.X)

        # Cancel Button
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.master.destroy)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)

        # Save Button
        self.save_button = ttk.Button(button_frame, text="Save Output Images", command=self._save_image) # Renamed button slightly
        self.save_button.pack(side=tk.RIGHT, padx=5)

        # --- Initial Transparency Update ---
        # Add slight delay for canvas to render before drawing? Not strictly needed.
        self.master.after(100, lambda: self._update_transparency(self.slider.get()))


    def _load_images(self):
        """Loads the original image and generates the depth map."""
        try:
            logging.info(f"Loading original image: {self.image_path}")
            # Make sure to release file handle with 'with'
            with Image.open(self.image_path) as img:
                 self.original_pil_image = img.copy() # Keep a copy

            # --- Depth Estimation ---
            logging.info("Initializing Depth Handler...")
            # Show a temporary loading message
            loading_label = ttk.Label(self.master, text="Estimating depth map, please wait...", font=("Arial", 12))
            loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            self.master.update_idletasks() # Make sure the label is shown

            try:
                depth_handler = DepthEstimationHandler()
                if not depth_handler.model:
                     raise RuntimeError("Depth model failed to initialize.")
                logging.info("Estimating depth map...")
                # Store the generated depth map
                self.depth_pil_image = depth_handler.estimate_depth(self.image_path)
            except Exception as e:
                 logging.error(f"Depth estimation failed: {e}", exc_info=True)
                 messagebox.showerror("Depth Error", f"Failed to estimate depth map:\n{e}")
                 return False
            finally:
                 loading_label.destroy() # Remove loading message

            if self.depth_pil_image is None:
                messagebox.showerror("Depth Error", "Depth estimation returned None.")
                return False

            # Ensure depth map is grayscale ('L') and matches original size
            if self.depth_pil_image.size != self.original_pil_image.size:
                logging.warning(f"Depth map size {self.depth_pil_image.size} differs from original {self.original_pil_image.size}. Resizing depth map.")
                self.depth_pil_image = self.depth_pil_image.resize(self.original_pil_image.size, Image.Resampling.BILINEAR)
            if self.depth_pil_image.mode != 'L':
                 self.depth_pil_image = self.depth_pil_image.convert('L')

            logging.info("Images loaded and depth map generated successfully.")

            # Prepare Tkinter-compatible image for original view
            self.tk_original_image = ImageTk.PhotoImage(self.original_pil_image)
            return True

        except FileNotFoundError:
            messagebox.showerror("Error", f"Input image not found: {self.image_path}")
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image or run depth estimation:\n{e}")
            logging.error(f"Error during image loading/depth: {e}", exc_info=True)
            return False

    def _update_transparency_from_slider(self, value_str):
        """Callback for the slider."""
        value = float(value_str)
        self.slider_label.config(text=f"Transparency Threshold (Lower depth = transparent): {value:.2f}")
        self._update_transparency(value)


    def _update_transparency(self, threshold_percent):
        """Applies transparency based on the depth map and threshold."""
        if self.original_pil_image is None or self.depth_pil_image is None:
            logging.warning("_update_transparency called before images loaded.")
            return

        try:
            # Convert original to RGBA to work with transparency
            img_rgba = self.original_pil_image.copy().convert("RGBA")
            depth_data = np.array(self.depth_pil_image) # Get depth values as numpy array
            img_data = np.array(img_rgba) # Get image data as numpy array

            # Calculate the threshold in the 0-255 range
            threshold_value = int(threshold_percent * 255)

            # Create a boolean mask where depth is below the threshold
            # Lower values in MiDaS depth maps usually mean closer objects
            # So, threshold sets the distance beyond which things become transparent
            # If you want NEARER things transparent, use `mask = depth_data > threshold_value`
            mask = depth_data < threshold_value

            # Apply the mask to the alpha channel (index 3)
            img_data[mask, 3] = 0   # Transparent where depth < threshold
            img_data[~mask, 3] = 255 # Opaque where depth >= threshold

            # Create a new PIL Image from the modified numpy array
            self.current_transparent_pil_image = Image.fromarray(img_data, 'RGBA')

            # Update the Tkinter image for display
            self.tk_transparent_image = ImageTk.PhotoImage(self.current_transparent_pil_image)

            # Clear previous image on canvas and draw the new one
            self.transparent_canvas.delete("all") # Clear canvas
            # Calculate position to center the image on the canvas
            canvas_width = self.transparent_canvas.winfo_width()
            canvas_height = self.transparent_canvas.winfo_height()
            img_width = self.tk_transparent_image.width()
            img_height = self.tk_transparent_image.height()

            # Get current canvas dimensions (might be 1x1 initially)
            self.master.update_idletasks() # Ensure dimensions are updated
            canvas_width = self.transparent_canvas.winfo_width()
            canvas_height = self.transparent_canvas.winfo_height()
            # Ensure minimum canvas size for calculation if not rendered
            if canvas_width <= 1: canvas_width = img_width
            if canvas_height <= 1: canvas_height = img_height

            x_pos = max(0, (canvas_width - img_width) // 2)
            y_pos = max(0, (canvas_height - img_height) // 2)

            # Place image on canvas (anchored at NW corner)
            self.transparent_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_transparent_image)


        except Exception as e:
            messagebox.showerror("Transparency Error", f"Failed to update transparency:\n{e}")
            logging.error(f"Error updating transparency: {e}", exc_info=True)


    def _save_image(self):
        """Saves the current transparent image AND the depth map."""
        if self.current_transparent_pil_image is None:
            messagebox.showwarning("Save Error", "No transparent image generated yet.")
            return
        if self.depth_pil_image is None:
             messagebox.showwarning("Save Error", "Depth map is not available to save.")
             return

        # --- Determine suggested filenames ---
        # Base name from the ORIGINAL input file
        original_basename = os.path.splitext(os.path.basename(self.image_path))[0]
        # Suggested filename for the transparent image
        suggested_trans_filename = f"{original_basename}_trans.png"
        # Suggested filename for the depth map
        suggested_depth_filename = f"{original_basename}_depth.png" # Will be saved in same dir

        # --- Ask user for location/name for the TRANSPARENT image ---
        save_path_trans = filedialog.asksaveasfilename(
            initialfile=suggested_trans_filename,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Transparent Image As"
        )

        if not save_path_trans:
            logging.info("Save cancelled by user.")
            return # User cancelled

        # --- Determine final save paths ---
        chosen_dir = os.path.dirname(save_path_trans)
        # Use the exact path chosen by user for transparent image
        final_trans_path = save_path_trans
         # Create the depth map path in the same chosen directory
        final_depth_path = os.path.join(chosen_dir, suggested_depth_filename)

        # --- Perform Saves ---
        saved_files = []
        try:
            # Save Transparent Image
            self.current_transparent_pil_image.save(final_trans_path, "PNG")
            logging.info(f"Transparent image saved successfully to: {final_trans_path}")
            saved_files.append(final_trans_path)

            # Save Depth Map Image
            try:
                 self.depth_pil_image.save(final_depth_path, "PNG")
                 logging.info(f"Depth map image saved successfully to: {final_depth_path}")
                 saved_files.append(final_depth_path)
            except Exception as e_depth:
                 logging.error(f"Failed to save depth map image to {final_depth_path}: {e_depth}", exc_info=True)
                 # Notify user but don't necessarily fail the whole operation if transparent save worked
                 messagebox.showerror("Depth Save Error", f"Failed to save depth map image to:\n{final_depth_path}\n\nError: {e_depth}")


            # --- Success Message ---
            if saved_files:
                 messagebox.showinfo("Save Successful", f"Image(s) saved successfully:\n\n" + "\n".join(saved_files))

            # Optionally close after saving:
            # self.master.destroy()

        except Exception as e_trans:
            # Handle error during transparent image save specifically
            messagebox.showerror("Save Error", f"Failed to save transparent image to {final_trans_path}:\n{e_trans}")
            logging.error(f"Failed to save transparent image: {e_trans}", exc_info=True)
        # Any error saving depth map is handled inside the inner try-except


def main():
    parser = argparse.ArgumentParser(description="Create a transparent image based on AI depth map.")
    parser.add_argument("image_path", help="Path to the input image file.")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        # Attempt to show error in message box if GUI libs are available, else print
        try:
            root = tk.Tk()
            root.withdraw() # Hide the main window
            messagebox.showerror("Error", f"Input image not found at:\n'{args.image_path}'")
            root.destroy()
        except Exception:
             print(f"Error: Input image not found at '{args.image_path}'")
        sys.exit(1)

    root = tk.Tk()
    # Set minimum size for the window
    root.minsize(600, 400)
    app = TransparencyApp(root, args.image_path)
    # Handle window closing cleanly
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

if __name__ == "__main__":
    main()