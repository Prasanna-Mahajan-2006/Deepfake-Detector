
# DeepFake Image Classifier Using MobileNet V3 Small

This Project is undertaken as a Term Paper for the course on Digital Signal Processing at IIT Delhi. We aim to implement a robust, lightweight CNN model based on the MobileNet V3-Small Architecture to efficiently classify DeepFake Images and quantify the accuracy of the model, with scope to extend the model to a wider class of detection problems. We have utilized several different approaches to making the model associated with the project more accurate to predicting real-world DeepFake images, which are a major cause of concern considering their adverse impacts on society including identitiy theft and as a means to spread fake, misleading and possibly harmful information.


## Features

- Dynamic Custom Data Generator: Features a highly optimized Keras Sequence generator that calculates 2D Fast Fourier Transforms and log based magnitude spectra during training, saving significant disk space and pre-processing time.
- Overfit Protection: We have implemented several regularization techniques including dynamic learning rate reduction (ReduceLROnPlateau), early stopping, dropout layers, and L2 kernel penalties.
- Evaluation Matrices: Generation of Sklearn Classification Reports (Precision, Recall, F1-Score) and visual Seaborn Confusion Matrices evaluated on strictly unshuffled validation data.
- Real Testing: Notebook 3 (FFT + MobileNet) also consists of a block of code that could be run to feed any real/fake image to a trained model and get the model prediction alongside the confidence score.
## Tools & Technologies Used

- TensorFlow / Keras: Model architecture, Transfer Learning, and for implementation of custom/modified sequential models
- OpenCV (cv2): Image processing, color-space conversions, and resizing.
- NumPy: Advanced matrix operations and FFT calculations.
- Scikit-Learn: Post-training evaluation (Classification Reports, Confusion Matrices).
- Matplotlib & Seaborn: Visualizing evaluation metrics and FFT spectrum heatmaps.

- Kagglehub: Seamless, secure dataset integration.
## Installation

The necessary libraries used in this project can be installed with the following commands. To run these notebooks, you will need a Python environment (Python 3.8+ recommended) with the following libraries installed. If you are using Google Colab, most of these are are already available, but kagglehub will have to be installed:

```bash
  pip install tensorflow opencv-python numpy pandas scikit-learn matplotlib seaborn kagglehub
```
## Documentation

This project aims to implement a DeepFake detetction system for mobile devices using lightweight architectures. We have utilized a modular Notebook Approach in this project, wherein the project is divided into dedicated, self-contained notebooks, allowing users to trace the architectural evolution from a baseline model to the final hybrid solution:
- Notebook 1: Baseline Regularized CNN: Establishes a foundational deep learning baseline using standard convolutional layers and Regularization methods along with image pre-processing.
- Notebook 2: Transfer Learning Approach: Explores spatial-domain feature extraction by fine-tuning a pre-trained MobileNetV3-Small architecture tail, and a modified, trained classification head suited to the specific problem.
- Notebook 3: The Hybrid Dual-Stream Model (Transfer Learning + FFT): The final implementation for the project. It merges the fine-tuned spatial MobileNet with a custom Deep FFT convolutional branch, analyzing both physical pixels and spectral artifacts generally left out by Generative Adversarial Networks (GANs) during generation of DeepFake images.
- Notebook 4: Experuimental Ablation Study: Although not a well-working implementation, we have triwd to merge the FFT based approach with attention mechanisms in this approach. Future analysis of this notebook will be undertaken, and suggestions are very welcome.

Within this modular approach, the best results were obtained from the Transfer Learning approach and the FFT approach. The Ablation study which implements Attention Mechanisms, although giving high accuracy on training dataset, has high probability of overfitting, since the dataset utilized is relatively small.

The important functions used in this project across all the notebooks are documented below for ready reference, though this list is certainly not exhaustive, and several functions and methods from the tensorflow library were utilized in building the project:

### Notebook 1: Baseline Regularized CNN
| Function | Parameters | Return Value | Description |
| :--- | :--- | :--- | :--- |
| `build_baseline_cnn` | `img_size` (tuple, default=(224,224)) | `tf.keras.Model` | Constructs a standard, spatial-only Convolutional Neural Network from scratch. Implements baseline regularization techniques (Dropout, L2) to efficiently train the model while reducing chances of overfitting. |

### Notebook 2: Transfer Learning (MobileNetV3)
| Function | Parameters | Return Value | Description |
| :--- | :--- | :--- | :--- |
| `build_mobilenet_model` | `img_size` (tuple, default=(224,224)) | `tf.keras.Model` | Instantiates the MobileNetV3-Small backbone with ImageNet weights. Freezes the lower layers to act as feature extractors, while the more advanced features and later on the classification task are handled by the cutom trained 'head' of the model. |

### Notebook 3: Hybrid Dual-Stream Model (Transfer Learning + FFT)
| Class/Function | Parameters | Return Value | Description |
| :--- | :--- | :--- | :--- |
| `DualStreamGenerator` | `directory` (str), `batch_size` (int), `img_size` (tuple), `shuffle` (bool) | `dict`, `np.array` | A custom Keras `Sequence` class. Yields a dictionary containing `spatial_rgb_input` and `frequency_fft_input` alongside an array of binary labels. |
| `__getitem__` | `idx` (int) | `dict`, `np.array` | Core generator method. Dynamically computes 2D FFT, center-shifting, and log-magnitude scaling on the CPU for a specific batch. |
| `build_optimized_hybrid_model`| `img_size` (tuple, default=(224,224)) | `tf.keras.Model` | Constructs the final dual-stream architecture. Merges the fine-tuned MobileNetV3 (top 79 layers unfrozen) with a 4-block deep CNN for the FFT stream, applying 50% Dropout and L2 regularization. |
| `preprocess_for_inference` | `image_path` (str), `img_size` (tuple) | `dict`, `np.array` | Reads a single unseen image and applies identical spatial normalization and FFT log-magnitude scaling as the training generator, expanding dimensions for Keras batch constraints. |
| `analyze_image` | `image_path` (str) | None | Wraps the preprocessing and prediction flow. Outputs a terminal classification ("REAL" or "FAKE"), calculates confidence percentages. |

### Notebook 4: Experimental Ablation Study (FFT + Attention)
| Function | Parameters | Return Value | Description |
| :--- | :--- | :--- | :--- |
| `build_attention_hybrid_model`| `img_size` (tuple, default=(224,224)) | `tf.keras.Model` | Constructs the experimental architecture utilizing CBAM/SE Attention mechanisms. Used to demonstrate and diagnose severe model overfitting on localized background noise. |
| `build_pure_fft_mobilenet` | `img_size` (tuple, default=(224,224)) | `tf.keras.Model` | Freezes the spatial MobileNet backbone entirely, forcing the model to learn exclusively through the FFT stream to isolate the impact of frequency-domain features. |


## Usage: How To Run

- Clone the repository to your local machine or open the notebooks directly in Google Colab.

- It is recommended to run the notebooks sequentially starting with the baseline model implementation to understand the academic progress of the project. To experiment with the final evaluation metrics, we recommend to use the notebook which fits your requirements on accuracy and training best.

- Data Loading: The dataset is automatically downloaded via kagglehub in the first cell of the data pipeline. No manual downloading is required.

- Inference: In ------, navigate to the final cell. You can supply the path to any unseen face image (Real or Fake), and the model will output a prediction alongside a confidence percentage and visual heatmap.

## References & Core Literature
This project's architecture and theoretical foundation were heavily inspired by recent advancements in multimedia forensics and lightweight convolutional networks. Below are the key research papers that informed our methodology:
- [Sidnal et al., "Efficient DeepFake Image Classification using
MobileNetV4-Small," INCOFT 2025.](https://www.scitepress.org/Link.aspx?doi=10.5220/0013613400004664)
- [M. Karki, "Deepfake and Real Images Dataset," Kaggle, 2024.](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
- [K. He et al., "Deep Residual Learning for Image Recognition,"
CVPR 2016.](https://www.researchgate.net/publication/286512696_Deep_Residual_Learning_for_Image_Recognition)
- ["LightFakeDetect: A Lightweight Model for Deepfake
Detection in Videos," 2023.](https://www.bohrium.com/en/paper-details/lightfakedetect-a-lightweight-model-for-deepfake-detection-in-videos-that-focuses-on-facial-regions/1178539714571927568-1106)
- [Tan and Le, "EfficientNet: Rethinking Model Scaling for
CNNs," ICML 2019.](https://openreview.net/pdf?id=S1jBcueAb)
- ["Deepfake Detection using EfficientNetV2-B2," 2023.](https://www.bohrium.com/en/paper-detailsdeepfake-detection-evaluating-the-performance-of-efficientnetv2-b2-on-real-vs-fake-image-classification/1151615473679335425-2717)
- ["ShuffleNeMt: Modern Lightweight CNN Architecture," 2023.](https://www.bohrium.com/en/paper-details/shufflenemt-modern-lightweight-convolutional-neural-network-architecture/1046259363544563721-2752)
- ["Deepfake Detection using MobileNetV2 with SVM," 2023](https://www.bohrium.com/en/paper-details/shufflenemt-modern-lightweight-convolutional-neural-network-architecture/1046259363544563721-2752)
- [Lightweight Deepfake Detection on Mobile Devices Using Attention
Enhanced MobileNet and Frequency Domain Analysis](https://www.researchgate.net/publication/390978624_Lightweight_Deepfake_Detection_on_Mobile_Devices_Using_Attention-Enhanced_MobileNet_and_Frequency_Domain_Analysis)
## Authors

- [Prasanna Prasad Mahajan](https://github.com/Prasanna-Mahajan-2006/Deepfake-Detector)
- [Abhineet Milind More](https://github.com/Abhineet75)
- [Ayush Tomar](https://github.com/AyushTomar1963)
- [Kirti Tekriwal]()
- [Tiya Mittal](https://GitHub.com/tiya27)
