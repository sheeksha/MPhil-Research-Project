# MPhil-Research-Project
**My research project for the Master in Philosophy (MPhil) degree in Remote Sensing with Artificial Intelligence.**
🚢 Ship Detection in the Exclusive Economic Zone of Mauritius in Sentinel-2 Imagery using CNNs & Yolov11

## 🙏 Acknowledgements

This project would not have been possible without the incredible people who supported me along the way. First, I’m grateful to my supervisors — Assoc. Prof. Heeralall-Issur N., Prof. Cullen D., and Dr. Beeharry Y., for their steady guidance, thoughtful feedback, and patience throughout the research journey. To my family, my mother Anita, my brothers Ranveer, Yashwant, and Vrisht, and my sister-in-law Youri, thank you for believing in me and cheering me on through every challenge. My friends Heerveen, Dinesh, and Nawshine also deserve special mention for their encouragement and for keeping me motivated when things got tough. A big thanks as well to Ruben Louis for his timely technical support. 

I’m also thankful to the **DARA** (Development in Africa with Radio Astronomy) scholarship, which funded and enabled this research, and to the examiners whose detailed feedback pushed me to refine my approach and reach better results. A special shoutout goes to new supporters I met along the way, like Sofia, whose encouragement during a challenging transition helped me stay the course. And finally — yes, I’ll admit it — even ChatGPT deserves a thank-you. Its help with debugging, research, and drafting made the process faster (and less painful).


## Overview

Maritime surveillance is important for national security, economic resilience, and environmental protection. For island nations like Mauritius, whose Exclusive Economic Zone (EEZ) spans an impressive 2.3 million km², the third largest in Africa and the 25th largest worldwide, safeguarding maritime borders is not only a matter of sovereignty but also of survival. The Mauritian EEZ lies across busy international shipping routes connecting Asia to Africa and beyond, with vessels frequently carrying hazardous cargo and large volumes of goods. This geo-strategic position presents unique challenges, including illegal fishing, piracy, narcotics trafficking, marine pollution, and accident prevention.
  
Traditional surveillance methods such as naval patrols and radar monitoring, while effective, have limited coverage. Satellite imagery offers a scalable and cost-effective alternative, providing continuous monitoring over vast ocean areas. Both optical imagery (visible spectrum) and Synthetic Aperture Radar (SAR) have proven valuable. Optical imagery captures fine details of ships during clear conditions, while SAR enables detection at night or through clouds. However, extracting meaningful insights from these large volumes of satellite data remains a challenge.

Conventional ship detection techniques often rely on handcrafted features and classical classifiers, which can be brittle under variations in ship size, orientation, or atmospheric conditions. Recent advances in deep learning, particularly Convolutional Neural Networks (CNNs) and real-time detectors such as YOLO, provide a promising alternative. These models excel at learning complex patterns directly from data, offering improved robustness and accuracy compared to traditional methods.

This study leverages Sentinel-2 optical satellite imagery, chosen for its free accessibility, 10-meter resolution, and five-day revisit frequency, making it suitable for monitoring ships traversing the Mauritian EEZ, where a cargo vessel typically requires six days to cross its full extent. Using Sentinel-2 data ensures not only affordability but also the ability to build a custom dataset tailored to local conditions, including coral reefs, “white water” from wave turbulence, and frequent cloud cover.


The research specifically evaluates and compares state-of-the-art CNN architectures (such as VGG-16, ResNet-50, Inception-V3, DenseNet, EfficientNet, and MobileNet) against the YOLOv11 framework. By applying transfer learning, the models were fine-tuned on a custom dataset, enabling effective training despite limited labeled data. Performance was measured in terms of precision, recall, F1-score, and inference speed, balancing accuracy against computational efficiency. Beyond technical performance, the study also explores broader implications:
- How well deep learning models generalize to other datasets and geographic regions.
- The trade-offs between high accuracy and real-time detection needs.
- The interpretability of model outputs for end-users like maritime authorities.
- The potential role of satellite-based ship detection systems in enhancing Maritime Domain Awareness (MDA) in small developing states.

To summarise, this research demonstrates that CNNs and YOLO offer significant potential for affordable, scalable, and effective maritime surveillance in Mauritius and similar contexts. While challenges remain in terms of dataset availability, computational costs, and real-world deployment, the results highlight a viable path toward integrating AI-driven ship detection into national defense and environmental monitoring strategies.


### Available overview of optical satellite datasets utilised for ship detection tasks.

This collection highlights the diversity of datasets used in maritime object detection, ranging from high-resolution commercial data to publicly available data with varying revisit times and resolutions. 

<p align="center">
  <img width="545" height="720" alt="image" src="https://github.com/user-attachments/assets/2868b308-bcf1-46f2-a05f-824ea73b8bbc" />
  <br>
  <em>Table 1. A comprehensive overview of optical satellite datasets utilised for ship detection tasks. This collection highlights the diversity of datasets used in maritime object detection, ranging from high-resolution commercial data to publicly available data with varying revisit times and resolutions. [Source: https://github.com/JasonManesis/Satellite-Imagery-
Datasets-Containing-Ships]</em>
</p>


### 🛰️ Sentinel-2 Satellite

The Sentinel-2 mission, operated by the European Space Agency (ESA), is a medium- to high-resolution Earth observation program designed for environmental monitoring, land use analysis, and maritime surveillance. It consists of twin satellites in the same orbit, phased 180° apart, ensuring a revisit time of five days at the Equator. Each satellite is equipped with the Multi-Spectral Instrument (MSI), capturing data in 13 spectral bands: four at 10 m, six at 20 m, and three at 60 m resolution, across a swath width of 290 km.

Sentinel-2 data are freely available and provide a cost-effective alternative to commercial imagery, which can be prohibitively expensive for small island states. For this research, Sentinel-2 was chosen as the optimal balance of spatial resolution, temporal frequency, and accessibility, making it well suited for monitoring large maritime areas such as Mauritius’ Exclusive Economic Zone (EEZ).

The satellites collect data over land, coastal regions, and islands worldwide, including inland water bodies and closed seas. Data are provided in two main product levels:
- Level-1C: Top-Of-Atmosphere (TOA) reflectance, orthorectified for geometric accuracy. (~600 MB per 100×100 km² tile)
- Level-2A: Bottom-Of-Atmosphere (BOA) reflectance, atmospherically corrected for surface-level analysis. (~800 MB per tile)

Both products are distributed as standardised 100×100 km² ortho-image tiles in the Universal Transverse Mercator (UTM/WGS84) projection, ensuring spatial consistency and seamless integration with other geospatial datasets.

The UTM projection is critical for precise vessel detection, as it allows Sentinel-2 imagery to align with maritime boundaries, AIS vessel data, and port infrastructure. For Mauritius, whose vast EEZ spans multiple UTM zones, this ensures reliable monitoring of shipping lanes, vessel movements, and potential threats.



<p align="center">
  <img width="728" height="386" alt="image" src="https://github.com/user-attachments/assets/2bf1711b-a57b-497d-a917-cba24b96f325" />
  <br>
  <em>Figure 1. Schematic representation of a Sentinel-2 satellite, highlighting its flight direction and nadir pointing orientation. The satellite’s solar panels and onboard instruments are visible, designed for Earth observation applications, including optical imaging for environmental monitoring and maritime surveillance. [Image credit: ESA] </em>
</p>

## 📚 Literature Review: Advances in Ship Detection

Ship detection in satellite imagery has evolved from traditional image processing methods with handcrafted features (HOG, SIFT) to deep learning-based approaches that learn robust patterns directly from data.

### Convolutional Neural Networks (CNNs)
- CNNs introduced automated feature extraction, making them far more effective than traditional methods in noisy maritime environments.
- Two-stage CNN frameworks (e.g., R-CNN family) achieve high precision but are computationally expensive.
- Single-stage CNN frameworks (e.g., SSD, MobileNet variants) are more efficient, enabling near real-time applications.
- Specialised innovations such as multi-scale feature fusion and attention mechanisms improved detection across diverse ship sizes and complex backgrounds.

### YOLO (You Only Look Once)
- YOLO revolutionised real-time detection by combining classification and localisation in a single forward pass.
- Recent versions (YOLOv5–YOLOv11) integrate transformer modules, spatial attention, and multi-scale detection, making them highly effective in maritime contexts.
- Studies applying YOLO to Sentinel-2 imagery have achieved strong precision/recall trade-offs, making it suitable for operational monitoring.

### Foundation Models & Emerging Trends
- Foundation models (DETR, DINO, Florence-2) leverage large-scale pretraining to generalise across tasks with minimal labelled data.
- They capture global context, making them robust in noisy environments with clouds, waves, and occlusions.
- Limitations include high computational costs, domain adaptation challenges, and interpretability issues.

### Key Insights
- CNNs excel in accuracy but are resource-heavy.
- YOLO balances speed and accuracy, ideal for real-time ship monitoring.
- Foundation models offer scalability and few-shot learning but require heavy compute.
- Research highlights a trade-off between precision and efficiency, shaping how models are chosen for maritime surveillance.

## 🌍 Concerned Area
The study focuses on the maritime zones of the Republic of Mauritius, located within the FAO Major Fishing Area 51. The key regions include:
- Mauritian Exclusive Economic Zone (EEZ)
- Joint Management Area (JMA) jointly managed with Seychelles
- Chagos Archipelago EEZ
- Two major global shipping routes that cross the Indian Ocean

To estimate the longest path a vessel can take across the Mauritian maritime zone, the distance between Point 1 and Point 2 was calculated using the Haversine formula, resulting in an approximate 3,550 km route. Assuming a bulk carrier traveling at 13 knots (24.07 km/h), the estimated travel time across this route is:
- 147.45 hours
- ≈ 6.14 days

- ### 🧮 Distance Calculation (Haversine Formula)
The distance \( d \) between two geographic points is calculated using the Haversine formula:
```
\[
d = 2r \, \arcsin\left(
\sqrt{
\sin^2\left(\frac{\varphi_2 - \varphi_1}{2}\right)
+
\cos(\varphi_1)\cos(\varphi_2)
\sin^2\left(\frac{\lambda_2 - \lambda_1}{2}\right)
}
\right)
\]

Where:  
- \( r \) is the Earth’s radius (≈ 6371 km)  
- \( \varphi_1, \varphi_2 \) are latitudes in radians  
- \( \lambda_1, \lambda_2 \) are longitudes in radians  

```
This reinforces the use of satellite imagery with a 6-day revisit frequency for effective ship monitoring, ensuring multiple detections along the route and improving surveillance reliability.

<p align="center">
  <img width="817" height="533" alt="image" src="https://github.com/user-attachments/assets/387caeaa-93d8-4846-a8b3-c85a1b7dd0dc" />
  <br>
  <em>Figure 2: Maritime zones of the Republic of Mauritius, including the EEZ, JMA, and major shipping routes.</em>
</p>

## 📂 Dataset Creation for CNN

Since no ready-to-use Sentinel-2 ship detection dataset was available, a custom dataset was built from scratch using Sentinel-2 optical imagery collected between March 2016 and March 2021.


<p align="center">
  <img width="700" height="479" alt="image" src="https://github.com/user-attachments/assets/81b797dd-6867-4580-939b-7d6daf57ef3c" />
  <br>
  <em>Figure 2. Sentinel-2 optical satellite imagery mosaic overlaid with the Exclusive Economic Zone (EEZ) of Mauritius. It illustrates the scarce coverage over whole EEZ.</em>
</p>

### Steps Taken
- Image Collection:
  - Downloaded 98 Sentinel-2 images in .SAFE format.
  - Each contained 13 bands; RGB images were generated manually by stacking B02 (Blue), B03 (Green), and B04 (Red) bands instead of using the prebuilt TCI, to ensure consistency.

- Ship Extraction:
  - Images were analysed in QGIS 3.10.
  - 522 ship instances were manually identified and cropped.

- Image Preprocessing:
  - Crops were resized to 49×49 pixels, chosen to capture the largest visible ships in Sentinel-2 data.
  - To improve classification accuracy, the dataset was expanded to five classes:
    - Ship (522)
    - Cloud (521)
    - Sea (123)
    - Land (159)
    - Coast (60)
  - Total: 1385 labelled instances.

<p align="center">
  <img width="758" height="608" alt="image" src="https://github.com/user-attachments/assets/169844a3-2bd7-4a0e-a38b-85676b01712f" />
  <br>
  <em>Figure 3. Sample images from the CNN-based dataset, illustrating the five object classes: Sea [0], Ship [1], Cloud [2], Land [3], and Coast [4], along with their corresponding labels.</em>
</p>

- Annotation Challenges:
  -  Many annotation tools didn’t support large JPEG2000 files or raw band stacking.
  -  QGIS was chosen as the annotation tool since it could process the raw bands directly and generate raster layers without quality loss.

### Why this approach?
- Unlike large public datasets (e.g., xView, DIOR), this dataset was tailored specifically to Sentinel-2 imagery and the Mauritian EEZ context.
- Instead of fine-grained ship classification (e.g., tankers, fishing vessels), the focus was on distinguishing ships from common maritime backgrounds like sea, land, clouds, and coastlines.


## 📂 Dataset Creation for YOLO

For the YOLO-based approach, a custom dataset was created from Sentinel-2 imagery (March 2019 – March 2021) to enable real-time ship detection.

### Steps Taken
- Image Collection & Preprocessing:
  - Used 98 Sentinel-2 raw tiles, each 10,000 × 10,000 pixels.
  - Split them into smaller 2,000 × 2,000 sub-tiles to maintain ship visibility while making images manageable for YOLO.
  - From 1,501 sub-tiles generated, 156 relevant sub-tiles were selected for annotation.

- Annotation:
  - Ships were annotated with bounding boxes using LabelImg.
  - Annotations followed YOLO’s required format:
      class center_x center_y width height (all coordinates normalised by image dimensions).

  - A single class (“Ship”) was used to avoid class imbalance and focus the model on vessel detection.

- Dataset Composition:
  - 504 total ship instances across 156 annotated images.
  - Train/validation/test split: 70% / 15% / 15%.
    - Training: 109 images (349 ships)
    - Validation: 24 images (73 ships)
    - Test: 23 images (82 ships)


### Why this approach?
- Unlike the CNN dataset (cropped 49×49 patches), YOLO requires full images with bounding boxes, enabling it to learn both localisation and classification in one pass.
- Using sub-tiles ensured that ships remained visible without overwhelming GPU memory.
- A single-class setup simplified the pipeline while directly aligning with the research goal: detecting ships in Sentinel-2 imagery.


## 🧠 CNN-Based Methodology
The CNN approach was designed to detect ships in Sentinel-2 imagery using transfer learning from state-of-the-art (SOTA) convolutional architectures. The workflow can be summarised as follows:

### Models Used
Eight CNN models were evaluated:
- VGG-16 – baseline, simple 3×3 conv filters, frozen feature extractor + new classifier.
- ResNet-50 – residual connections, added dropout, tuned dense layers.
- Inception-V3 – parallel filters, batch norm, increased training epochs.
- NASNetMobile – neural architecture search (lightweight, efficient).
- EfficientNetB0 – compound scaling for balanced efficiency.
- DenseNet-121 – dense connectivity for gradient flow.
- MobileNet-V2 – lightweight, designed for real-time/mobile use.
- SimpleNet2021 – custom-built CNN trained from scratch as a baseline.


### Experimental Setup
- Environment: Python + TensorFlow 2 on GPU (NVIDIA GTX 1660 Ti).
- Dataset: 1,385 labelled images (5 classes: Ship, Cloud, Land, Sea, Coast).
  - Train/test split = 1067 / 318 images.

- Preprocessing:
  - Pixel values normalised (0–1).
  - Labels one-hot encoded.
  - Resized to match input requirements (e.g., 224×224 for transfer learning).

### Training & Compilation
- Transfer Learning: Pre-trained ImageNet weights with frozen base layers; new classifier layers added.
- Loss Function: Categorical Cross-Entropy.
- Optimiser: Adam / SGD with tuned learning rates.
- Evaluation Metrics: Precision, Recall, F1-score, Accuracy, Confusion Matrix, Classification Report.


### Detection on Large Images
- Used sliding window (49×49 crops) to scan 2000×2000 scene images.
- Predictions combined into detection maps.
- Applied Non-Maximum Suppression (NMS) to remove duplicate bounding boxes.


### Hyperparameter Optimisation (HPO)
- Random Search used instead of grid search (less computationally heavy).
- Adjustments included neuron counts, dropout rates, and epochs per model.


### Data Augmentation
- Initially tested (rotation, flipping, scaling), but degraded model performance.
- Omitted from final methodology to keep data realistic.


## 🚀 YOLOv11-Based Methodology
To overcome the limitations of CNNs (sliding windows, slow inference, limited localisation), a YOLOv11 Nano model was implemented for real-time ship detection in Sentinel-2 imagery. Unlike CNN pipelines, YOLO directly predicts bounding boxes and class labels in a single forward pass, making it efficient for maritime surveillance tasks.


### Why YOLOv11?
- Handles small objects like ships effectively using spatial pyramid pooling and attention mechanisms.
- Lightweight and deployable on resource-constrained devices (e.g., drones, CubeSats).
- Supports real-time detection with high accuracy.


### Experimental Setup
- Framework: Ultralytics YOLOv11, built on PyTorch.
- Dataset: Custom Sentinel-2 dataset (156 annotated images, 504 ship instances).
- Input Size: 640×640 pixels (balanced resolution vs. memory).
- Annotation Format: YOLO text files → <class> <x_center> <y_center> <width> <height> (normalised).
- Hardware: Trained on Google Colab (Tesla T4 GPU) for speed and scalability.


### Training Configuration
- Epochs: 55 (avoided under/overfitting).
- Batch Size: 16.
- Optimiser: AdamW (auto-configured by Ultralytics).
- Learning Rate: 0.002 with momentum 0.9.
-  Data Augmentation: Mosaic, flips, rotations, scaling applied to improve generalisation.


### Key Loss Functions Monitored
- Box Loss → error in bounding box localisation.
- Cls Loss → classification accuracy for “ship”.
- Dfl Loss → confidence/stability in box predictions.


### Evaluation Metrics
- Precision, Recall, F1-Score → classification quality.
- IoU & mIoU → bounding box overlap accuracy.
- AP & mAP50 / mAP[50–95] → average precision at multiple thresholds.
- ROC & Precision-Recall curves → assessed class imbalance and trade-offs.


### Challenges & Fine-Tuning
- No True Negatives: Initially, test data only had ships. TN images were later added to allow full confusion matrix evaluation.
- Missed Detections: Model sometimes failed on small/occluded ships → mitigated with augmentation & IoU threshold tuning.
- Hyperparameter Adjustments: Tweaked learning rate, batch size, weight decay for better convergence.
- Early Stopping: Stopped training if validation performance plateaued.


### Outcome
YOLOv11 successfully demonstrated fast and accurate ship detection with potential for real-time maritime monitoring. Despite dataset limitations, the model outperformed CNN approaches in efficiency and scalability, marking it as a practical framework for Maritime Domain Awareness (MDA) applications.


## 📊 Results & Discussion – CNN-Based Approach
### Model Performance
Eight CNN architectures were tested on the custom Sentinel-2 dataset (5 classes: Ship, Cloud, Sea, Land, Coast).

- Top Performers:
  - ResNet-50 → Highest validation accuracy (97.8%) but signs of overfitting.
  - VGG-16 & Inception-V3 → Strong validation accuracy (95.6%) with more stable convergence.
  - EfficientNetB0 → Balanced accuracy (97.1%) and efficiency.
  - MobileNet-V2 → Competitive (93.1%) with excellent computational efficiency.

- Weaker Models:
  - DenseNet-121 → Oscillations, less stable.
  - NASNetMobile → Poor generalisation (82.1% validation accuracy).
  - SimpleNet2021 → Lightweight but limited performance (<80% accuracy).

### Confusion Matrix Insights
- ResNet-50 → Strongest overall, high true positives (TP) and true negatives (TN).
- VGG-16 → Best at detecting ships (only 4 false predictions in Ship class).
- Common Misclassifications:
  - Cloud vs. Sea – frequent confusion due to spectral similarity.
  - Land vs. Coast – overlaps caused misclassifications.
- SimpleNet2021 → Failed to detect ships reliably, mostly misclassifying clouds/land.

### Large Scene Evaluation (2000×2000 px images)
- MobileNet-V2 → Best generalisation, correctly detected most ships with minimal false positives, though it missed near-shore vessels.
- NASNetMobile & EfficientNetB0 → Detected many ships but high false positives.
- ResNet-50, VGG-16, Inception-V3 → Despite strong accuracy metrics, showed poor scene-level performance with excessive false positives.
- Key Takeaway: Accuracy on small crops ≠ robust detection on full-scene satellite images.


<p align="center">
  <img width="534" height="777" alt="image" src="https://github.com/user-attachments/assets/2ad5423a-278e-4bb6-8e47-bb756477aa57" />
  <br>
  <em>Figue 3. Output results of the first inferences of CNN models on large scene images without optimisation. MobileNet-V2demonstrated superior generalisation with no false positives but missed some near-shore ships. Models like ResNet-50 and Inception-V3 displayed many false positives, indicating poor generalisation despite high accuracy metrics in confusion matrix evaluations.</em>
</p>

### Optimisation (HPO + NMS)
- Hyperparameter Optimisation (HPO): Improved convergence but varied by model.
- Non-Maximum Suppression (NMS): Reduced redundant bounding boxes and false positives.
- Best F1-Scores after optimisation:
  - Inception-V3 → 61.5%
  - ResNet-50 → 58.3%
  - EfficientNetB0 → 61.1%
- Trade-offs: Higher accuracy models required long training/inference times (e.g., ResNet-50 took ~4.5h for one scene image).



<p align="center">
  <img width="565" height="804" alt="image" src="https://github.com/user-attachments/assets/e51dff63-1c94-4e99-91f9-fb94204e5565" />
  <br>
  <em>Figure 5. Visual outputs of CNN models after applying hyperparameter optimisation (HPO) and Non-Maximum Suppression (NMS). Red bounding boxes indicate detections with a 100% confidence level. Models like MobileNet-V2, EfficientB0, and Inception-V3 showed significant reductions in false positives, while DenseNet-121 struggled to generalise effectively.</em>
</p>


<p align="center">
  <img width="674" height="302" alt="image" src="https://github.com/user-attachments/assets/5618f391-22b3-43c7-9074-7511e275974a" />
  <br>
  <em>Table 2. Performance metrics of CNN models after applying hyperparameter optimisation (HPO) and Non-Maximum Suppression (NMS). These metrics provide a comprehensive evaluation of each model’s effectiveness post-optimisation.</em>
</p>

### Testing with MASATI-V2 Dataset
- On MASATI-V2 (larger dataset) → All models improved, with ResNet-50 achieving perfect scores.
- Cross-dataset testing (trained on custom dataset, tested on MASATI-V2) → Performance dropped significantly, showing that dataset size and diversity are critical.
- Ship Class Focus → High precision across most models, confirming robustness for detecting ships even with dataset constraints.

### ✅ Overall Takeaway:
- CNNs can achieve high classification accuracy on small cropped patches, but generalisation to full-scene images remains challenging.
- MobileNet-V2 was the most practical candidate for deployment due to its balance of accuracy, efficiency, and generalisation.
- Larger, more diverse datasets (like MASATI-V2) significantly improve CNN performance, underscoring the importance of dataset quality over model complexity.


## 📊 Results & Discussion – YOLOv11-Based Approach
### Overall Performance
YOLOv11 was evaluated for ship detection in Sentinel-2 imagery, focusing on precision, recall, F1-score, and mean Average Precision (mAP). Both a default YOLOv11 model and a fine-tuned version were tested.
- Default YOLOv11: Achieved respectable results with mAP = 78.05%, but struggled with precise localisation at higher IoU thresholds.
- Fine-Tuned YOLOv11: Outperformed the default, reaching mAP = 85.51%, showing better adaptability and generalisation.

### Training Insights
- Loss curves (Box, Class, DFL) showed consistent decreases, indicating stable learning.
- mAP50 improved steadily to ~98% during training, while mAP50–95 peaked at ~23%, reflecting challenges in very strict localisation.
- Precision rose rapidly after epoch 6, peaking around 76%, while recall improved more gradually to ~64%. Fine-tuning helped balance both metrics.

### Test Results
- Best operating threshold: IoU = 0.5, confidence > 0.3.
- Confusion Matrix (Fine-Tuned model):
  - True Positives (TP): 53
  - False Positives (FP): 10
  - False Negatives (FN): 6
  - True Negatives (TN): 1
- Achieved high recall (88.1%), critical for maritime surveillance where missing ships is more costly than over-detecting.

<p align="center">
  <img width="529" height="760" alt="image" src="https://github.com/user-attachments/assets/d4dfe33d-ff39-4ef1-b516-81c0229e550e" />
  <br>
  <em>Figure 6. Training and validation performance metrics over epochs for Yolov11n model with data augmentation and fine-tuning the parameters and hyperparameters. The plot shows the training loss (top-left), validation loss (top-right), mAP at IoU=0.50 and mAP at IoU=0.50-0.95 (bottom) during the training process. Compared to Figure 4.7, the loss curves are smoother and the mAP values are higher.</em>
</p>

<p align="center">
  <img width="515" height="351" alt="image" src="https://github.com/user-attachments/assets/0d3e046a-27a8-4699-9aa2-4caca8e9da1e" />
  <br>
  <em>Figure 7. Precision and Recall curves during training, showing the model’s performance over epochs. The blue curve represents the Precision score, while the red curve represents the Recall score. The curves highlights the model’s ability to correctly classify positive samples (Precision) and identify all relevant positive samples (Recall) evolve throughout the training process.</em>
</p>

<p align="center">
  <img width="510" height="694" alt="image" src="https://github.com/user-attachments/assets/31d36f11-a774-4733-b5c3-ccd7cf3943b2" />
  <br>
  <em>Figure 8. Comparison of performance metrics across IoU thresholds (0.10 to 1.00) for the Default YOLOv11 Configuration and Fine-Tuned YOLOv11 Training. The graphs illustrate the variations in key evaluation metrics, including precision, recall, F1-score, accuracy and AP, as a function of increasing IoU thresholds.</em>
</p>

<p align="center">
  <img width="635" height="624" alt="image" src="https://github.com/user-attachments/assets/c25b7da8-80e1-44ec-9cbc-051f084860c7" />
  <br>
  <em>Figure 9. In test images where ships were detected, the model had a tendency of misclassifying cloud patterns as ships, indicated with yellow arrows. The figure also illustrates examples where relatively small ships were not detected (green bounding boxes) and closely packed and ships near the shore being missed (cyan bounding box)</em>
</p>


### Strengths
- Robust against false detections over land, coasts, reefs, and in cloudy/foggy conditions.
- Correctly detected small ships with visible wakes and partial ships cut by image boundaries.
- Demonstrated practical real-time detection potential for maritime domain awareness.

### Limitations
- Missed small ships unless visually distinct (e.g., red vessels).
- Struggled with closely spaced ships and harbour/shoreline clutter.
- Occasionally misclassified cloud patterns as ships.
- Multiple bounding boxes sometimes drawn around the same ship (NMS inefficiencies).
- Challenges due to dataset constraints: limited size, low resolution, and lack of true negatives in early experiments.

### Key Takeaways
- YOLOv11 generalised better than CNNs for full-scene images, offering strong recall and adaptability.
- Fine-tuning significantly improved performance across IoU thresholds, stabilising precision and recall.
- Despite limitations with small objects and dataset size, YOLOv11 showed clear promise as a practical, real-time framework for ship detection in satellite imagery.

**Summary: Fine-Tuned YOLOv11 achieved 85.5% mAP, strong recall, and robustness in complex maritime scenes, outperforming traditional CNNs in both efficiency and adaptability.**


## ⚖️ Comparative Analysis – CNN vs YOLOv11
### Architectural Differences
- **CNN Approach:**
  - Worked on 49×49 image chips, classifying them into Ship, Sea, Land, Coast, or Cloud.
  - Required multiple steps → classification → merging → bounding box placement.
  - Accurate on small chips, but weak when generalising to large full-scene images.

- **YOLOv11 Approach:**
  - Processes entire images in one pass, predicting both class and bounding box simultaneously.
  - Optimised for real-time detection with fewer preprocessing steps.
  - More effective in cluttered maritime environments (clouds, reefs, coasts).


### Dataset Requirements
- CNN: Needed thousands of small, manually cropped patches, with multiple classes.
- YOLOv11: Used 2000×2000 sub-tiles annotated with bounding boxes.
- Result: YOLO dataset was more scalable and context-rich.

### Quantitative Performance (Scene-Level)

<p align="center">
  <img width="833" height="218" alt="image" src="https://github.com/user-attachments/assets/bb33454c-8bac-4393-847c-695950dccbca" />
  <br>
  <em>Table 4. Performance metrics of better performing CNN models versus YOLOv11 fine-tuned model. YOLOv11 model showed better performance across all metrics. *The inference time for YOLO model is over one test image as opposed to over one chip for CNN models.</em>
</p>

### Qualitative Observations
- **CNN Limitations:**
  - Overfitting (ResNet-50).
  - Frequent false positives in scene images.
  - Missed near-shore and harbour ships due to background clutter.
  - Confusion between Cloud vs Sea, Land vs Coast.

- **YOLOv11 Strengths:**
  - Fewer false detections in land/coastal/cloudy regions.
  - Better detection of small ships and partially visible ships.
  - More robust in cluttered, real-world maritime scenes.
  - Still struggled with:
    - Misclassifying certain cloud patterns as ships.
    - Detecting closely spaced vessels.
    - Occasional duplicate bounding boxes.

### Efficiency & Scalability
- CNN: Heavy preprocessing (tiling), slow inference, higher memory needs.
- YOLOv11: Entire image processed at once, faster inference and fewer epochs to converge.

### Application Suitability
- CNN: Useful in controlled environments with well-defined classes (e.g., forensic analysis, classification tasks).
- YOLOv11: Better suited for real-world maritime surveillance and real-time applications (live monitoring, drones, CubeSats).
  

## ✅ Summary:
CNNs achieved high accuracy on small patches but struggled in full-scene detection. YOLOv11 outperformed CNNs across accuracy, recall, and inference speed, making it the more practical and scalable solution for ship detection in Sentinel-2 imagery.

| Feature                 | CNN Approach                               | YOLOv11 Approach                                   |
| ----------------------- | ------------------------------------------ | -------------------------------------------------- |
| **Input Format**        | 49×49 patches, multi-class labels          | 2000×2000 images with bounding boxes               |
| **Architecture**        | Classification → localisation (multi-step) | End-to-end detection (single forward pass)         |
| **Performance (Scene)** | Accuracy 17–30%, Recall ≤ 44%              | Accuracy 77%, Recall 84%                           |
| **Strengths**           | Good at fine-grained classification        | Robust in clutter, real-time detection             |
| **Weaknesses**          | Overfitting, many false positives, slow    | Missed small/closely spaced ships, cloud confusion |
| **Inference Speed**     | 10–16s per chip                            | 0.03s per full image                               |
| **Best Use Case**       | Controlled classification tasks            | Maritime surveillance, real-time monitoring        |



## 🏁 Conclusion & Future Work
### Summary of Findings

This research investigated ship detection in Sentinel-2 imagery over the Mauritian EEZ using two approaches:
1. CNN-based classification (chip-level, multi-class).
2. YOLOv11-based detection (end-to-end, real-time).


### Key findings:
- CNNs (ResNet-50, EfficientNetB0) achieved high validation accuracy on small patches, but struggled to generalise to full-scene images, with frequent misclassifications (e.g., Cloud vs Sea, Land vs Coast) and long inference times.
- YOLOv11 (fine-tuned) consistently outperformed CNNs, delivering higher accuracy, precision, recall, and F1-scores while avoiding false detections over land, reefs, and cloudy regions. It also detected small ships and partially cut-off vessels, making it the more practical solution for real-world maritime surveillance.
- Trade-offs remained: CNNs excelled in controlled classification tasks, while YOLOv11 showed stronger generalisation and scalability but missed some very small or closely spaced ships.

### Contributions
- Built two custom Sentinel-2 datasets tailored for CNN and YOLO.
- Delivered a comparative study highlighting model strengths/limitations.
- Produced a fine-tuned YOLOv11 model adapted for Sentinel-2 imagery.

### Recommendations for Future Work
- Expand Datasets: Larger, more diverse Sentinel-2 collections, crowdsourced or collaborative annotation, and use of newer imagery (post-2021).
- Optimise Inputs: Use smaller, higher-resolution tiles (e.g., 640×640) to preserve ship detail for YOLO.
- Address Class Imbalance: Apply resampling methods like SMOTE.
- Enhance Data Augmentation: Employ advanced techniques (CutMix, MixUp, GAN-based, multi-modal with SAR).
- Hybrid Detection: Use YOLO for coarse detection + CNNs for refined classification.
- Temporal Tracking: Integrate sequential imagery to track ship movement and direction (e.g., YOLOv11-OBB with oriented bounding boxes).
- Improve Interpretability: Develop user-friendly outputs (natural language alerts, interactive maps, confidence-based warnings).
- Integrate Other Sensors: Combine Sentinel-2 with Sentinel-1 SAR for all-weather, day/night monitoring.
- Tackle Limitations: Custom loss functions, attention/transformer mechanisms, adaptive thresholds for small ships and cluttered harbours.
- Foundation Models: Explore DETR, SAM, and DINO for zero/few-shot detection and domain adaptation to maritime surveillance.
  

## ✅ Conclusion:
YOLOv11 emerged as the most practical and scalable solution for Sentinel-2 ship detection, offering real-time potential for Maritime Domain Awareness (MDA). However, future research must expand datasets, enhance interpretability, and integrate multi-sensor approaches to build robust, deployable surveillance systems for small island states like Mauritius.

## 📄 Full Thesis Document
You can access the complete MPhil thesis here:

[**➡️ Download / View Thesis (PDF)**](Thesis.pdf)

---

## © Copyright
© 2025 Sheeksha D. Joyseeree. All rights reserved.

This repository, including the thesis, code, figures, and documentation, is the intellectual property of the author.  
Unauthorized copying, distribution, or use of this material without explicit permission is strictly prohibited.

