# MPhil-Research-Project
My research project for MPhil degree
🚢 Ship Detection in the Exclusive Economic Zone of Mauritius in Sentinel-2 Imagery using CNNs & Yolov11

🙏 Acknowledgements


This project wouldn’t have been possible without the incredible people who supported me along the way.
First, I’m grateful to my supervisors — Assoc. Prof. Heeralall-Issur N., Prof. Cullen D., and Dr. Beeharry Y. — for their steady guidance, thoughtful feedback, and patience throughout the research journey. To my family — my mother Anita, my brothers Ranveer, Yashwant, and Vrisht, and my sister-in-law Youri — thank you for believing in me and cheering me on through every challenge. My friends Heerveen, Dinesh, and Nawshine also deserve special mention for their encouragement and for keeping me motivated when things got tough. A big thanks as well to Ruben Louis for his timely technical support. I’m also thankful to the DARA (Development in Africa with Radio Astronomy) scholarship, which funded and enabled this research, and to the examiners whose detailed feedback pushed me to refine my approach and reach better results. A special shoutout goes to new supporters I met along the way, like Sofia, whose encouragement during a challenging transition helped me stay the course. And finally — yes, I’ll admit it — even ChatGPT deserves a thank-you. Its help with debugging, research, and drafting made the process faster (and less painful).


📌 Overview


Maritime surveillance is important for national security, economic resilience, and environmental protection. For island nations like Mauritius, whose Exclusive Economic Zone (EEZ) spans an impressive 2.3 million km², the third largest in Africa and the 25th largest worldwide, safeguarding maritime borders is not only a matter of sovereignty but also of survival. The Mauritian EEZ lies across busy international shipping routes connecting Asia to Africa and beyond, with vessels frequently carrying hazardous cargo and large volumes of goods. This geo-strategic position presents unique challenges, including illegal fishing, piracy, narcotics trafficking, marine pollution, and accident prevention.
  

Traditional surveillance methods such as naval patrols and radar monitoring, while effective, have limited coverage. Satellite imagery offers a scalable and cost-effective alternative, providing continuous monitoring over vast ocean areas. Both optical imagery (visible spectrum) and Synthetic Aperture Radar (SAR) have proven valuable. Optical imagery captures fine details of ships during clear conditions, while SAR enables detection at night or through clouds. However, extracting meaningful insights from these large volumes of satellite data remains a challenge.


Conventional ship detection techniques often rely on handcrafted features and classical classifiers, which can be brittle under variations in ship size, orientation, or atmospheric conditions. Recent advances in deep learning, particularly Convolutional Neural Networks (CNNs) and real-time detectors such as YOLO, provide a promising alternative. These models excel at learning complex patterns directly from data, offering improved robustness and accuracy compared to traditional methods.


This study leverages Sentinel-2 optical satellite imagery, chosen for its free accessibility, 10-meter resolution, and five-day revisit frequency — making it suitable for monitoring ships traversing the Mauritian EEZ, where a cargo vessel typically requires six days to cross its full extent. Using Sentinel-2 data ensures not only affordability but also the ability to build a custom dataset tailored to local conditions, including coral reefs, “white water” from wave turbulence, and frequent cloud cover.


The research specifically evaluates and compares state-of-the-art CNN architectures (such as VGG-16, ResNet-50, Inception-V3, DenseNet, EfficientNet, and MobileNet) against the YOLOv11 framework. By applying transfer learning, the models were fine-tuned on a custom dataset, enabling effective training despite limited labeled data. Performance was measured in terms of precision, recall, F1-score, and inference speed, balancing accuracy against computational efficiency. Beyond technical performance, the study also explores broader implications:
- How well deep learning models generalize to other datasets and geographic regions.
- The trade-offs between high accuracy and real-time detection needs.
- The interpretability of model outputs for end-users like maritime authorities.
- The potential role of satellite-based ship detection systems in enhancing Maritime Domain Awareness (MDA) in small developing states.



To summarise, this research demonstrates that CNNs and YOLO offer significant potential for affordable, scalable, and effective maritime surveillance in Mauritius and similar contexts. While challenges remain in terms of dataset availability, computational costs, and real-world deployment, the results highlight a viable path toward integrating AI-driven ship detection into national defense and environmental monitoring strategies.



Available overview of optical satellite datasets utilised for ship detection tasks.

This collection highlights the diversity of datasets used in maritime object detection, ranging from high-resolution commercial data to publicly available data with varying revisit times and resolutions. 

<img width="545" height="720" alt="image" src="https://github.com/user-attachments/assets/2868b308-bcf1-46f2-a05f-824ea73b8bbc" />



🛰️ Sentinel-2 Satellite


The Sentinel-2 mission, operated by the European Space Agency (ESA), is a medium- to high-resolution Earth observation program designed for environmental monitoring, land use analysis, and maritime surveillance. It consists of twin satellites in the same orbit, phased 180° apart, ensuring a revisit time of five days at the Equator. Each satellite is equipped with the Multi-Spectral Instrument (MSI), capturing data in 13 spectral bands: four at 10 m, six at 20 m, and three at 60 m resolution, across a swath width of 290 km.


Sentinel-2 data are freely available and provide a cost-effective alternative to commercial imagery, which can be prohibitively expensive for small island states. For this research, Sentinel-2 was chosen as the optimal balance of spatial resolution, temporal frequency, and accessibility, making it well suited for monitoring large maritime areas such as Mauritius’ Exclusive Economic Zone (EEZ).


The satellites collect data over land, coastal regions, and islands worldwide, including inland water bodies and closed seas. Data are provided in two main product levels:
- Level-1C: Top-Of-Atmosphere (TOA) reflectance, orthorectified for geometric accuracy. (~600 MB per 100×100 km² tile)
- Level-2A: Bottom-Of-Atmosphere (BOA) reflectance, atmospherically corrected for surface-level analysis. (~800 MB per tile)


Both products are distributed as standardised 100×100 km² ortho-image tiles in the Universal Transverse Mercator (UTM/WGS84) projection, ensuring spatial consistency and seamless integration with other geospatial datasets.


The UTM projection is critical for precise vessel detection, as it allows Sentinel-2 imagery to align with maritime boundaries, AIS vessel data, and port infrastructure. For Mauritius, whose vast EEZ spans multiple UTM zones, this ensures reliable monitoring of shipping lanes, vessel movements, and potential threats.


<img width="728" height="386" alt="image" src="https://github.com/user-attachments/assets/2bf1711b-a57b-497d-a917-cba24b96f325" />


📚 Literature Review: Advances in Ship Detection


Ship detection in satellite imagery has evolved from traditional image processing methods with handcrafted features (HOG, SIFT) to deep learning-based approaches that learn robust patterns directly from data.


🔹 Convolutional Neural Networks (CNNs)
- CNNs introduced automated feature extraction, making them far more effective than traditional methods in noisy maritime environments.
- Two-stage CNN frameworks (e.g., R-CNN family) achieve high precision but are computationally expensive.
- Single-stage CNN frameworks (e.g., SSD, MobileNet variants) are more efficient, enabling near real-time applications.
- Specialised innovations such as multi-scale feature fusion and attention mechanisms improved detection across diverse ship sizes and complex backgrounds.


🔹 YOLO (You Only Look Once)
- YOLO revolutionised real-time detection by combining classification and localisation in a single forward pass.
- Recent versions (YOLOv5–YOLOv11) integrate transformer modules, spatial attention, and multi-scale detection, making them highly effective in maritime contexts.
- Studies applying YOLO to Sentinel-2 imagery have achieved strong precision/recall trade-offs, making it suitable for operational monitoring.


🔹 Foundation Models & Emerging Trends
- Foundation models (DETR, DINO, Florence-2) leverage large-scale pretraining to generalise across tasks with minimal labelled data.
- They capture global context, making them robust in noisy environments with clouds, waves, and occlusions.
- Limitations include high computational costs, domain adaptation challenges, and interpretability issues.


🔹 Key Insights
- CNNs excel in accuracy but are resource-heavy.
- YOLO balances speed and accuracy, ideal for real-time ship monitoring.
- Foundation models offer scalability and few-shot learning but require heavy compute.
- Research highlights a trade-off between precision and efficiency, shaping how models are chosen for maritime surveillance.


📂 Dataset Creation for CNN


Since no ready-to-use Sentinel-2 ship detection dataset was available, a custom dataset was built from scratch using Sentinel-2 optical imagery collected between March 2016 and March 2021.


<img width="700" height="479" alt="image" src="https://github.com/user-attachments/assets/81b797dd-6867-4580-939b-7d6daf57ef3c" />


Steps Taken
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


<img width="758" height="608" alt="image" src="https://github.com/user-attachments/assets/169844a3-2bd7-4a0e-a38b-85676b01712f" />


- Annotation Challenges:
  -  Many annotation tools didn’t support large JPEG2000 files or raw band stacking.
  -  QGIS was chosen as the annotation tool since it could process the raw bands directly and generate raster layers without quality loss.

Why this approach?
- Unlike large public datasets (e.g., xView, DIOR), this dataset was tailored specifically to Sentinel-2 imagery and the Mauritian EEZ context.
- Instead of fine-grained ship classification (e.g., tankers, fishing vessels), the focus was on distinguishing ships from common maritime backgrounds like sea, land, clouds, and coastlines.


📂 Dataset Creation for YOLO


For the YOLO-based approach, a custom dataset was created from Sentinel-2 imagery (March 2019 – March 2021) to enable real-time ship detection.

Steps Taken
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


Why this approach?


- Unlike the CNN dataset (cropped 49×49 patches), YOLO requires full images with bounding boxes, enabling it to learn both localisation and classification in one pass.
- Using sub-tiles ensured that ships remained visible without overwhelming GPU memory.
- A single-class setup simplified the pipeline while directly aligning with the research goal: detecting ships in Sentinel-2 imagery.
