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



🛰️ Background


- Monitoring maritime activity in EEZs is essential for safety, security, and resource management.
- Traditional surveillance (patrol vessels, radars) is expensive and limited in coverage.
- Satellite imagery + AI offers scalable and cost-effective monitoring.
