

<div align="center">

<samp>

<h2> Revisiting Distillation for Continual Learning on Visual Question Localized-Answering in Robotic Surgery </h1>

<h4> Long Bai*, Mobarakol Islam*, Hongliang Ren </h3>

<h3> Medical Image Computing and Computer Assisted Intervention (MICCAI) 2023 </h2>

</samp>   

| **[[```arXiv```](<https://arxiv.org/abs/2307.12045>)]** | **[[```Paper```](<https://link.springer.com/chapter/10.1007/978-3-031-43996-4_7>)]** |
|:-------------------:|:-------------------:|

---

</div>     

If you find our code or paper useful, please cite the paper as

```bibtex
@inproceedings{bai2023revisiting,
  title={Revisiting Distillation for Continual Learning on Visual Question Localized-Answering in Robotic Surgery},
  author={Bai, Long and Islam, Mobarakol and Ren, Hongliang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={68--78},
  year={2023},
  organization={Springer}
}
```

---

## Abstract
The visual-question answering (VQA) system can serve as a knowledgeable assistant in surgical education. Nevertheless, various similar surgical instruments and actions can further confuse learners. To tackle this, we introduce an extended visual-question localized-answering (VQLA) system which can provide the answer with localization, to further help the learners understand the surgical scene. However, deep neural networks (DNNs) suffer from catastrophic forgetting when learning new knowledge. Specifically, when DNNs learn on incremental classes or tasks, their performance on old tasks drops dramatically. Furthermore, due to medical data privacy and licensing issues, it is often difficult to access old data when updating continual learning (CL) models. Therefore, we develop a non-exemplar continual surgical VQLA framework, to explore and balance the rigidity-plasticity trade-off of DNNs in a sequential learning paradigm. We revisit the distillation loss in CL tasks, and propose rigidity-plasticity-aware distillation (RP-Dist) and self-calibrated heterogeneous distillation (SH-Dist) to preserve the old knowledge. The weight aligning (WA) technique is also integrated to adjust the weight bias between old and new tasks. We further establish a CL framework on three public surgical datasets in the context of surgical settings that consist of overlapping classes between old and new surgical VQLA tasks. With extensive experiments, we demonstrate that our proposed method excellently reconciles learning and forgetting on the continual surgical VQLA over conventional CL methods.

---
## Environment

- PyTorch
- numpy
- pandas
- scipy
- scikit-learn
- timm
- transformers
- h5py

## Directory Setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch library, the structure is as follows: 

- `checkpoints/`: Contains trained weights.
- `data/`
    - `EndoVis-18-VQLA/` : seq_{1,2,3,4,5,6,7,9,10,11,12,14,15,16}. Each sequence folder follows the same folder structure. 
        - `seq_1`: 
            - `left_frames`: Image frames (left_frames) for each sequence can be downloaded from EndoVIS18 challange.
            - `vqla`
                - `label`: Q&A pairs and bounding box label.
                - `img_features`: Contains img_features extracted from each frame with different patch size.
                    - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
        - `...`
        - `seq_16`
    - `EndoVis-17-VQLA/` : selected frames from EndoVIS17 challange for external validation. 
        - `train`: 
            - `left_frames`
            - `vqla`
                - `label`: Q&A pairs and bounding box label.
                - `img_features`: Contains img_features extracted from each frame with different patch size.
                    - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
        - `test`:
            - `...`
    - `m2cai2016-VQLA/` : selected frames from EndoVIS17 challange for external validation. 
        - `train`: 
            - `left_frames`
            - `vqla`
                - `label`: Q&A pairs and bounding box label.
                - `img_features`: Contains img_features extracted from each frame with different patch size.
                    - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
        - `test`:
            - `...`
    - `featre_extraction/`:
        - `feature_extraction_EndoVis18-VQLA-resnet`: Used to extract features with ResNet18 (based on patch size).
- `models/`: 
    - base.py : Basic continual learning network.
    - continual.py : Our proposed continual learning network.
    - evaluation.py : Evalution functions for different datasets and combination.
    - VisualBertPrediction.py : VisualBERT encoder-based model for VQLA task.
- `utils/`: 
    - data_manager.py : Data management for continual learning setup.
    - inc_net.py : Continual learning network setup.
    - linears.py : Linear layer definition for the classifier layer.
    - toolkit.py : Toolkit functions for continual learning.
    - vqla.py : Utils functions for VQLA tasks.
- `dataloader/`: 
    - dataloader.py
- train.py

---

## Run training
- Train
    ```bash
    python train.py --config ./exp/csvqla.json
    ```
- Training Configuration Setup:
    ```bash
    "prefix": "reproduce",
    "shuffle": true,
    "model_name": "csvqla",
    "convnet_type": "visualbert",
    "device": ["0"],
    "init_epoch": 60,
    "init_lr": 0.00001,
    "epochs": 25,
    "lrate": 0.00005,
    "T_overlap": 25,
    "T_old": 20,
    "checkpoint_dir": "checkpoints/",
    "batch_size": 64
    ```

---
