# Cancer_Tumor_Detection_on_Gigapixel_Pathology_Images

## Introduction

**Applied Deep Learning Project**

Reproduce and improve the methods of the paper [Detecting Cancer Metastases on Gigapixel Pathology Images](https://arxiv.org/abs/1703.02442)

---
## Project Structure

- Please make sure you have the following directories before running the project 

```bash
.
├── README.md
├── config.py
├── create_model.py
├── docs
│   └── Project_slides.pdf
├── generate_samples.ipynb
├── markdown-toc-tree.sh
├── model
├── pipeline_base.ipynb
├── pipeline_one_zoom.ipynb
├── pipeline_three_zooms_357.ipynb
├── pipeline_three_zooms_567.ipynb
├── plot_heatmap.py
├── processed_data
│   ├── negative
│   └── positive
├── processing.py
├── project_starter_code.ipynb
├── raw_data
├── result
├── test_model.ipynb
└── video.txt
```

- *config.py* has the necessary folder paths, including data, model and result
- *processing.py* has common functions for data processing and data augmentation.
- *create_model.py* is to create three models used in this project.
- *plot_heatmap.py* has the functions used to plot heatmap.

--- 
## How to run this project

0) **Run *project_starter_code.ipynb* to learn about the original data.**

1) **Run *generate_samples.ipynb* to generate training samples.**  
All samples will be saved as .npy files, which are very big. Please make sure you have enough space to save samples. if you don't want to save samples on your local, you can take step 2 directly.

2) **Run *pipeline_xxx.ipynb* to train the model and plot the heatmap.**
*pipeline_three_zooms_357.ipynb* has the code to generate samples directly, which is not required to saving samples on local.

3) **Run *test_model.ipynb* to test the model.**
I have a separate file to test models because I hope to draw the ROC curves of models together.

You can change the parameters in the files to generate you own samples and create you own models.

---
## Others

- Youtube video: https://www.youtube.com/watch?v=4w0jdRBDIYI
- Project slides: [slides](https://github.com/Alex2Yang97/Cancer_Tumor_Detection_on_Gigapixel_Pathology_Images/blob/main/docs/Project_slides.pdf)
