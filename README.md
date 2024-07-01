# Improving Prompt Tuning-based Software Vulnerability Assessment by Fusing Source Code and Vulnerability Description

This is the source code to the paper "Improving Prompt Tuning-based Software Vulnerability Assessment by Fusing Source Code and Vulnerability Description". Please refer to the paper for the experimental details.

## Approach
![](https://github.com/1-001/PT-SVA/blob/main/Fig/framework.png)
## About dataset.
Due to the large size of the datasets, we have stored them in Google Drive: [Dataset Link](https://drive.google.com/drive/folders/1P42XsDWeMqAW33oS0gGamXEqxYiMjO5i?usp=drive_link)

if you want to use the original dataset(MegaVul), you can download it from the following link:https://github.com/Icyrockton/MegaVul

We provide a [code file](https://github.com/1-001/PT-SVA/blob/main/data%20crawling%20and%20processing/scrape_CVSS_v3.py) for crawling ``CVSS v3`` data, and on this basis, you can crawl other data you need.
## Requirements
You can install the required dependency packages for our environment by using the following command: ``pip install - r requirements.txt``.

## Reproducing the experiments:
1.Use the py file under ``data crawling and processing`` for data processing. Of course, you can directly use the ``dataset`` we have processed: [Google Drive Link](https://drive.google.com/drive/folders/1P42XsDWeMqAW33oS0gGamXEqxYiMjO5i?usp=drive_link)

2.Run ``prompt_code&desc.py``. After running, you can retrain the ``model`` and obtain results.

3.You can find the implementation code for the ``RQ1-RQ4`` section and the ``Discussion`` section experiments in the corresponding folders. The ``results`` obtained from the experiment are also in the ``corresponding folder``.

## About model.
You can obtain our ``saved model`` and reproduce our results through the link:[Model Link](https://drive.google.com/file/d/1RdWlH40EgAkyJ4QNGWwH1ZiQe1qGgG06/view?usp=sharing).
