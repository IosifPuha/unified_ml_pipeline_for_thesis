
This repository contains the code of the final Apache Airflow DAG developed to create a pipeline with the objective of using EffNet-L2 and ViT-H/14 for inference on multiple datasets

# HOW TO USE

### directory structure

To execute the pipeline you should have the following directory structure:
* `dags`
    * `unified_ml_inference.py`
* `input`
    * `ir_seg_dataset`
    * `sar-dataset`
* `output`
    * `Multispectral`
        * `model_checkpoint.pth`
        * `checkpoint-255`
   * `SAR`
        * `model_checkpoint.pth`
        * `checkpoint-7500`
    

Where `ir_seg_dataset` and `sar-dataset are`, intuitively, the directories of the two datasets, whereas the two `model_checpoint.pth` files are the checkpoints for the EfficientNet-L2 models and the checkpoints for the ViT-H/14 models are saved in another format (using the `checkpoint-255` and `checkpoint-7500` directories).

If you want these files contact me, because they are too big to stay on GitHub, so I didn't put it in the repo.

### execution
start the scheduler with `airflow scheduler`, start the web server, for example with `airflow webserver --port 8080` if you do this and go on the localhost:8080 of your browser you will see a new DAG, you can execute it by simply triggering it with the button
