from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Param
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.dummy_operator import DummyOperator

from datetime import datetime, timedelta
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_FOLDER = os.path.join(BASE_DIR, 'input/ir_seg_dataset/images')
LABELS_FOLDER = os.path.join(BASE_DIR, 'input/ir_seg_dataset/labels')
SAR_DATA_ROOT = os.path.join(BASE_DIR, 'input/sar-dataset')
MODEL_PATH_SAR_EFFNET = os.path.join(BASE_DIR, 'output', 'SAR/model_checkpoint.pth')
MODEL_PATH_MULTISPECTRAL_EFFNET = os.path.join(BASE_DIR, 'output', 'Multispectral/model_checkpoint.pth')
MODEL_PATH_SAR_VIT = os.path.join(BASE_DIR, 'output', 'SAR/checkpoint-7500')
MODEL_PATH_MULTISPECTRAL_VIT = os.path.join(BASE_DIR, 'output', 'Multispectral/checkpoint-255')
OUTPUT_PATH = os.path.join(BASE_DIR, 'output')
TEMP_DIR = os.path.join(BASE_DIR, 'temp') # Serve per il dataset SAR, ma la definiamo all'inizio perché è comodo avercela poi per il task di pulizia 


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'unified_classification_pipeline',
    default_args=default_args,
    description='Unified Classification Pipeline for SAR and Multispectral datasets',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={'dataset_type': Param("Multispectral", enum=["SAR", "Multispectral"]), 'max_images': Param(40, type="integer", minimum=1, maximum=1000), 'architecture': Param("EfficientNet-L2", enum=["ViT-Huge/14", "EfficientNet-L2"])} 
)

# Funzione per decidere quale dataset utilizzare
def choose_dataset_type(**kwargs):
    dataset_type = kwargs['dag_run'].conf.get('dataset_type', kwargs['params'].get('dataset_type'))
    if dataset_type == 'SAR':
        return 'prepare_data_SAR'  
    elif dataset_type == 'Multispectral':
        return 'prepare_data_Multispectral'  
    else:
        raise ValueError("Invalid dataset type. Choose 'SAR' or 'Multispectral'.")

branch_task = BranchPythonOperator(
    task_id='choose_dataset_type',
    python_callable=choose_dataset_type,
    dag=dag
)


def prepare_data_SAR(noise_factor=1, **kwargs):
    import zipfile
    import random
    import numpy as np
    import shutil  # per copiare i file senza aprirli
    from PIL import Image

    max_images = kwargs['dag_run'].conf.get('max_images', kwargs['params'].get('max_images'))

    image_paths, labels = [], []
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Iterazione su tutti i file zip nella cartella SAR_DATA_ROOT
    for zip_name in os.listdir(SAR_DATA_ROOT):
        zip_path = os.path.join(SAR_DATA_ROOT, zip_name)
        
        if zipfile.is_zipfile(zip_path):
            print(f"Processing zip file: {zip_name}")
            with zipfile.ZipFile(zip_path, 'r') as main_zip:
                # Identifichiamo i sub-archivi zip all'interno del file principale
                sub_zips = [f for f in main_zip.namelist() if f.endswith('.zip')]
                
                for sub_zip_name in sub_zips:
                    print(f"  Found sub-zip: {sub_zip_name}")
                    with main_zip.open(sub_zip_name) as sub_zip_file:
                        with zipfile.ZipFile(sub_zip_file) as sub_zip:
                            # Esaminiamo la struttura delle directory all'interno del sub-zip
                            dir_names = [f for f in sub_zip.namelist() if f.endswith('/')]
                            print(f"    Directories in {sub_zip_name}: {dir_names}")

                            # Filtriamo per scegliere solo le directory di interesse
                            directories_of_interest = ['Patch/', 'Patch_Cal/', 'Patch_Uint8/']
                            dirs_to_process = [d for d in dir_names if any(d.endswith(interest_dir) for interest_dir in directories_of_interest)]

                            print(f"    Processing directories: {dirs_to_process}")

                            # Esaminiamo solo i file nelle directory di interesse
                            for img_name in sub_zip.namelist():
                                # Se il file si trova in una delle directory di interesse, lo trattiamo
                                if any(img_name.startswith(d) for d in dirs_to_process):
                                    print(f"      Checking file: {img_name}")
                                    if img_name.endswith('.tif') and len(image_paths) < max_images:
                                        # Percorso temporaneo dove salvare l'immagine
                                        temp_img_path = os.path.join(
                                            TEMP_DIR, f"{zip_name}_{sub_zip_name}_{os.path.basename(img_name)}"
                                        )
                                        print(f"      Copying image to: {temp_img_path}")
                                        # Estraiamo il file .tif direttamente nella cartella temporanea
                                        with sub_zip.open(img_name) as img_file, open(temp_img_path, 'wb') as temp_file:
                                            shutil.copyfileobj(img_file, temp_file)
                                        
                                        image_paths.append(temp_img_path)
                                        labels.append(1)  # 1 per immagini di NAVI
                                    if len(image_paths) >= max_images:
                                        break
                    if len(image_paths) >= max_images:
                        break
            if len(image_paths) >= max_images:
                break

    print(f"We found {len(image_paths)} RADAR images.")

    # Creazione di immagini di rumore
    print("Starting to create random noise!")
    for _ in range(len(image_paths) * noise_factor):
        width, height = random.randint(50, 300), random.randint(50, 300)
        noise_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        noise_image_path = os.path.join(TEMP_DIR, f'noise_{_}.png')
        print(f"Creating noise image: {noise_image_path}")
        noise_image_pil = Image.fromarray(noise_image)
        noise_image_pil.save(noise_image_path)
        image_paths.append(noise_image_path)
        labels.append(0)  # 0 per immagini di RUMORE
    kwargs['ti'].xcom_push(key='dataset', value=(image_paths, labels))
    print(f"Prepared dataset with {len(image_paths)} images and {len(labels)} labels.")    
    return "Data preparation completed"

prepare_data_SAR_task = PythonOperator(
    task_id='prepare_data_SAR',
    python_callable=prepare_data_SAR,
    dag=dag
)


def prepare_data_Multispectral(**kwargs):
    import random
    from torchvision import transforms
    from PIL import Image

    max_images = kwargs['dag_run'].conf.get('max_images', kwargs['params'].get('max_images'))
    architecture = kwargs['dag_run'].conf.get('architecture', kwargs['params'].get('architecture'))

    all_filenames = os.listdir(LABELS_FOLDER)
    filenames = random.sample(all_filenames, max_images)
    kwargs['ti'].xcom_push(key='filenames', value=filenames)
    print("Pushed the filenames to XCom")
    # valori di media e deviazione standard presi da varie versioni del notebook
    c_mean=[0.23, 0.27, 0.23, 0.34]
    c_std=[0.22, 0.22, 0.23, 0.15]
    if architecture == 'EfficientNet-L2':
        transform = transforms.Compose([
        transforms.Resize((475, 475)),
        transforms.ToTensor(),
        transforms.Normalize(mean=c_mean, std=c_std)
    ])
    else: # default to ViT
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=c_mean, std=c_std)
    ])
    images = []
    for filename in filenames:
        image_path = os.path.join(IMAGES_FOLDER, filename)
        image = Image.open(image_path)
        image = transform(image)
        images.append(image)
    
    kwargs['ti'].xcom_push(key='processed_images', value=images)
    return "Data loaded, preprocessed and pushed to XCom"

prepare_data_Multispectral_task = PythonOperator(
    task_id='prepare_data_Multispectral',
    python_callable=prepare_data_Multispectral,
    dag=dag
)

# Fare attenzione al branching sul caricamento del modello, perché per il dataset SAR usiamo il modello così com'è (a 3 canali) mentre per l'altro cambiamo l'architettura usando 4 canali
def setup_model(**kwargs):
    import torch
    torch.cuda.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_type = kwargs['dag_run'].conf.get('dataset_type', kwargs['params'].get('dataset_type'))
    architecture = kwargs['dag_run'].conf.get('architecture', kwargs['params'].get('architecture'))

    print("Loading model...")
    
    if architecture == 'EfficientNet-L2':
        import timm
        from timm import create_model
        import torch.nn as nn

        model = create_model('tf_efficientnet_l2.ns_jft_in1k_475', pretrained=False)
        if dataset_type == 'Multispectral':
            model.conv_stem = nn.Conv2d(4, model.conv_stem.out_channels, kernel_size=model.conv_stem.kernel_size, stride=model.conv_stem.stride, padding=model.conv_stem.padding, bias=model.conv_stem.bias)
            model_path = MODEL_PATH_MULTISPECTRAL_EFFNET
        else:  # default to SAR
            model_path = MODEL_PATH_SAR_EFFNET
        model.classifier = nn.Linear(model.classifier.in_features, 2)  # 2 classi generali: 'Presenza' e 'Assenza' dell'oggetto in questione (ship o person)
        print(f"Model will be moved on: {device}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        model_save_path = os.path.join(BASE_DIR, 'output', dataset_type, 'loaded_model.pth')
        torch.save(model, model_save_path)
        print(f"Model for {dataset_type} saved to disk")
        kwargs['ti'].xcom_push(key='model_path', value=model_save_path)
        data_config = timm.data.resolve_model_data_config(model)
        print(data_config)
        test_transforms=timm.data.create_transform(**data_config, is_training=False)
        kwargs['ti'].xcom_push(key='transforms', value=test_transforms)
        return f"Model path '{model_save_path}' and transforms pushed to XCom."
    else: # default to ViT
        from transformers import AutoConfig, AutoModelForImageClassification
        if dataset_type == 'Multispectral':
            config_path = os.path.join(MODEL_PATH_MULTISPECTRAL_VIT, 'config.json')
            model_path = os.path.join(MODEL_PATH_MULTISPECTRAL_VIT, 'model.safetensors')
        else: # default to SAR
            config_path = os.path.join(MODEL_PATH_SAR_VIT, 'config.json')
            model_path = os.path.join(MODEL_PATH_SAR_VIT, 'model.safetensors')
        config = AutoConfig.from_pretrained(config_path)
        model = AutoModelForImageClassification.from_pretrained(model_path, config=config)
        print("Model will be moved on:", device)
        model.eval()
        model.to(device)
        model_save_path = os.path.join(BASE_DIR, 'output', dataset_type, 'loaded_model.pth')
        torch.save(model, model_save_path)
        print(f"Model for {dataset_type} saved to disk")
        kwargs['ti'].xcom_push(key='model_path', value=model_save_path)
        return f"Model path '{model_save_path}' pushed to XCom."

setup_model_task = PythonOperator(
    task_id='setup_model',
    python_callable=setup_model,
    dag=dag
)


def run_inference(**kwargs):
    dataset_type = kwargs['dag_run'].conf.get('dataset_type', kwargs['params'].get('dataset_type'))
    architecture = kwargs['dag_run'].conf.get('architecture', kwargs['params'].get('architecture'))
    model_path = kwargs['ti'].xcom_pull(task_ids='setup_model', key='model_path')
    if not model_path:
        raise ValueError("Model path not found in XCom. Check if `setup_model` task succeeded.")

    print(f"Inference with model: {model_path}")

    import numpy as np
    import torch
    from PIL import Image


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    if dataset_type == 'SAR':
        import rasterio
        image_paths, labels = kwargs['ti'].xcom_pull(task_ids='prepare_data_SAR', key='dataset')
        results = []

        if architecture == "EfficientNet-L2":
            transform = kwargs['ti'].xcom_pull(task_ids='setup_model', key='transforms')
        else: # default to ViT
            import torchvision.transforms as transforms
            from transformers import ViTImageProcessor
            feature_extractor = ViTImageProcessor.from_pretrained("google/vit-huge-patch14-224-in21k")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
            ])
        # Inferenza immagine per immagine 
        with torch.no_grad():
            for img, label in zip(image_paths, labels):
                if isinstance(img, str):
                    with rasterio.open(img) as src:
                        bands = src.read()
                        if bands.shape[0] == 1:
                            image = np.repeat(bands[0, :, :][..., np.newaxis], 3, axis=-1)
                        elif bands.shape[0] == 2:
                            image = np.zeros((bands.shape[1], bands.shape[2], 3), dtype=np.uint8)
                            image[..., 0] = bands[0]
                            image[..., 1] = bands[1]
                        elif bands.shape[0] == 3:
                            image = np.stack((bands[0], bands[1], bands[2]), axis=-1)
                        elif bands.shape[0] == 4:
                            image = np.stack((bands[0], bands[1], bands[2]), axis=-1)
                        else:
                            raise ValueError(f"Unexpected number of bands: {bands.shape[0]}")
                    image = Image.fromarray(image.astype(np.uint8))
                else:
                    image = img
                image = transform(image).unsqueeze(0).to(device)
                output = model(image)
                if architecture == 'EfficientNet-L2':
                    _, predicted = torch.max(output, 1)
                else: # default to ViT
                    logits = output.logits
                    _, predicted = torch.max(logits, 1)
                results.append((img, label, predicted.item()))
            kwargs['ti'].xcom_push(key='inference_results', value=results)

    elif dataset_type == 'Multispectral':
        PERSON_LABEL = 2
        def load_label(label_path):
            return np.array(Image.open(label_path))

        def contains_person(label_image, person_label=PERSON_LABEL):
            return person_label in label_image

        images = kwargs['ti'].xcom_pull(task_ids='prepare_data_Multispectral', key='processed_images')
        filenames = kwargs['ti'].xcom_pull(task_ids='prepare_data_Multispectral', key='filenames')
        
        results = []
        for filename, image in zip(filenames, images):
            # Load true label from label image file
            label_path = os.path.join(LABELS_FOLDER, filename)
            label_image = load_label(label_path)
            true_label = 1 if contains_person(label_image) else 0
            # Predict using model
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                output = model(image)
                if architecture == 'EfficientNet-L2':
                    _, predicted = torch.max(output, 1)
                else: # default to ViT
                    logits = output.logits
                    _, predicted = torch.max(logits, 1)
                predicted_label = predicted.item()
            results.append((filename, true_label, predicted_label))  # Store all details in a tuple
        kwargs['ti'].xcom_push(key='inference_results', value=results)

    else:
        raise ValueError("Invalid dataset type. Choose 'SAR' or 'Multispectral'.")

    return "Inference completed and results with true labels pushed to XCom"

run_inference_task = PythonOperator(
    task_id='run_inference',
    python_callable=run_inference,
    dag=dag
)

# Funzione per salvare i risultati dell'inferenza
def save_results(**kwargs):
    import csv

    results = kwargs['ti'].xcom_pull(task_ids='run_inference', key='inference_results')
    dataset_type = kwargs['dag_run'].conf.get('dataset_type', kwargs['params'].get('dataset_type'))
    OUTPUT_CSV = os.path.join(OUTPUT_PATH, dataset_type, 'inference_results.csv')

    if dataset_type == 'SAR':
        label_map = {1: "ship", 0: "no ship"}
    elif dataset_type == 'Multispectral':
        label_map = {1: "person", 0: "no person"}
    else:
        raise ValueError("Invalid dataset type. Choose 'SAR' or 'Multispectral'.")
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'true_label', 'predicted_label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for filename, true_label, predicted_label in results:
                writer.writerow({'filename': filename, 'true_label': label_map[true_label], 'predicted_label': label_map[predicted_label]})
    return "Results with true and predicted labels saved to CSV"

save_results_task = PythonOperator(
    task_id='save_results',
    python_callable=save_results,
    dag=dag
)

# Task di pulizia per rimuovere i file temporanei
def cleanup_temp_files():
    import shutil
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"Temporary files in {TEMP_DIR} have been deleted.")
    else:
        print(f"No temporary files found in {TEMP_DIR}.")

cleanup_temp_files_task = PythonOperator(
    task_id='cleanup_temp_files',
    python_callable=cleanup_temp_files,
    dag=dag
)


sync_task = DummyOperator(
    task_id='sync_task',
    dag=dag,
    trigger_rule=TriggerRule.ONE_SUCCESS
)

# Definizione delle dipendenze
branch_task >> [prepare_data_SAR_task, prepare_data_Multispectral_task]
[prepare_data_SAR_task, prepare_data_Multispectral_task] >> sync_task
[sync_task, setup_model_task] >> run_inference_task >> save_results_task >> cleanup_temp_files_task