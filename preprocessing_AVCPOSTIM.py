import os
import subprocess
import nibabel as nib

OUTPUT_DIR = "./preprocessed_without_RAS"  # Assicurati che questa directory esista
MNI_TEMPLATE = "./MNI152_T1_1mm.nii.gz"  # Percorso al template MNI

def morph_op(input_filename, output_filename, op):
    cmd1 = f"./animaMorphologicalOperations -i {input_filename}  -o {output_filename} -a {op}"
    result1 = subprocess.run(cmd1, shell=True)
    if result1.returncode != 0:
        raise RuntimeError(f"Error in animaMorphologicalOperations: {cmd1}")
    return output_filename


def fillHole(input_filename, output_filename):
    cmd1 = f"./animaFillHoleImage -i {input_filename}  -o {output_filename}"
    result1 = subprocess.run(cmd1, shell=True)
    if result1.returncode != 0:
        raise RuntimeError(f"Error in animaFillHoleImage: {cmd1}")
    return output_filename


def skull_strip(input_filename, output_filename):
    """
    Esegue lo skull stripping utilizzando HD-BET.
    """
    os.system(f"hd-bet -i {input_filename} -o {output_filename} -device cpu --disable_tta")
    return output_filename

def register_to_reference(input_filename, output_filename, reference_input, mask_input=None):
    """
    Registra un'immagine di input a un riferimento usando Anima.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    subject = input_filename.split("/")
    subject_folder = os.path.join(OUTPUT_DIR, subject[2])
    anat_folder = os.path.join(subject_folder, 'anat')
    seg_folder = os.path.join(subject_folder, 'seg')
    
    os.makedirs(anat_folder, exist_ok=True)
    os.makedirs(seg_folder, exist_ok=True)
    
    #output_filepath = os.path.join(anat_folder, output_filename)
    
    #transform_aff = os.path.join(anat_folder, f"{output_filename}_aff.txt")
    #transforms_xml = os.path.join(anat_folder, f"{output_filename}_transforms.xml")
    #resampled_mask_output = os.path.join(seg_folder, f"{subject_folder}_mask.nii.gz")
    
    transform_aff = output_filename.replace('.nii.gz', "_aff.txt")
    transforms_xml = output_filename.replace('.nii.gz', "_transforms.xml")
    
    resampled_mask_output = output_filename.replace('skullstripped_N4_T1.nii.gz', "reg.nii.gz").replace('skullstripped_N4_MNI.nii.gz', "reg.nii.gz").replace('anat', 'seg')
    
    print(transform_aff)
    print(transforms_xml)

    print(output_filename)
    print(f"--- Registering {input_filename} to {reference_input} ---")
    
    cmd1 = f"./animaPyramidalBMRegistration -m {input_filename} -r {reference_input} -o {output_filename} -O {transform_aff}"
    result1 = subprocess.run(cmd1, shell=True)
    if result1.returncode != 0:
        raise RuntimeError(f"Error in animaPyramidalBMRegistration: {cmd1}")
    
    cmd2 = f"./animaTransformSerieXmlGenerator -i {transform_aff} -o {transforms_xml}"
    result2 = subprocess.run(cmd2, shell=True)
    if result2.returncode != 0:
        raise RuntimeError(f"Error in animaTransformSerieXmlGenerator: {cmd2}")
    
    if mask_input:
        cmd3 = f"./animaApplyTransformSerie -i {mask_input} -g {reference_input} -t {transforms_xml} -n nearest -o {resampled_mask_output}"
        result3 = subprocess.run(cmd3, shell=True)
        if result3.returncode != 0:
            raise RuntimeError(f"Error in animaApplyTransformSerie: {cmd3}")
        print(resampled_mask_output)
        morph_op(resampled_mask_output, resampled_mask_output.replace('.nii.gz', '_clos.nii.gz'), "clos")
        fillHole(resampled_mask_output, resampled_mask_output.replace('.nii.gz', '_fh.nii.gz'))
    return output_filename 

def bias_correct(input_filename, output_filename):
    os.system(f"./animaN4BiasCorrection -i {input_filename} -o {output_filename}")
    return output_filename


def reorient_RAS(input_filename, output_filename):
    img = nib.load(input_filename)
    reoriented_img = nib.as_closest_canonical(img) # Reorient to RAS
    nib.save(reoriented_img, output_filename)
    return output_filename


def process_subject(subject_data):
    """
    Processa i dati di un soggetto registrando T1 a MNI, quindi T2 e FLAIR a T1.
    """
    t1_path = subject_data['RawData'].get('T1')
    t2_path = subject_data['RawData'].get('T2')
    flair_path = subject_data['RawData'].get('FLAIR')
    
    t1_mask = subject_data['Derivatives'].get('T1')
    t2_mask = subject_data['Derivatives'].get('T2')
    flair_mask = subject_data['Derivatives'].get('FLAIR')
    
    if not t1_path:
        print("❌ T1 mancante, impossibile procedere con la registrazione.")
        return
    
    subject = t1_path.split("/")[2]
    anat_folder = os.path.join(OUTPUT_DIR, subject, 'anat')
    os.makedirs(anat_folder, exist_ok=True)
    
    skull_stripped_t1 = os.path.join(anat_folder, os.path.basename(t1_path).replace('.nii.gz', '_skullstripped.nii.gz'))
    skull_strip(t1_path, skull_stripped_t1)

    n4_output_t1 =  skull_stripped_t1.replace('.nii.gz', '_N4.nii.gz')
    bias_correct(skull_stripped_t1, n4_output_t1)

    reg_t1 = n4_output_t1.replace('.nii.gz', '_MNI.nii.gz')
    final_t1 = register_to_reference(n4_output_t1, reg_t1, MNI_TEMPLATE, t1_mask)
    

    if t2_path:
        skull_stripped_t2 = os.path.join(anat_folder, os.path.basename(t2_path).replace('.nii.gz', '_skullstripped.nii.gz'))
        skull_strip(t2_path, skull_stripped_t2)
        
        n4_output_t2 = skull_stripped_t2.replace('.nii.gz', '_N4.nii.gz')
        bias_correct(skull_stripped_t2, n4_output_t2)


        reg_t2 = n4_output_t2.replace('.nii.gz', '_T1.nii.gz')
        final_t2 = register_to_reference(skull_stripped_t2, reg_t2, final_t1, t2_mask)
    
    if flair_path:
        skull_stripped_flair = os.path.join(anat_folder, os.path.basename(flair_path).replace('.nii.gz', '_skullstripped.nii.gz'))
        skull_strip(flair_path, skull_stripped_flair)

        n4_output_flair = skull_stripped_flair.replace('.nii.gz', '_N4.nii.gz')
        bias_correct(skull_stripped_flair, n4_output_flair)

        reg_flair = n4_output_flair.replace('.nii.gz', '_T1.nii.gz')
        final_flair = register_to_reference(skull_stripped_flair, reg_flair, final_t1, flair_mask)


    



def find_files(base_folder, folder_type):
    """
    Trova i file di imaging nelle cartelle rawdata e derivatives.
    """
    results = {}
    for root, _, files in os.walk(base_folder):
        relative_path = os.path.relpath(root, base_folder).replace("/anat", "/seg")
        t1_file = next((f for f in files if 'T1' in f), None)
        t2_file = next((f for f in files if 'T2' in f), None)
        flair_file = next((f for f in files if 'FLAIR' in f), None)
        if t1_file or t2_file or flair_file:
            results[relative_path] = {
                'T1': os.path.join(root, t1_file) if t1_file else None,
                'T2': os.path.join(root, t2_file) if t2_file else None,
                'FLAIR': os.path.join(root, flair_file) if flair_file else None
            }
    return results

def compare_folders(rawdata_folder, derivatives_folder):
    """
    Confronta rawdata e derivatives per associare le immagini alle rispettive maschere.
    """
    files_rawdata = find_files(rawdata_folder, "rawdata")
    files_derivatives = find_files(derivatives_folder, "derivatives")
    combined_results = {}
    all_subfolders = set(files_rawdata.keys()).union(set(files_derivatives.keys()))
    for subfolder in all_subfolders:
        combined_results[subfolder] = {
            "RawData": files_rawdata.get(subfolder, {"T1": None, "T2": None, "FLAIR": None}),
            "Derivatives": files_derivatives.get(subfolder, {"T1": None, "T2": None, "FLAIR": None}),
        }
    return combined_results

rawdata_path = "./rawdata"
derivatives_path = "./derivatives"
result_dict = compare_folders(rawdata_path, derivatives_path)
for subfolder, data in result_dict.items():
    print(f"\n=== Sottocartella: {subfolder} ===")
    process_subject(data)
