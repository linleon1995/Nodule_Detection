import os
from data.volume_generator import asus_nodule_volume_generator, lidc_nodule_volume_generator
from dataset_conversion.crop_data_utils import save_center_info, build_crop_data
# TODO: parser
# Malignancy

def main():
    CROP_RANGE =  {'index': 32, 'row': 64, 'column': 64}    
    NEGATIVE_POSITIVE_RATIO = 10
    CONNECTIVITY = 26
    CENTER_SHIFT = False
    SHIFT_STEP = 0

    # RAW_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess'
    VOL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-preprocess'
    DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    MASK_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-preprocess\masks_test\3'
    vol_data_path = os.path.join(VOL_DATA_PATH, 'crop')
    ANNOTATION_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data\annotations.csv'

    # VOLUME_GENERATOR = asus_nodule_volume_generator(DATA_PATH)
    # if not os.path.isfile(ANNOTATION_PATH):
    #     save_center_info(
    #         volume_generator=VOLUME_GENERATOR, connectivity=CONNECTIVITY, save_path=ANNOTATION_PATH)
    
    VOLUME_GENERATOR = lidc_nodule_volume_generator(DATA_PATH, MASK_PATH)
    # TODO: data_path & voluume generator are repeat info, should only input one
    build_crop_data(data_path=DATA_PATH,
                    crop_range=CROP_RANGE, 
                    vol_data_path=vol_data_path, 
                    volume_generator_builder=VOLUME_GENERATOR, 
                    annotation_path=ANNOTATION_PATH, 
                    negative_positive_ratio=NEGATIVE_POSITIVE_RATIO,
                    center_shift=CENTER_SHIFT,
                    shift_step=SHIFT_STEP)
  

if __name__ == '__main__':
    main()