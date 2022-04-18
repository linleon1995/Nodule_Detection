import slicerio
import json
import nrrd
import numpy as np
import matplotlib.pyplot as plt


def nrrd_write(arr):
    data = np.linspace(1, 60, 60).reshape((3, 10, 2))
    header = {'kinds': ['domain', 'domain', 'domain'], 
              'units': ['mm', 'mm', 'mm'], 
              'spacings': [1.0458, 1.0458, 2.5], 
              'space': 'right-anterior-superior', 
              'space directions': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 
              'encoding': 'ASCII', 
              'custom_field_here1': 24.34, 
              'custom_field_here2': np.array([1, 2, 3, 4])}
    custom_field_map = {'custom_field_here1': 'double', 'custom_field_here2': 'int list'}

    nrrd.write('output.nrrd', data, header, custom_field_map=custom_field_map)


def read_seg_nrrd(filename):
    segmentation_info = slicerio.read_segmentation_info(filename)
    data, header = nrrd.read(filename)
    for i in header.items():
        print(i)

    # for i in range(data.shape[2]):
    #     img = data[...,i]
    #     if np.sum(img)>0:
    #         plt.imshow(img)
    #         plt.show()

    # segment_names_to_labels = [("ribs", 10), ("right lung", 12), ("left lung", 6)]
    segment_names_to_labels = [('Segment_1', 1), ('test_abc', 3)]
    extracted_voxels, extracted_header = slicerio.extract_segments(data, header, segmentation_info, segment_names_to_labels)
    nrrd.write('outSegmentation2.seg.nrrd', extracted_voxels, extracted_header)

    # nrrd.write('output_test.nrrd', data, header)

    # number_of_segments = len(segmentation_info["segments"])
    # print(f"Number of segments: {number_of_segments}")

    # segment_names = slicerio.segment_names(segmentation_info)
    # print(f"Segment names: {', '.join(segment_names)}")

    # segment0 = slicerio.segment_from_name(segmentation_info, segment_names[0])
    # print("First segment info:\n" + json.dumps(segment0, sort_keys=False, indent=4))


def seg_nrrd_write(voxels, direction, origin, cmap, space):
    seg_header = build_nrrd_seg_header(voxels, direction, origin, cmap, space)
    nrrd.write('outSegmentation2.seg.nrrd', voxels, seg_header)
    

def build_nrrd_seg_header(arr, direction, origin, cmap, space='left-posterior-superior'):
    header = {
        'type': 'unsigned char',
        'dimension': arr.ndim,
        'space': space,
        'sizes': arr.shape,
        'space directions': direction,
        'kinds': ['domain', 'domain', 'domain'],
        'encoding': 'gzip',
        'space origin': origin,
    }
    header.update(build_common_custom_field())
    
    for idx, id in enumerate(np.unique(arr)[1:]):
        data = np.uint8(arr==id)
        color = cmap[idx%len(cmap)]
        seg_header = build_segment_custom_field(id, color, data)
        header.update(seg_header)
    return header


def build_common_custom_field():
    conversion_params = 'Collapse labelmaps|1|Merge the labelmaps \
                        into as few shared labelmaps as possible \
                        1 = created labelmaps will be shared if possible \
                        without overwriting each other.&Compute surface \
                        normals|1|Compute surface normals. 1 (default) = \
                        surface normals are computed. 0 = surface normals \
                        are not computed (slightly faster but produces less \
                        smooth surface display).&Crop to reference image \
                        geometry|0|Crop the model to the extent of reference \
                        geometry. 0 (default) = created labelmap will contain \
                        the entire model. 1 = created labelmap extent will be \
                        within reference image extent.&Decimation \
                        factor|0.0|Desired reduction in the total \
                        number of polygons. Range: 0.0 (no decimation)\
                        to 1.0 (as much simplification as possible). \
                        Value of 0.8 typically reduces data set size \
                        by 80% without losing too much details.&Fractional \
                        labelmap oversampling factor|1|Determines the \
                        oversampling of the reference image geometry. \
                        All segments are oversampled with the same value \
                        (value of 1 means no oversampling).&Joint \
                        smoothing|0|Perform joint smoothing.&Oversampling \
                        factor|1|Determines the oversampling of the \
                        reference image geometry. If it\'s a number, \
                        then all segments are oversampled with the \
                        same value (value of 1 means no oversampling). If it has the value "A", then automatic oversampling is calculated.&Reference image geometry|-0.64453125;0;0;166;0;-0.64453125;0;133.600006103516;0;0;2.49990081787109;74.0438003540039;0;0;0;1;0;511;0;511;0;130;|Image geometry description string determining the geometry of the labelmap that is created in course of conversion. Can be copied from a volume, using the button.&Smoothing factor|0.5|Smoothing factor. Range: 0.0 (no smoothing) to 1.0 (strong smoothing).&Threshold fraction|0.5|Determines the threshold that the closed surface is created at as a fractional value between 0 and 1.&'
    
    common_custom_field = {
        'Segmentation_ContainedRepresentationNames': 'Binary labelmap|',
        'Segmentation_ConversionParameters': conversion_params,
        'Segmentation_MasterRepresentation': 'Binary labelmap',
        'Segmentation_ReferenceImageExtentOffset': '0 0 0',
    }
    return common_custom_field


def build_segment_custom_field(id: int, color: tuple, data: np.array) -> dict:
    key = f'Segment{id-1}'
    color = f'{color[0]} {color[1]} {color[2]}'

    assert data.ndim == 3
    zs, ys, xs = np.where(data)
    extent = []
    for d in [xs, ys, zs]:
        extent.append(str(np.min(d)))
        extent.append(str(np.max(d)))

    segment_custom_field = {
        f'{key}_Color': color,
        f'{key}_ColorAutoGenerated': '0',
        f'{key}_Extent': extent.join(' '),
        f'{key}_ID': f'Segment_{id}',
        f'{key}_LabelValue': id,
        f'{key}_Layer': '0',
        f'{key}_Name': f'Segment_{id}',
        f'{key}_NameAutoGenerated': '0',
        f'{key}_Tags': 'Segmentation.Status:inprogress|TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~SCT^272673000^Bone~^^~Anatomic codes - DICOM master list~^^~^^|',
    }
    return segment_custom_field


def main():
    path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\plot\nrrd\Segmentation2.seg.nrrd'
    path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\plot\nrrd\Segmentation_multi.seg.nrrd'
    path2 = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\new_malignant\1m0024\1m0024raw\Segmentation.seg.nrrd'
    # nrrd_write(0)
    read_seg_nrrd(path)


if __name__ == '__main__':
    main()