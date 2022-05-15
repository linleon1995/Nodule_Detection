import pickle as cPickle
import logging
import os
import xml.etree.ElementTree as etree
import numpy as np

# from .nodule_structs import RadAnnotation, SmallNodule, NormalNodule, \
#     NoduleRoi, NonNodule, AnnotationHeader
# from .utils import find_all_files

NS = {'nih': 'http://www.nih.gov'}




class NoduleCharstics:
    def __init__(self):
        self.subtlety = 0
        self.internal_struct = 0
        self.calcification = 0
        self.sphericity = 0
        self.margin = 0
        self.lobulation = 0
        self.spiculation = 0
        self.texture = 0
        self.malignancy = 0
        return

    def __str__(self):
        str = "subtlty (%d) intstruct (%d) calci (%d) sphere (%d) " \
              "margin (%d) lob (%d) spicul (%d) txtur (%d) malig (%d)" % (
                  self.subtlety, self.internal_struct, self.calcification,
                  self.sphericity,
                  self.margin, self.lobulation, self.spiculation, self.texture,
                  self.malignancy)
        return str

    def set_values(self, sub, inter, calc, spher, lob, spic, tex, malig):
        self.subtlety = sub
        self.internal_struct = inter
        self.calcification = calc
        self.sphericity = spher
        self.lobulation = lob
        self.spiculation = spic
        self.texture = tex
        self.malignancy = malig
        return


class NoduleRoi:  # is common for nodule and non-nodule
    def __init__(self, z_pos=0., sop_uid=''):
        self.z = z_pos
        self.sop_uid = sop_uid
        self.inclusion = True

        self.roi_xy = []  # to hold list of x,ycords in edgemap(edgmap pairs)
        self.roi_rect = []  # rectangle to hold the roi
        self.roi_centroid = []  # to hold centroid of the roi
        return

    def __str__(self):
        n_pts = len(self.roi_xy)
        str = "Inclusion (%s) Z = %.2f SOP_UID (%s) \n ROI points [ %d ]  ::  " \
              "" % (
            self.inclusion, self.z, self.sop_uid, n_pts)

        if (n_pts > 2):
            str += "[[ %d,%d ]] :: " % (
            self.roi_centroid[0], self.roi_centroid[1])
            str += "(%d, %d), (%d,%d)..." % (
                self.roi_xy[0][0], self.roi_xy[0][1], self.roi_xy[1][0],
                self.roi_xy[1][1])
            str += "(%d, %d), (%d,%d)" % (
                self.roi_xy[-2][0], self.roi_xy[-2][1], self.roi_xy[-1][0],
                self.roi_xy[-1][1])
        else:
            for i in range(n_pts):
                str += "(%d, %d)," % (self.roi_xy[i][0], self.roi_xy[i][1])
        return str


class Nodule:  # is base class for all nodule types (NormalNodule,
    # SmallNodule, NonNodule)
    def __init__(self):
        self.id = None
        self.rois = []
        self.is_small = False

    def __str__(self):
        strng = "--- Nodule ID (%s) Small [%s] ---\n" % (
        self.id, str(self.is_small))
        strng += self.tostring() + "\n"
        return strng

    def tostring(self):
        pass


class NoduleAnnotationCluster():  # to be seen
    def __init__(self):
        self.id = []
        self.z_pos = []
        self.centroid = []  # (x,y) of the centroid
        #  convex hull description
        #   p0 ---- p1
        #   |       |
        #   p2-----p3
        self.convex_hull = []  # [()_0 ()_1 ()_2 ()_3]
        self.convex_hull_with_margin = []
        self.no_annots = 0
        self.nodules_data = []

    def compute_centroid(self):
        self.set_convex_hull()
        xc = 0.5 * (
        self.convex_hull[0][0] + self.convex_hull[3][0])  # (x_min + x_max)/2
        yc = 0.5 * (
        self.convex_hull[0][1] + self.convex_hull[3][1])  # (y_min + y_max)/2
        self.centroid = (xc, yc)
        return self.centroid

    def set_convex_hull(self):
        x_min, x_max = 640, 0
        y_min, y_max = 640, 0

        for nodule in self.nodules_data:
            for roi in nodule.rois:
                for dt_pt in roi.roi_xy:
                    # roi.roi_xy -> [(x,y)]
                    # TODO : finish this loop  #?????????????????????????????
                    x_min = dt_pt[0] if (x_min > dt_pt[0]) else x_min
                    x_max = dt_pt[0] if (x_max < dt_pt[0]) else x_max
                    y_min = dt_pt[1] if (y_min > dt_pt[1]) else y_min
                    y_max = dt_pt[1] if (y_max < dt_pt[1]) else y_max
        self.convex_hull = [(x_min, y_min), (x_max, y_min), (x_min, y_max),
                            (x_max, y_max)]
        w, h = (x_max - x_min), (y_max - y_min)
        x_min = int(x_min - 0.15 * w)
        x_max = int(x_max + 0.15 * w)
        y_min = int(y_min - 0.15 * h)
        y_max = int(y_max + 0.15 * h)
        self.convex_hull_with_margin = [(x_min, y_min), (x_max, y_min),
                                        (x_min, y_max),
                                        (x_max, y_max)]


class NormalNodule(Nodule):
    def __init__(self):
        Nodule.__init__(self)
        self.characteristics = NoduleCharstics()
        self.is_small = False

    def tostring(self):
        strng = str(self.characteristics)
        strng += "\n"

        for roi in self.rois:
            strng += str(
                roi) + "\n"  # str calls __str__ of NoduleRoi's class
            # i.e.converting roi to
        return strng  # string to prepare it for printing(it doesn't print it)


class SmallNodule(Nodule):
    def __init__(self):
        Nodule.__init__(self)
        self.is_small = True

    def tostring(self):
        strng = ''
        for roi in self.rois:
            strng += str(roi) + "\n"
        return strng


class NonNodule(Nodule):
    def __init__(self):
        Nodule.__init__(self)
        self.is_small = True

    def tostring(self):
        strng = ''
        for roi in self.rois:
            strng += str(roi)
        return strng


class RadAnnotation:
    def __init__(self, init=True):
        self.version = None
        self.id = None

        self.nodules = []  # is normalNodule i.e in xml unblindedReadNodule
        # with characteristics info
        self.small_nodules = []  # in xml unblindedReadNodule with no
        # characteristics info
        self.non_nodules = []  # located inside readingSession
        self.initialized = init
        return

    def is_init(self):
        return self.initialized

    def set_init(self, init):
        self.initialized = init
        return

    def __str__(self):
        n_nodules = len(self.nodules)
        n_small_nodules = len(self.small_nodules)
        n_non_nodules = len(self.non_nodules)
        strng = "Annotation Version [%s] Radiologist ID [%s] \n" % (
        self.version, self.id)
        strng += "#Nodules [%d] #SmallNodules [%d] #NonNodules[%d] \n" % (
            n_nodules, n_small_nodules, n_non_nodules)

        if (n_nodules > 0):
            strng += "--- Nodules [%d]---\n" % n_nodules
            for i in range(n_nodules):
                strng += str(self.nodules[i])

        if (n_small_nodules > 0):
            strng += "--- Small Nodules [%d] ---\n" % n_small_nodules
            for i in range(n_small_nodules):
                strng += str(self.small_nodules[i])

        if (n_non_nodules > 0):
            strng += "--- Non Nodules [%d] ---\n" % n_non_nodules
            for i in range(n_non_nodules):
                strng += str(self.non_nodules[i])

        strng += "-" * 79 + "\n"
        return strng


class AnnotationHeader:
    def __init__(
            self):  # 4 elements are not included b/c they don't have data
        # inside
        self.version = None
        self.message_id = None
        self.date_request = None
        self.time_request = None
        self.task_desc = None
        self.series_instance_uid = None
        self.date_service = None
        self.time_service = None
        self.study_instance_uid = None

    def __str__(self):
        str = ("--- XML HEADER ---\n"
               "Version (%s) Message-Id (%s) Date-request (%s) Time-request ("
               "%s) \n"
               "Series-UID (%s)\n"
               "Time-service (%s) Task-descr (%s) Date-service (%s) "
               "Time-service (%s)\n"
               "Study-UID (%s)") % (
                  self.version, self.message_id, self.date_request,
                  self.time_request,
                  self.series_instance_uid, self.time_service, self.task_desc,
                  self.date_service,
                  self.time_service, self.study_instance_uid)
        return str


class IdriReadMessage:
    def __init__(self):
        self.header = AnnotationHeader()
        self.annotations = []


def find_all_files(root, suffix=None):
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
    return res


def parse_dir(dirname, flatten=True, pickle=True):
    assert os.path.isdir(dirname)

    if not flatten:
        return parse_original_xmls(dirname, pickle)

    pickle_file = os.path.join(dirname, 'annotation_flatten.pkl')
    if os.path.isfile(pickle_file):
        logging.info("Loading annotations from file %s" % pickle_file)
        with open(pickle_file, 'r') as f:
            annotations = cPickle.load(f)
        logging.info("Load annotations complete")
        return annotations
    annotations = parse_original_xmls(dirname, pickle)
    annotations = flatten_annotation(annotations)
    if pickle:
        logging.info("Saving annotations to file %s" % pickle_file)
        with open(pickle_file, 'w') as f:
            cPickle.dump(annotations, f)
    return annotations


def parse_original_xmls(dirname, pickle=True):
    pickle_file = pickle and os.path.join(dirname, 'annotation.pkl') or None
    if pickle and os.path.isfile(pickle_file):
        logging.info("Loading annotations from file %s" % pickle_file)
        with open(pickle_file, 'r') as f:
            annotations = cPickle.load(f)
        logging.info("Load annotations complete")
    else:
        logging.info("Reading annotations")
        annotations = []
        xml_files = find_all_files(dirname, '.xml')
        for f in xml_files:
            annotations.append((f))
    if pickle and not os.path.isfile(pickle_file):
        logging.info("Saving annotations to file %s" % pickle_file)
        with open(pickle_file, 'w') as f:
            cPickle.dump(annotations, f)
    return annotations


def parse(xml_filename):
    logging.info("Parsing %s" % xml_filename)
    annotations = []
    # ET is the library we use to parse xml data
    tree = etree.parse(xml_filename)
    root = tree.getroot()
    header = parse_header(root)
    # readingSession-> holds radiologist's annotation info
    for read_session in root.findall('nih:readingSession', NS):
        # to hold each radiologists annotation
        # i.e. readingSession in xml file
        rad_annotation = RadAnnotation()
        rad_annotation.version = \
            read_session.find('nih:annotationVersion', NS).text
        rad_annotation.id = \
            read_session.find('nih:servicingRadiologistID', NS).text

        if rad_annotation.id not in ['2103845659', '-750896469', '540461523', '1428348137']:
            print('error')
        # nodules
        nodule_nodes = read_session.findall('nih:unblindedReadNodule', NS)
        for node in nodule_nodes:
            nodule = parse_nodule(node)
            if nodule.is_small:
                rad_annotation.small_nodules.append(nodule)
            else:
                rad_annotation.nodules.append(nodule)

        # non-nodules
        non_nodule = read_session.findall('nih:nonNodule', NS)
        for node in non_nodule:
            nodule = parse_non_nodule(node)
            rad_annotation.non_nodules.append(nodule)
        annotations.append(rad_annotation)
    return header, annotations


def parse_header(root):
    header = AnnotationHeader()
    # print(root.findall('nih:*', NS))
    resp_hdr = root.findall('nih:ResponseHeader', NS)[0]
    header.version = resp_hdr.find('nih:Version', NS).text
    header.message_id = resp_hdr.find('nih:MessageId', NS).text
    header.date_request = resp_hdr.find('nih:DateRequest', NS).text
    header.time_request = resp_hdr.find('nih:TimeRequest', NS).text
    header.task_desc = resp_hdr.find('nih:TaskDescription', NS).text
    header.series_instance_uid = resp_hdr.find('nih:SeriesInstanceUid', NS).text
    date_service = resp_hdr.find('nih:DateService', NS)
    if date_service is not None:
        header.date_service = date_service.text
    time_service = resp_hdr.find('nih:TimeService', NS)
    if time_service is not None:
        header.time_service = time_service.text
    header.study_instance_uid = resp_hdr.find('nih:StudyInstanceUID', NS).text
    return header


def parse_nodule(xml_node):  # xml_node is one unblindedReadNodule
    char_node = xml_node.find('nih:characteristics', NS)
    # if no characteristics, it is smallnodule  i.e. is_small=TRUE
    is_small = (char_node is None or len(char_node) == 0)
    nodule = is_small and SmallNodule() or NormalNodule()
    nodule.id = xml_node.find('nih:noduleID', NS).text
    if not is_small:
        subtlety = char_node.find('nih:subtlety', NS)
        nodule.characteristics.subtlety = int(subtlety.text)
        nodule.characteristics.internal_struct = \
            int(char_node.find('nih:internalStructure', NS).text)
        nodule.characteristics.calcification = \
            int(char_node.find('nih:calcification', NS).text)
        nodule.characteristics.sphericity = \
            int(char_node.find('nih:sphericity', NS).text)
        nodule.characteristics.margin = \
            int(char_node.find('nih:margin', NS).text)
        nodule.characteristics.lobulation = \
            int(char_node.find('nih:lobulation', NS).text)
        nodule.characteristics.spiculation = \
            int(char_node.find('nih:spiculation', NS).text)
        nodule.characteristics.texture = \
            int(char_node.find('nih:texture', NS).text)
        nodule.characteristics.malignancy = \
            int(char_node.find('nih:malignancy', NS).text)
    xml_rois = xml_node.findall('nih:roi', NS)
    for xml_roi in xml_rois:
        roi = NoduleRoi()
        roi.z = float(xml_roi.find('nih:imageZposition', NS).text)
        roi.sop_uid = xml_roi.find('nih:imageSOP_UID', NS).text
        # when inclusion = TRUE ->roi includes the whole nodule
        # when inclusion = FALSE ->roi is drown twice for one nodule
        # 1.ouside the nodule
        # 2.inside the nodule -> to indicate that the nodule has donut
        # hole(the inside hole is
        # not part of the nodule) but by forcing inclusion to be TRUE,
        # this situation is ignored
        roi.inclusion = (xml_roi.find('nih:inclusion', NS).text == "TRUE")
        edge_maps = xml_roi.findall('nih:edgeMap', NS)
        for edge_map in edge_maps:
            x = int(edge_map.find('nih:xCoord', NS).text)
            y = int(edge_map.find('nih:yCoord', NS).text)
            roi.roi_xy.append([x, y])
        xmax = np.array(roi.roi_xy)[:, 0].max()
        xmin = np.array(roi.roi_xy)[:, 0].min()
        ymax = np.array(roi.roi_xy)[:, 1].max()
        ymin = np.array(roi.roi_xy)[:, 1].min()
        if not is_small:  # only for normalNodules
            roi.roi_rect = (xmin, ymin, xmax, ymax)
            roi.roi_centroid = (
                (xmax + xmin) / 2., (ymin + ymax) / 2.)  # center point
        nodule.rois.append(roi)
    return nodule  # is equivalent to unblindedReadNodule(xml element)


def parse_non_nodule(xml_node):  # xml_node is one nonNodule
    nodule = NonNodule()
    nodule.id = xml_node.find('nih:nonNoduleID', NS).text
    roi = NoduleRoi()
    roi.z = float(xml_node.find('nih:imageZposition', NS).text)
    roi.sop_uid = xml_node.find('nih:imageSOP_UID', NS).text
    loci = xml_node.findall('nih:locus', NS)
    for locus in loci:
        x = int(locus.find('nih:xCoord', NS).text)
        y = int(locus.find('nih:yCoord', NS).text)
        roi.roi_xy.append((x, y))
    nodule.rois.append(roi)
    return nodule  # is equivalent to nonNodule(xml element)


def flatten_annotation(annotation_dict):
    logging.info("Start flatten")
    res = {}
    for annotations in annotation_dict:
        # annotations in each file
        for anno in annotations:
            flatten_nodule(anno.nodules, 'nodules', res)
            flatten_nodule(anno.small_nodules, 'small_nodules', res)
            flatten_nodule(anno.non_nodules, 'non_nodules', res)
    logging.info("Flatten complete")
    return res


def flatten_nodule(nodules, type, result):
    for nodule in nodules:
        for roi in nodule.rois:
            # logging.info(roi)
            sop_uid = roi.sop_uid
            # logging.info(sop_uid)
            # logging.info(result)
            if not result.has_key(sop_uid):
                result[sop_uid] = {
                    'nodules': [], 'small_nodules': [], 'non_nodules': []
                }
            centroid = type == 'nodules' and roi.roi_centroid or roi.roi_xy[0]
            point = {'centroid': centroid, 'pixels': roi.roi_xy, 'field': roi.roi_rect}
            result[sop_uid][type].append(point)

if __name__ == '__main__':
    f = rf'D:\Leon\Datasets\LIDC-IDRI'
    # parse_dir(f)
    from data.data_utils import get_files
    f_list = get_files(f, 'xml')
    for idx, f in enumerate(f_list):
        header, annotations = parse(rf'D:\Leon\Datasets\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-30178\3000566.000000-03192\069.xml')
        print(f'{idx}/{len(f_list)}')
        # print(30*'==')