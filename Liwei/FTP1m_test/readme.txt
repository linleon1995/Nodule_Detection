Modified Date: 2021/12/14
Author: Li-Wei Hsiao
mail: nfsmw308@gmail.com
=========================================================================
生成所有資料庫的csv檔，包含nodule中心座標。
安裝:
pip install connected-components-3d

run:
python util.py

參數設定:
save_dataset_csv():
    1. area_th : 設定要被忽略的nodule大小
    2. dataset_path : 資料庫位置
    3. save_csv_path : csv檔儲存位置

========================================================================
可視化 Nodule 的 segmentation mask。

run:
python dataset.py

參數設定:
save_path : 要把可視化結果存在哪個資料夾
TrainingLuna2dSegmentationDataset : 可視化nodule中心點的那個slice
Luna2dSegmentationDataset : 可視化每片slice或是只有nodule的slice

========================================================================
驗證Luna16訓練好的模型效果:
1.將參數修改成與訓練時一樣，且要指定模型路徑
2.可設定要不要存下可視化結果

run:
python test.py

========================================================================
Demo 未來沒有標註的CT影像，或是驗證模型的好壞:
1.要指定raw的mhd檔位置
2.如果有mask可以比對，就要設定mask的mhd檔位置，若沒有，在mask-mhd-path的default寫None

run:
python demo.py