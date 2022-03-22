Modified Date: 2021/12/14
Author: Li-Wei Hsiao
mail: nfsmw308@gmail.com

================================================================================
設備:
GPU:  2070 SUPER 8GB
CPU:   i7 9700 @3.00Hz
RAM:  32GB

================================================================================
比較重要的套件，其它最新版即可:
pip install diskcache==4.1

================================================================================
可視化Luna 16 Nodule 的 segmentation mask:
run:
python vis_nodule.py

參數設定:
save_path : 要把可視化結果存在哪個資料夾
TrainingLuna2dSegmentationDataset : 可視化nodule中心點的那個slice
Luna2dSegmentationDataset : 可視化每片slice或是只有nodule的slice

================================================================================
可視化Luna 16 lung 的 segmentation mask:
run:
python vis_lung.py

參數設定:
filename : raw檔
filename_mask : 對應raw的mask檔
save_path : 要把可視化結果存在哪個資料夾

================================================================================
訓練模型:
1.先將資料庫放到指定的資料夾底下，路徑參考dataset_seg.py的 37-40 行
2.若有更新dataset_seg.py的生成mask程式碼，記得刪除該目錄底下的"data-unversioned"資料夾
3.在train.py設定好參數後，就可以訓練了
run:
python train.py

驗證模型:
1.將參數修改成與訓練時一樣，可設定是否要存下模型預測結果
run:
python valid.py