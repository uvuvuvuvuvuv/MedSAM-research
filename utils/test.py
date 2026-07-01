import pydicom
import os


def check_dicom_series(dicom_folder):
    """检查DICOM系列完整性"""
    files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm') or f.endswith('.DCM')]
    print(f"找到 {len(files)} 个DICOM文件")

    series_dict = {}
    for file in files:
        try:
            ds = pydicom.dcmread(os.path.join(dicom_folder, file))
            series_uid = ds.SeriesInstanceUID
            if series_uid not in series_dict:
                series_dict[series_uid] = []
            series_dict[series_uid].append({
                'file': file,
                'series': getattr(ds, 'SeriesDescription', 'Unknown'),
                'instance': getattr(ds, 'InstanceNumber', 0)
            })
        except Exception as e:
            print(f"读取文件 {file} 失败: {e}")

    return series_dict


# 使用示例
dicom_folder = "你的DICOM文件夹路径"
series_info = check_dicom_series(dicom_folder)
for series_uid, files in series_info.items():
    print(f"系列: {files[0]['series']}")
    print(f"文件数量: {len(files)}")
    print(f"实例号范围: {min(f['instance'] for f in files)} - {max(f['instance'] for f in files)}")