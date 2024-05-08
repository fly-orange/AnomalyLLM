import os
import subprocess


ANOMALY_ARCHIVE_DATA_FAMILY_TO_DATA_TYPE = {
    'ECG': [
        'sddb40', 
        'BIDMC1', 
        'CHARISfive', 
        'CHARISten', 
        'ECG', 
        'sddb49', 
        'longtermecg', 
        'ltstdbs30791', 
        'sel840mECG', 
        's20101m', 
        'qtdbSel100', 
        'apneaecg', 
        'STAFFIIIDatabase', 
        'Fantasia'],
    'Gait': [
        'weallwalk', 
        'taichidbS0715Master', 
        'park3m', 
        'gait', 
        'GP711MarkerLFM5z'],
    'PowerDemand': [
        'PowerDemand', 
        'Italianpowerdemand'],
    'EPG': [
        'insectEPG', 
        'Lab2Cmac011215EPG'],
    'ABP': [
        'tiltAPB', 
        'tilt127', 
        'InternalBleeding'],
    'RESP': [
        'resperation'],
    'Acceleration': [
        'WalkingAceleration', 
        'MesoplodonDensirostris'],
    'NASA': [
        'MARS'],
    'AirTemperature': [
        'CIMIS44AirTemperature']
}

ANOMALY_ARCHIVE_DATA_TYPE_TO_DATA_FAMILY = {}
for key, value in ANOMALY_ARCHIVE_DATA_FAMILY_TO_DATA_TYPE.items(): 
    for data_type in value: 
        ANOMALY_ARCHIVE_DATA_TYPE_TO_DATA_FAMILY[data_type] = key

for filename in os.listdir('UCR_Anomaly_FullData'):
    scene_name = filename.split('_')[3]
    for key, value in ANOMALY_ARCHIVE_DATA_TYPE_TO_DATA_FAMILY.items():
        if key in scene_name:
            new_dir = f"UCR_Anomaly_ClassData/{value}"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            subprocess.run(["mv", f"UCR_Anomaly_FullData/{filename}", os.path.join(new_dir, filename) ], check=True)
            break

            