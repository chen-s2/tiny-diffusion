import subprocess

def extract_stats_from_dataset():
    dataset_dir_path = './data/cifar10_unzipped/train'
    dataset_stats_out_path = './data/cifar10_unzipped/train_stats'
    process = subprocess.run("python -m pytorch_fid --save-stats " + dataset_dir_path + " " + dataset_stats_out_path, shell=True, capture_output=True, text=True)

def test_fid_of_dataset_vs_cifar(dataset_dir_path):
    cifar_extracted_fid_stats_path = './data/cifar10_unzipped/train_stats.npz'
    process = subprocess.run("python -m pytorch_fid " + dataset_dir_path + " " + cifar_extracted_fid_stats_path)

def test_fid_of_two_dataests(dataset_path_1, dataset_path_2):
    process = subprocess.run("python -m pytorch_fid " + dataset_path_1 + " " + dataset_path_2)

# test_fid_of_dataset_vs_cifar("./data/butterfly_tiny")
# test_fid_of_dataset_vs_cifar("./data/cifar10_tiny")
cifar_extracted_fid_stats_path = './data/cifar10_unzipped/train_stats.npz'

cifar10_tiny_noise = "./data/cifar10_tiny_noise"
cifar10_tiny_truck = "./data/cifar10_tiny_truck"
cifar10_tiny_truck2 = "./data/cifar10_tiny_truck2"
cifar10_horse = "./data/cifar10_horse"
butterfly_tiny = "./data/butterfly_tiny"
cifar10_first1k = "./data/cifar10_first1k"
cifar10_test_10k_custom = "./data/cifar10_unzipped/test_10k_custom"
butterfly_train_6k = "../dit/data/train/butterfly"
# test_fid_of_two_dataests(cifar_extracted_fid_stats_path, cifar10_test_10k_custom)
test_fid_of_two_dataests(cifar_extracted_fid_stats_path, butterfly_train_6k)

# datasets = [butterfly_tiny,cifar10_tiny_truck,cifar10_tiny_truck2,cifar10_tiny_noise, cifar10_horse]
#
# for dataset1 in datasets:
#     print("data1:", dataset1, ", data2:", cifar_extracted_fid_stats_path)
#     test_fid_of_two_dataests(dataset1, cifar_extracted_fid_stats_path)

# datasets = [butterfly_tiny,cifar10_tiny_truck,cifar10_tiny_truck2,cifar10_tiny_noise, cifar10_horse]
# for dataset1 in datasets:
#     for dataset2 in datasets:
#         print("data1:", dataset1, ", data2:", dataset2)
#         test_fid_of_two_dataests(dataset1, dataset2)
