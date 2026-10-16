from dataset_preprocessing.CAMUS.data_preparation import create_camus_npz

general_path='<path_to_CAMUS>/CAMUS_public/'
create_camus_npz(
    patient_root=general_path+"database_nifti",
    train_split=general_path+"database_split/subgroup_training.txt",
    val_split=general_path+"database_split/subgroup_validation.txt",
    test_split=general_path+"database_split/subgroup_testing.txt",
    output_file="<path_to_CAMUS_preprocessing>/camus_echonet_style_mobi.npz",
    target_size=(112, 112),
)