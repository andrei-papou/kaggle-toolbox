dataset_zip_path=kaggle/dataset/kaggle_toolbox.zip

zip -r $dataset_zip_path kaggle_toolbox -x '*__pycache__*' && \
kaggle datasets create -p $(pwd)/kaggle/dataset && \
rm $dataset_zip_path
