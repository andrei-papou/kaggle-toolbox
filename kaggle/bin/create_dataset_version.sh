version_message="$1"
dataset_zip_path=kaggle/dataset/kaggle_toolbox.zip

zip -r $dataset_zip_path kaggle_toolbox -x '*__pycache__*' && \
kaggle datasets version -m "$version_message" -p $(pwd)/kaggle/dataset && \
rm $dataset_zip_path
