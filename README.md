# **TrueColor**  

*Restore the past, enhance the future*   


#### Google Cloud Storage 
Command to upload the dataset from local to GCS:   

gsutil -m cp -r /media/axelrom16/Axel/TrueColor/Data/bw_images gs://image-restoration-dataset   
gsutil -m cp -r /media/axelrom16/Axel/TrueColor/Data/color_images gs://image-restoration-dataset    

gsutil -m cp -r /media/axelrom16/Axel/TrueColor/Data/train.csv gs://image-restoration-dataset   
gsutil -m cp -r /media/axelrom16/Axel/TrueColor/Data/val.csv gs://image-restoration-dataset    
gsutil -m cp -r /media/axelrom16/Axel/TrueColor/Data/test.csv gs://image-restoration-dataset
