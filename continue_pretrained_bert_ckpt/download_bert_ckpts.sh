# For downloading our continue-pretrained Bert model from our Google Drive
pip install gdown

# Use the file id to download `cont_pretrained_bert.ckpt_3`
gdown 1blbETqovLIVCjVfFl3cccnm9YyxmXUyE

# If run bash in the main direcotry, this will move it to the correct folder
mv cont_pretrained_bert.ckpt_3 ./continue_pretrained_bert_ckpt