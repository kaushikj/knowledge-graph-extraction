# downlaod pre trained bert model
wget -P ./pretrain/nre https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/pretrain/nre/wiki80_bert_softmax.pth.tar

#download bert
cd pretrain
sh download_bert.sh
cd ..

cd benchmark
#download wiki80
sh download_wiki80.sh
cd ..

python3 setup_python.py