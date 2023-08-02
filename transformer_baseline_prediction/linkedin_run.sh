device=0
data=../data/linkedin_data/
batch=16
n_head=4
n_layers=4
d_model=256
d_inner=128
d_k=128
d_v=128
dropout=0.1
lr=1e-4
epoch=300
log=linkedin_log.txt


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -epoch $epoch -log $log

