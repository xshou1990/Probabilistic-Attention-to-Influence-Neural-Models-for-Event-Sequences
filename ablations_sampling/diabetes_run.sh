device=1
data=../data/diabetes_data/
prior=../prior/diabetes/uniform/
batch=16
n_head=8
n_layers=4
d_model=128
d_inner=64
d_k=64
d_v=64
dropout=0.1
lr=2e-4
epoch=300
num_samples=2
event_interest=1
threshold=-0.5
log=diabetes_log.txt


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -prior $prior -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -epoch $epoch -num_samples $num_samples -event_interest $event_interest -threshold $threshold -log $log

