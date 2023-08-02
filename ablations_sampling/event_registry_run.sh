device=0
data=../data/event_registry_data/
prior=../prior/event_registry/uniform/
batch=16
n_head=4
n_layers=4
d_model=128
d_inner=64
d_k=64
d_v=64
dropout=0.1
lr=1e-4
epoch=300
num_samples=2
event_interest=1
threshold=-0.5
log=event_register_unif6.txt


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main_event_registry.py -data $data -prior $prior -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -epoch $epoch -num_samples $num_samples -event_interest $event_interest -threshold $threshold -log $log
