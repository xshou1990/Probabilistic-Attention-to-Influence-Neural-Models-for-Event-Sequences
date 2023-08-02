device=0
data=data/linkedin_data/
prior=prior/linkedin/uniform/
batch=32
n_head=8
n_layers=4
d_model=256
d_inner=128
d_k=128
d_v=128
dropout=0.1
lr=5e-5
epoch=300
num_samples=1
event_interest=1
threshold=0.5
log=linkedin_log.txt



CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main_linkedin_infoset.py -data $data -prior $prior -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -epoch $epoch -num_samples $num_samples -event_interest $event_interest -threshold $threshold -log $log

