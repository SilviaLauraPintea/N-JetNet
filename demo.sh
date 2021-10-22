DATASET=cifar10
EPOCHS=90
MODEL='nin_shared_srf'
OPTIM='sgd'
LR=0.1

# SRF parameters
INIT_K=2.0
INIT_SCALE=0.0 # Not used if learned
INIT_ORDER=4.0

python3 demo.py \
    -c checkpoints/cifar10/${MODEL} \
    -a ${MODEL} \
    --data ${DATASET} \
    --epochs ${EPOCHS} \
    --optim ${OPTIM} \
    --lr ${LR} \
    --srf-init-k ${INIT_K} \
    --srf-init-scale ${INIT_SCALE} \
    --srf-init-order ${INIT_ORDER} \

